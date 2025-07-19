import json
import logging
import pathlib
import re
from collections import defaultdict, deque
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Tuple

import bs4
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from igcs import configs
from igcs.datasets.igcsbench_datasets.base import SourceDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample, SelectionType
from igcs.utils import ioutils, log

logger = logging.getLogger(__name__)


class DebateSumDataset(SourceDataset):
    name = "DebateSum"
    splits_fnames: Dict[str, Tuple[str, ...]] = {
        "train": "train.1000.json",
        "dev": "dev.1000.json",
        "test": "test.1000.json",
    }
    dirname: pathlib.Path = configs.SRC_DATASETS_DIR / "DebateSum"

    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]:
        with open(fname) as fp:
            data = json.load(fp)

        for raw_sample in data:
            selections = []
            for selection_content, selection_start, selection_end in raw_sample["SelectionSpans"]:
                selections.append(
                    Selection(
                        doc_id=0,
                        start_pos=selection_start,
                        end_pos=selection_end,
                        content=selection_content,
                    )
                )

            yield SelectionSample(
                id=str(raw_sample["Id"]),
                source_dataset=self.name,
                selections=[SelectionGroup(selections=selections)],
                docs=[Doc(id=0, text=raw_sample["Full-Document"])],
                instruction_context={"argument": re.sub(r"\s+", " ", raw_sample["Abstract"])},
                source_sample=raw_sample,
                type=SelectionType.span,
            )

    @staticmethod
    def prepare_debatesum_dataset(
        data_dir: str | Path,
        outfile: str | Path = configs.SRC_DATASETS_DIR / "DebateSum-v2.jsonl.gz",
        last_year: int = 2019,  # Original DebateSum used 2013-2019
    ):
        """A function to recreate DebateSum dataset with an additional `SelectionSpans`
        field for exact text indices and `Year` for splitting into train/dev/test splits.

        First, one must download all cards data from - https://opencaselist.com/openev
        Links are of the form: https://caselist-files.s3.us-east-005.backblazeb2.com/openev/2023OpenEv.zip
        Please ensure you agree & comply to the terms of service of the website.
        """
        data_dir = Path(data_dir)
        expected_years = range(2013, last_year + 1)

        # sanity check for data integrity
        logger.info(f"verifying all years exist on {data_dir}")
        year2files = {}
        for year in expected_years:
            year_data_dir = data_dir / f"{year}OpenEv"
            assert year_data_dir.exists()
            year2files[year] = list(glob(str(year_data_dir / "**" / "*.html5"), recursive=True))
            assert year2files[year]

        fnames = sorted(fname for fnames in year2files.values() for fname in fnames)

        seen_card_set = set()
        parsing_stats = defaultdict(int)
        samples = []

        logger.info(f"found {len(fnames)} total html5 files.")
        for fname in tqdm(fnames):
            samples += DebateSumDataset.parse_html_file(
                fname, data_dir, seen_card_set, parsing_stats, last_year
            )

        logger.info("parsing stats:\n" + "\n".join(f"{k}: {v}" for k, v in parsing_stats.items()))
        ioutils.jsonl_dump(samples, outfile)

    @staticmethod
    def parse_html_file(
        fname: str, data_dir: Path, seen_cards: set, parsing_stats: dict, last_year: int
    ) -> list[dict]:
        """This function is based on `make_debate_dataset.py` logic with cosmetic changes and
        additional parsing of text positions of selection.
        """
        # Force the documents to be unique - still possible for near duplicates
        # to filter through though - seems rare in practice.
        with open(fname) as fp:
            html = fp.read()

        data = []

        soup = BeautifulSoup(html, "lxml")
        all_card_tags = soup.find_all("h4")
        logger.debug(f"found {len(all_card_tags)} cards from file {fname}")

        card_id = 0
        for h4 in all_card_tags:
            # extract one card at a time
            try:
                card = h4.find_next("p").find_next("p")
            except AttributeError:
                parsing_stats["skipped.no_p_tag"] += 1
                logger.debug("Skipping card - no <p> tag")
                continue

            if card is None:
                parsing_stats["skipped.empty"] += 1
                logger.debug("Skipping card - empty")
                continue

            abstract = h4.text  # this is the abstract
            abs_length = len(abstract)  # character length of abstract
            abs_word_length = len(abstract.split())
            citation = h4.find_next("p").text  # get the citation information

            full_doc = card.text  # get the full document
            doc_length = len(full_doc)
            doc_word_length = len(full_doc.split())

            # Original DebateSum filters
            if doc_word_length < 20:
                parsing_stats["skipped.too_short"] += 1
                logger.debug("Skipping card - too short")
                continue

            if full_doc in seen_cards:
                parsing_stats["skipped.duplicate"] += 1
                logger.debug("Skipping card - duplicate text already exist")
                continue

            seen_cards.add(full_doc)

            # iterate recursively over elements inside the card and extract text and
            # underlined selections with its positions.
            text_segments = []
            selections = []
            size = 0
            all_elems = deque(card.contents)
            while all_elems and (elem := all_elems.popleft()):
                # case1: u-tag (i.e. <u>)
                if isinstance(elem, bs4.Tag) and elem.name.lower() == "u":
                    etext = elem.text
                    selections.append((etext, size, size + len(etext)))
                    text_segments.append(etext)
                    size += len(etext)

                # case2: non-u tag
                elif isinstance(elem, bs4.Tag):
                    all_elems.extendleft(elem.contents[::-1])

                # case3: a string
                elif isinstance(elem, bs4.NavigableString):
                    etext = str(elem)
                    text_segments.append(etext)
                    size += len(etext)
                else:
                    raise ValueError(f"Unknown element type {type(elem)} - {elem!r}")

            extract = " ".join(selection[0] for selection in selections)
            extract_len = len(extract)
            extract_word_length = len(extract.split())
            compression_ratio_abs = abs_word_length / doc_word_length
            compression_ratio_extract = extract_word_length / doc_word_length

            # Original DebateSum filters
            if extract_word_length <= 10 or abs_word_length <= 3:
                parsing_stats["skipped.too_short"] += 1
                logger.debug("Skipping card - too short extracted text or abstract text")
                continue

            if compression_ratio_abs >= 1 or compression_ratio_extract >= 1:
                parsing_stats["skipped.too_long"] += 1
                logger.debug("Skipping card - too long extracted text or abstract text")
                continue

            # Verify full text is the same to the original logic from DebateSum code
            assert full_doc == "".join(text_segments)
            # Verify selections text spans are correct
            for select in selections:
                assert select[0] == full_doc[select[1] : select[2]]

            # Verify extraction is the same to the original extraction logic from DebateSum code
            original_extract = " ".join(
                [x.text for x in card.find_all("u")]
            )  # should get anything underlined
            assert original_extract == extract

            dir_prefix = str(data_dir) + "/"
            assert fname.startswith(dir_prefix)
            short_name = fname[len(dir_prefix) :]
            year = int(short_name[:4])
            assert year in range(2013, last_year + 1)

            # Add an additional unique Id field
            sample_id = f"{short_name.replace('/', '.').removesuffix('.html5')}.{card_id}"

            # Keep original DebateSum field names
            data.append(
                {
                    "Full-Document": full_doc,
                    "Citation": citation,
                    "Extract": extract,
                    "Abstract": abstract,
                    "#CharsDocument": doc_length,
                    "#CharsAbstract": abs_length,
                    "#CharsExtract": extract_len,
                    "#WordsDocument": doc_word_length,
                    "#WordsAbstract": abs_word_length,
                    "#WordsExtract": extract_word_length,
                    "AbsCompressionRatio": compression_ratio_abs,
                    "ExtCompressionRatio": compression_ratio_extract,
                    "OriginalDebateFileName": short_name,
                    # Added by us
                    "Id": sample_id,
                    "Year": year,
                    "SelectionSpans": selections,
                }
            )
            card_id += 1

        logger.info(f"found {len(data)}/{len(all_card_tags)} valid cards")
        parsing_stats["valid"] += len(data)
        parsing_stats["total"] += len(all_card_tags)
        return data

    @staticmethod
    def split_and_sample_debatesum(
        sample_size: int = 1000,
        debatesum_fname: Path = configs.SRC_DATASETS_DIR / "DebateSum-v2.jsonl.gz",
        out_dir: Path = configs.SRC_DATASETS_DIR / "DebateSum",
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(debatesum_fname, lines=True)
        df.sort_values(["Year"], inplace=True)
        assert df["Id"].is_unique

        # split the years roughly to train 65% / dev 15% / test 20%
        train_split = df[(2013 <= df["Year"]) & (df["Year"] <= 2016)].copy()
        dev_split = df[(2017 <= df["Year"]) & (df["Year"] <= 2017)].copy()
        test_split = df[(2018 <= df["Year"]) & (df["Year"] <= 2019)].copy()

        # Remove overlapping samples
        for ds1, ds2 in [
            (train_split, dev_split),
            (train_split, test_split),
            (dev_split, test_split),
        ]:
            for key in ["Abstract", "Extract", "Full-Document"]:
                unique_vals2 = set(ds2[key])
                contaminated_indices = ds1[ds1[key].isin(unique_vals2)].index
                logger.info(
                    f'dropping {len(contaminated_indices)}/{len(ds1)} with duplicate "{key}" sample'
                )
                ds1.drop(contaminated_indices, axis=0, inplace=True)

        # Contamination test
        for key in ["Abstract", "Extract", "Full-Document", "Id"]:
            assert set(train_split[key]) & set(dev_split[key]) == set()
            assert set(train_split[key]) & set(test_split[key]) == set()
            assert set(test_split[key]) & set(dev_split[key]) == set()

        for split, split_df in [
            ("train", train_split),
            ("dev", dev_split),
            ("test", test_split),
        ]:
            sampled_split_df = split_df.sample(n=sample_size, random_state=42).reset_index()

            # We save the sample in a readable format as it is small and we load it eagerly.
            sampled_split_df.to_json(
                out_dir / f"{split}.{sample_size}.json",
                index=False,
                orient="records",
                indent=2,
            )
            split_df.to_json(out_dir / f"{split}.jsonl.gz", lines=True, orient="records")


if __name__ == "__main__":  # pragma: nocover
    log.init(logging.INFO)
    ds = DebateSumDataset().prepare_dataset()
    sample = ds["dev"][0]
    print(sample)
