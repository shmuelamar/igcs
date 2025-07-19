import logging
import pathlib
import re
from abc import ABC
from collections import defaultdict
from typing import Dict, Iterable

import pandas as pd

from igcs import configs
from igcs.datasets.igcsbench_datasets.base import SourceDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample, SelectionType
from igcs.utils import log

logger = logging.getLogger(__name__)


class SparkBaseDataset(SourceDataset, ABC):
    splits_fnames: Dict[str, str] = {"test": "test.csv"}
    dirname: pathlib.Path = configs.SRC_DATASETS_DIR / "SPARK"

    @staticmethod
    def make_selections_from_alignment(
        topic_id: int,
        doc: Doc,
        doc_span_text: str,
        doc_span_indices: list[tuple[int, int]],
    ) -> list[Selection]:
        doc_spans_texts = doc_span_text.split("<SPAN_SEP>")
        assert len(doc_spans_texts) == len(doc_span_indices)

        logger.debug(
            f"found {len(doc_spans_texts)} spans for topic {topic_id} of total size "
            f"{sum(e - s for s, e in doc_span_indices)}"
        )

        selections = []
        for span_idx, (span_start, span_end) in enumerate(doc_span_indices):
            assert span_start < len(doc.text)
            assert span_end <= len(doc.text)
            content = doc.text[span_start:span_end]
            assert content == doc_spans_texts[span_idx]

            selections.append(
                Selection(
                    doc_id=doc.id,
                    start_pos=span_start,
                    end_pos=span_end,
                    content=content,
                )
            )
        return selections

    def load_topic2docs(self) -> dict[int, list[Doc]]:
        docs_fnames = sorted((self.dirname / "docs").glob("test*/*.txt"))
        topic_to_doc_to_text = defaultdict(dict)

        for fname in docs_fnames:
            topic_id, doc_file = fname.parts[-2:]
            assert re.search(r"^test\d{1,2}$", topic_id) is not None
            topic_id = int(topic_id.removeprefix("test"))
            assert 0 <= topic_id <= 99

            doc_id = self.parse_doc_id(doc_file)
            assert doc_id not in topic_to_doc_to_text[topic_id], "not unique"
            with open(fname) as fp:
                topic_to_doc_to_text[topic_id][doc_id] = fp.read()

        # verify data integrity
        topic2docs = {}
        for topic_id, docs in topic_to_doc_to_text.items():
            num_docs = len(docs)
            assert num_docs == max(docs.keys()) + 1
            assert set(docs.keys()) == set(range(0, num_docs)), "docs not consecutive"
            topic2docs[topic_id] = [Doc(id=i, text=docs[i]) for i in range(num_docs)]

        assert sum(len(docs) for docs in topic2docs.values()) == len(docs_fnames)
        return topic2docs

    @staticmethod
    def load_file(fname: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(fname)
        df["doc_span_indices"] = df["docSpanOffsets"].apply(SparkBaseDataset.parse_offsets)
        df["summary_span_indices"] = df["summarySpanOffsets"].apply(SparkBaseDataset.parse_offsets)
        df["doc_id"] = df["documentFile"].apply(SparkBaseDataset.parse_doc_id)

        df.sort_values(["topic_id", "cluster_idx", "docSpanOffsets"], inplace=True)
        return df

    @staticmethod
    def parse_offsets(offsets: str) -> list[tuple[int, int]]:
        pairs = [o.split(",") for o in offsets.split(";")]
        assert all(len(x) == 2 for x in pairs), offsets
        parsed_offsets = [(int(start.strip()), int(end.strip())) for start, end in pairs]
        assert all(s < e for s, e in parsed_offsets), parsed_offsets
        return parsed_offsets

    @staticmethod
    def parse_doc_id(doc_file: str) -> int:
        assert re.search(r"^[0-5]\.txt$", doc_file) is not None
        return int(doc_file.split(".")[0])


class SparkSaliencyDataset(SparkBaseDataset):
    name = "SaliencyDetection"

    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]:
        df = self.load_file(fname)
        topic2docs = self.load_topic2docs()

        for topic_id, topic_df in df.groupby("topic_id"):
            topic_id = int(topic_id)
            docs = topic2docs[topic_id]

            # convert each alignment to one or more selections (in case non-continuous)
            selections = []
            for _, raw_alignment in topic_df.iterrows():
                alignment_doc_id = raw_alignment["doc_id"]
                selections += self.make_selections_from_alignment(
                    topic_id=topic_id,
                    doc=docs[alignment_doc_id],
                    doc_span_indices=raw_alignment["doc_span_indices"],
                    doc_span_text=raw_alignment["docSpanText"],
                )

            yield SelectionSample(
                id=f"{self.name}.{topic_id}",
                source_dataset=self.name,
                selections=[SelectionGroup(selections=selections)],
                docs=docs,
                instruction_context={},
                source_sample=topic_df.to_dict(orient="records"),
                type=SelectionType.span,
            )


class SparkEvidenceDataset(SparkBaseDataset):
    name = "EvidenceDetection"

    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]:
        df = self.load_file(fname)
        topic2docs = self.load_topic2docs()

        for (topic_id, cluster_idx), topic_df in df.groupby(["topic_id", "cluster_idx"]):
            topic_id = int(topic_id)
            docs = topic2docs[topic_id]

            # convert each alignment to one or more selections (in case non-continuous)
            selections = []
            for _, raw_alignment in topic_df.iterrows():
                alignment_doc_id = raw_alignment["doc_id"]
                selections += self.make_selections_from_alignment(
                    topic_id=topic_id,
                    doc=docs[alignment_doc_id],
                    doc_span_indices=raw_alignment["doc_span_indices"],
                    doc_span_text=raw_alignment["docSpanText"],
                )

            query = topic_df["query"].iloc[0]
            assert set(topic_df["query"]) == {query}
            yield SelectionSample(
                id=f"{self.name}.{topic_id}.{cluster_idx}",
                source_dataset=self.name,
                selections=[SelectionGroup(selections=selections)],
                docs=docs,
                instruction_context={"query": query},
                source_sample=topic_df.to_dict(orient="records"),
                type=SelectionType.span,
            )


if __name__ == "__main__":
    log.init(logging.DEBUG)

    for dtype in [SparkSaliencyDataset, SparkEvidenceDataset]:
        dataset = dtype().prepare_dataset()
        s = dataset["test"][0]
        print(s)
        print({split: len(data) for split, data in dataset.items()})
