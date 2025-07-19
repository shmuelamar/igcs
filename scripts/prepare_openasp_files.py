import argparse
import glob
import hashlib
import logging
import os
import pathlib

import pandas as pd

from igcs import configs
from igcs.datasets.igcsbench_datasets import OpenAspDataset
from igcs.entities import SelectionSample
from igcs.utils import ioutils, log

logger = logging.getLogger(__name__)


def prepare_openasp_files(
    openasp_dir: str | pathlib.Path, igcs_dataset_dir: str | pathlib.Path = configs.IGCS_DATA_DIR
):
    """This function adds OpenAsp's licensed data to the given IGCS dataset files by using the OpenAsp dataset.
    To use OpenAsp dataset - you must call this function once to set it up without placeholders for texts.

    :param openasp_dir: Path to the directory containing the OpenAsp dataset (typically "openasp-v1/") with
      analysis_valid.jsonl and analysis_test.jsonl files.
    :param igcs_dataset_dir: Path to the directory containing the IGCS dataset which its files will be enriched
      with correct texts instead of placeholders.
    """
    openasp_dir = pathlib.Path(openasp_dir)

    openasp_df = pd.concat(
        [
            pd.read_json(openasp_dir / f"analysis_{split}.jsonl", lines=True)
            for split in ["valid", "test"]
        ]
    )

    # Find all files with OpenAsp data
    for fname in glob.glob(os.path.join(f"{igcs_dataset_dir}", "**", "*OpenAsp.jsonl.gz")):
        logger.info(f"Processing {fname}")
        file_samples = [SelectionSample(**s) for s in ioutils.jsonl_load(fname)]

        # Loop each record in the anonymized file and copy the span_text from the OpenAsp records
        for sample in file_samples:
            assert sample.source_dataset == OpenAspDataset.name
            topic_id, aspect_label = sample.id.removeprefix(OpenAspDataset.name + ".").rsplit(
                "/", 1
            )
            res = openasp_df[
                (openasp_df["topic_id"] == topic_id) & (openasp_df["aspect_label"] == aspect_label)
            ]
            assert len(res) == 1, f"{topic_id}, {aspect_label}"
            openasp_row = res.iloc[0].to_dict()

            # We use salt to make it harder to guess hashes.
            salt = OpenAspDataset.name + "." + topic_id + aspect_label

            # First, replace the content of docs.
            for doc in sample.docs:
                matching_doc = [
                    x for x in openasp_row["documents"] if x["filename"] == doc.filename
                ]
                assert len(matching_doc) == 1
                digest = doc.text  # It's currently a hash placeholder.
                doc.text = "\n".join(matching_doc[0]["text"])
                verify_integrity(doc.text, digest, salt)

            # Next, replace the content of the selections.
            for sel_group in sample.selections:
                for selection in sel_group.selections:
                    digest = selection.content  # It's currently a hash placeholder.
                    selection.content = sample.docs[selection.doc_id].text[
                        selection.start_pos : selection.end_pos
                    ]
                    verify_integrity(selection.content, digest, salt)

        logger.info(f"Saving {fname}")
        ioutils.jsonl_dump([x.model_dump(mode="json") for x in file_samples], fname)


def anonymize_openasp_files(igcs_dataset_dir: str | pathlib.Path = configs.IGCS_DATA_DIR):
    """This function removes OpenAsp's licensed data from the given IGCS dataset files.
    No need to use it unless you are developing this library.
    """
    # Find all files with OpenAsp data files.
    for fname in glob.glob(f"{igcs_dataset_dir}/**/*OpenAsp.jsonl.gz"):
        logger.info(f"Removing OpenAsp licensed data from {fname}")
        file_samples = [SelectionSample(**s) for s in ioutils.jsonl_load(fname)]

        # Loop each record in the anonymized file and copy the span_text from the OpenAsp records
        for sample in file_samples:
            assert sample.source_dataset == OpenAspDataset.name
            topic_id, aspect_label = sample.id.rsplit("/", 1)

            # We use salt to make it harder to guess hashes.
            salt = topic_id + aspect_label

            # First, replace the content of docs.
            for doc in sample.docs:
                doc.text = compute_digest(doc.text, salt)

            # Next, replace the content of the selections.
            for sel_group in sample.selections:
                for selection in sel_group.selections:
                    selection.content = compute_digest(selection.content, salt)

            # Finally, remove the source sample - which may contain original data.
            sample.source_sample = None

        # Override the file with the anonymized content.
        logger.info(f"Saving {fname}")
        ioutils.jsonl_dump([x.model_dump(mode="json") for x in file_samples], fname)


def verify_integrity(text: str, digest: str, salt: str):
    if compute_digest(text, salt) != digest:
        if digest == text:
            raise ValueError(
                f"Content is already decoded, inspect the file manually. found content - {text[:500] + '...'!r}"
            )
        raise ValueError("Bad content integrity, something went wrong during decoding.")


def compute_digest(text: str, salt: str) -> str:
    return hashlib.sha512((salt + text).encode("utf-8")).hexdigest()


def parse_args(args=None) -> argparse.ArgumentParser:
    """Return a CLI parser for prepare_openasp_files."""

    parser = argparse.ArgumentParser(
        prog="prepare_openasp_files",
        description=(
            "Add OpenAsp's licensed data to the IGCS dataset directory. "
            "Run once to replace the placeholder texts."
        ),
    )

    # Required positional argument
    parser.add_argument(
        "openasp_dir",
        type=pathlib.Path,
        help=(
            "Directory containing the OpenAsp dataset "
            '(e.g. "openasp-v1/") with analysis_valid.jsonl '
            "and analysis_test.jsonl files "
            "(see https://github.com/liatschiff/OpenAsp for details on creating OpenAsp)."
        ),
    )

    # Optional flag with default taken from configs
    parser.add_argument(
        "--igcs-dataset-dir",
        type=pathlib.Path,
        default=configs.IGCS_DATA_DIR,
        metavar="DIR",
        help=(
            "Directory of the IGCS dataset to be enriched " f"(default: {configs.IGCS_DATA_DIR})"
        ),
    )

    parser.add_argument(
        "--remove-data",
        action="store_true",
        default=False,
        help=(
            "Replaces real OpenAsp's data with placeholder digests, the opposite of what this script does by default. "
            "There is no reason to use it, unless you are developing this library."
        ),
    )

    return parser.parse_args(args)


def main(argv=None):
    args = parse_args(argv)

    if not args.remove_data:
        prepare_openasp_files(openasp_dir=args.openasp_dir, igcs_dataset_dir=args.igcs_dataset_dir)
    else:
        anonymize_openasp_files(igcs_dataset_dir=args.igcs_dataset_dir)


if __name__ == "__main__":
    log.init()
    main()
