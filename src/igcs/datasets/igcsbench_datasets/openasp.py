import json
import pathlib
from typing import Dict, Iterable

from igcs.configs import SRC_DATASETS_DIR
from igcs.datasets.igcsbench_datasets.base import SourceDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample, SelectionType


class OpenAspDataset(SourceDataset):
    name = "OpenAsp"
    splits_fnames: Dict[str, str] = {
        "train": "valid.1.json",
        "dev": "valid.2.json",
        "test": "test.json",
    }
    dirname: pathlib.Path = SRC_DATASETS_DIR / "OpenAsp"

    def prepare_file(self, fname: str | pathlib.Path, split: str) -> Iterable[SelectionSample]:
        with open(fname) as fp:
            samples = json.load(fp)

        # sort by question id for aggregating
        samples.sort(key=lambda x: (x["topic_id"], x["aspect_label"]))

        # there are multiple samples for the same query - aggregate by them
        for sample in samples:
            selections = SelectionGroup(
                selections=[
                    Selection(
                        doc_id=selection["doc_id"],
                        start_pos=selection["start_char"],
                        end_pos=selection["end_char"],
                        content=selection["span_text"].strip(),
                    )
                    for selection in sample["selected_sentences"]
                ]
            )

            docs = [
                Doc(
                    id=doc["doc_id"],
                    filename=doc["doc_filename"],
                    text=doc["doc_text"].strip(),
                    chunks_pos=self.sentences_to_chunks_pos(sentences=doc["doc_text"].split("\n")),
                )
                for doc in sample["documents"]
            ]
            yield SelectionSample(
                id=f'{self.name}.{sample["topic_id"]}/{sample["aspect_label"]}',
                source_dataset=self.name,
                selections=[selections],
                docs=docs,
                instruction_context={
                    "aspect_label": sample["aspect_label"],
                    "title": sample["title"],
                },
                source_sample=sample,
                type=SelectionType.sentence,
            )


if __name__ == "__main__":
    dataset = OpenAspDataset().prepare_dataset()
    sample = dataset["dev"][5]
    print(sample.selections)
