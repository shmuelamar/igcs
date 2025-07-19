import json
import logging
import pathlib
from typing import Iterable

import pandas as pd

from igcs.configs import SRC_DATASETS_DIR
from igcs.datasets.igcsbench_datasets.base import SourceDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample, SelectionType

logger = logging.getLogger(__name__)

TOPIC_NAMES = {
    "eq": "Earthquake",
    "fraud": "Fraud",
}

ASPECTS_NAMES = {
    "geo": "Geography",
    "rescue": "Recovery",
    "penalty": "Penalty",
    "nature": "Nature",
}

# From Table 1 in the paper
ASPECT_DESCRIPTIONS = {
    "geo": "geography, region, or location",
    "rescue": "recovery and aid efforts (death toll and injuries, foreign/domestic government assistance, impact on survivors)",
    "penalty": "penalty or consequences for the fraudster, or for others",
    "nature": "nature of the fraud: the amount of money taken, benefits for the fraudster, and how the fraud worked",
}


class AspectNewsDataset(SourceDataset):
    name = "AspectNews"
    splits_fnames: dict[str, tuple[str, ...]] = {
        "train": ("eq_100_geo.1.csv", "eq_100_rescue.1.csv"),
        "dev": ("eq_100_geo.2.csv", "eq_100_rescue.2.csv"),
        "test": ("fraud_100_nature.csv", "fraud_100_penalty.csv"),
    }
    dirname: pathlib.Path = SRC_DATASETS_DIR / "AspectNews"

    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]:
        raw_samples = self._parse_raw_samples(fname)

        topic, _, aspect = fname.name.split(".")[0].split("_")
        # we split the EQ topic into two sets, so to keep ids unique we start indexing at 80 for the dev set.
        # otherwise the dev and train set will have overlapping ids (but not overlapping samples).
        if split == "dev":
            start = 80
        else:
            start = 0

        for idx, raw_sample in enumerate(raw_samples, start=start):
            sentences = [s.strip() for s in raw_sample["sentences"]]

            chunks_pos = self.sentences_to_chunks_pos(sentences)

            selections = [
                SelectionGroup(
                    selections=[
                        Selection(
                            doc_id=0,
                            # start not below zero
                            start_pos=chunks_pos[index - 1] if index > 0 else 0,
                            # end - remove the newline, except in case of last sentence
                            # where there is no newline
                            end_pos=chunks_pos[index] - bool(index < len(chunks_pos) - 1),
                            content=sentences[index],
                            # score between 3-1 of aspect relevancy
                            metadata={"importance": priority},
                        )
                        for index, priority in annotator_annotation
                    ],
                    id=str(annotator_id),
                )
                for annotator_id, annotator_annotation in enumerate(raw_sample["annotations"])
            ]

            yield SelectionSample(
                id=f"{self.name}.{topic}_{aspect}_{idx}",
                source_dataset=self.name,
                selections=selections,
                docs=[
                    Doc(
                        id=0,
                        filename=None,
                        text="\n".join(sentences),
                        chunks_pos=chunks_pos,
                    )
                ],
                # convert from short names to long names
                instruction_context={
                    "aspect": ASPECTS_NAMES[aspect],
                    "topic": TOPIC_NAMES[topic],
                    "aspect_description": ASPECT_DESCRIPTIONS[aspect],
                },
                source_sample=raw_sample,
                type=SelectionType.sentence,
            )

    def _parse_raw_samples(self, fname: pathlib.Path) -> list[dict]:
        # code adapted from original implementation on: https://github.com/oja/aosumm/blob/master/data/redisplay.py
        # commented out code is from the original implementation and left for clarity.
        annotations = pd.read_csv(fname)
        seen = {}
        article_ids = [seen.setdefault(x, x) for x in list(annotations["HITId"]) if x not in seen]

        id2article = {}
        for id in article_ids:
            id2article[id] = json.loads(
                list(annotations[annotations["HITId"] == id]["Input.articlejson"])[0]
            )

        raw_samples = []
        for id in article_ids:
            # remove the "article" as the first sentence
            sentences = [(x[0]["text"] + x[1]["text"]) for x in id2article[id]["paragraphs"][2:]][
                :11
            ]

            raw_annotations = [
                json.loads(x)[0]
                for x in list(annotations[annotations["HITId"] == id]["Answer.taskAnswers"])
            ]
            annotation_ids = self._convert_raw_annotations_to_ids(raw_annotations)

            raw_samples.append({"sentences": sentences, "annotations": annotation_ids})

        return raw_samples

    @staticmethod
    def _convert_raw_annotations_to_ids(raw_annotations):
        full_ids = []
        for annotation in raw_annotations:
            annotation.pop("comments", None)
            query_keys = [f"topic-segment-{x}" for x in range(2, 2 * len(annotation) + 1, 2)]
            single_user_ids = []
            for i, k in enumerate(query_keys):
                # print(raw_annotation[k])
                sentence_annotation = annotation[k]
                if sentence_annotation == "not-in-summary":
                    pass
                else:
                    single_user_ids.append((i, int(sentence_annotation.split("-")[1])))

            single_user_ids.sort(key=lambda x: x[1], reverse=True)
            full_ids.append(single_user_ids)

        return full_ids

    def split_train_dev(self):
        for name in ["eq_100_rescue", "eq_100_geo"]:
            df = pd.read_csv(self.dirname / f"{name}.csv")
            assert len(df) == 500
            df[:400].to_csv(self.dirname / f"{name}.1.csv")
            df[400:].to_csv(self.dirname / f"{name}.2.csv")


if __name__ == "__main__":  # pragma: nocover
    dataset = AspectNewsDataset().prepare_dataset()
    sample = dataset["test"][5]
    print(sample.selections)
