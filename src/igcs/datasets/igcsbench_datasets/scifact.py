import pathlib
from typing import Dict, Iterable

import pandas as pd

from igcs.configs import SRC_DATASETS_DIR
from igcs.datasets.igcsbench_datasets.base import SourceDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample, SelectionType
from igcs.utils import ioutils


class SciFactDataset(SourceDataset):
    name = "SciFact"
    # SciFact gold test is on leaderboard so we use dev
    splits_fnames: Dict[str, str] = {
        "train": "claims_train.1.jsonl",
        "dev": "claims_train.2.jsonl",
        "test": "claims_dev.jsonl",
    }
    dirname: pathlib.Path = SRC_DATASETS_DIR / "SciFact"

    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]:
        id2doc = self.get_corpus()

        data = list(ioutils.jsonl_load(fname))
        for raw_sample in data:
            docs = []

            selections = []
            for ev_doc_id, doc_evidences in raw_sample["evidence"].items():
                ev_doc_id = int(ev_doc_id)
                ev_doc = id2doc[ev_doc_id]
                doc = self.parse_raw_doc(id2doc[ev_doc_id], doc_id=len(docs))
                docs.append(doc)

                for evidence in doc_evidences:
                    for sent_idx in evidence["sentences"]:
                        content = ev_doc["abstract"][sent_idx].strip()
                        assert doc.text.count(content) == 1

                        start_pos = doc.text.find(content)
                        end_pos = start_pos + len(content)
                        selections.append(
                            Selection(
                                doc_id=len(docs) - 1,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                content=content,
                            )
                        )
                        # sanity check
                        assert doc.text[start_pos:end_pos] == content

            # add docs without any evidence but that included in the sample
            for doc_id in raw_sample["cited_doc_ids"]:
                if not any(d.metadata["doc_id"] == doc_id for d in docs):
                    docs.append(self.parse_raw_doc(id2doc[doc_id], len(docs)))

            # Add relevant docs info from the corpus.
            raw_sample["source_docs"] = {
                doc_id: id2doc[doc_id] for doc_id in raw_sample["cited_doc_ids"]
            }
            claim = raw_sample["claim"].strip().strip(".").strip()
            assert len(docs) == len(set(raw_sample["cited_doc_ids"]))

            yield SelectionSample(
                id=f"{self.name}.{raw_sample['id']}",
                source_dataset=self.name,
                selections=[SelectionGroup(selections=selections)],
                docs=docs,
                instruction_context={"claim": claim},
                source_sample=raw_sample,
                type=SelectionType.sentence,
            )

    @staticmethod
    def parse_raw_doc(doc: dict, doc_id: int) -> Doc:
        sentences = [sent.strip() for sent in doc["abstract"]]
        return Doc(
            id=doc_id,
            text="\n".join(sentences),
            filename=doc["title"],
            chunks_pos=SciFactDataset.sentences_to_chunks_pos(sentences),
            metadata={"doc_id": doc["doc_id"], "abstract": doc["abstract"]},
        )

    def get_corpus(self) -> dict[int, dict]:
        corpus = list(ioutils.jsonl_load(self.dirname / "corpus.jsonl"))
        id2doc = {x["doc_id"]: x for x in corpus}
        assert len(corpus) == len(id2doc)
        return id2doc

    def split_train_dev(self):
        df = pd.read_json(self.dirname / f"claims_train.jsonl", lines=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        pos = int(len(df) * 0.85)
        df[:pos].to_json(self.dirname / f"claims_train.1.jsonl", lines=True, orient="records")
        df[pos:].to_json(self.dirname / f"claims_train.2.jsonl", lines=True, orient="records")


if __name__ == "__main__":  # pragma: nocover
    dataset = SciFactDataset().prepare_dataset()
    sample = dataset["dev"][0]
    print(sample)
