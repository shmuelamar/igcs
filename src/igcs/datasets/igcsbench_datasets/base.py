import pathlib
from abc import ABC, abstractmethod
from operator import attrgetter
from typing import ClassVar, Dict, Iterable, List, Tuple

from pydantic import BaseModel

from igcs.entities import SelectionSample, SelectionType


class SourceDataset(BaseModel, ABC):
    name: ClassVar[str] = "base"
    splits_fnames: Dict[str, str | Tuple[str, ...]]
    dirname: pathlib.Path

    def prepare_dataset(
        self, verify_hallucinations: bool = True
    ) -> Dict[str, List[SelectionSample]]:
        dataset_splits = {}
        for split in self.splits_fnames:
            dataset_splits[split] = self.prepare_split(split, verify_hallucinations)
        return dataset_splits

    def prepare_split(
        self, split: str, verify_hallucinations: bool = True
    ) -> List[SelectionSample]:
        fnames = self.splits_fnames[split]
        if isinstance(fnames, str):
            fnames = [fnames]

        split_data = []
        for fname in fnames:
            filename = self.dirname / fname
            for sample in self.prepare_file(filename, split):
                for sel_group in sample.selections:
                    sel_group.selections.sort(key=attrgetter("doc_id", "start_pos", "end_pos"))
                    for sel in sel_group.selections:
                        sel.metadata = sel.metadata or {}
                        sel.metadata["total_count"] = sum(
                            doc.text.find(sel.content) >= 0 for doc in sample.docs
                        )
                self.verify_sample_integrity(sample, verify_hallucinations)

                split_data.append(sample)
        return split_data

    @abstractmethod
    def prepare_file(self, fname: pathlib.Path, split: str) -> Iterable[SelectionSample]: ...

    @staticmethod
    def sentences_to_chunks_pos(sentences: list[str]) -> list[int]:
        chunks_pos = []
        cur_pos = 0
        for sentence in sentences:
            cur_pos += len(sentence) + 1  # +1 for \n
            chunks_pos.append(cur_pos)

        # last sentence has no new line
        chunks_pos[-1] -= 1
        return chunks_pos

    @staticmethod
    def verify_sample_integrity(sample: SelectionSample, verify_hallucinations: bool = True):
        # verify selections content
        for sel_group in sample.selections:
            for sel in sel_group.selections:
                sel_doc = sample.docs[sel.doc_id]

                if sel.doc_id == -1:
                    assert verify_hallucinations, "hallucination of not-grounded selection"
                    continue

                assert sel_doc.text[sel.start_pos : sel.end_pos] == sel.content

                if sample.type == SelectionType.span:  # only sentences have chunks
                    continue

                # +1 for the new line which we don't include
                assert (sel.start_pos, sel.end_pos + 1) in list(
                    zip([0] + sel_doc.chunks_pos, sel_doc.chunks_pos)
                ) or [sel.start_pos, sel.end_pos] == sel_doc.chunks_pos[-2:]
                assert sel.content in [
                    s.content for s in sel_doc.sentences()
                ] or sel.content + "\n" in [s.content for s in sel_doc.sentences()]

        # verify doc ids
        assert [doc.id for doc in sample.docs] == list(range(len(sample.docs)))

        if sample.type == SelectionType.span:  # only sentences have chunks
            return

        for doc in sample.docs:
            assert all(a < b for a, b in zip(doc.chunks_pos, doc.chunks_pos[1:]))
            assert len(doc.text) == doc.chunks_pos[-1]
