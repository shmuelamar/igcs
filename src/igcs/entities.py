from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import unquote

from pydantic import BaseModel


class SelectionType(str, Enum):
    sentence = "sentence"
    span = "span"


class Selection(BaseModel):
    doc_id: int
    start_pos: int
    end_pos: int
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SelectionGroup(BaseModel):
    selections: List[Selection]
    id: Optional[str] = None


class Doc(BaseModel):
    id: int
    text: str
    filename: Optional[str] = None
    chunks_pos: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None

    def sentences(self) -> list[Selection]:
        return [
            Selection(
                doc_id=self.id,
                start_pos=start,
                end_pos=end,
                content=self.text[start:end],
            )
            for start, end in self.get_sentences_boundaries()
        ]

    def get_sentences_boundaries(self) -> list[tuple[int, int]]:
        if self.chunks_pos is None:
            raise ValueError("no chunks for this sample, cannot make sentence boundaries")
        return list(zip([0] + self.chunks_pos, self.chunks_pos))


class SelectionSample(BaseModel):
    id: str
    source_dataset: str
    selections: list[SelectionGroup]
    docs: List[Doc]
    instruction_context: dict[str, str | bool]
    source_sample: Any = None
    type: SelectionType = SelectionType.span

    @property
    def is_negative(self) -> bool:
        return self.instruction_context.get("is_negative", False)

    @staticmethod
    def parse_guid(guid: str):
        dataset, split, encoded_id = guid.split(".", maxsplit=2)
        return dataset, split, unquote(encoded_id)


class ModelPrediction(TypedDict):
    """Type hints for prediction, using dict for simplicity and interoperability with other tools"""

    sample_id: str
    grounded_selection: list[
        TypedDict(
            "grounded_selection",
            {"doc_id": int, "start_pos": int, "end_pos": int, "content": str},
        )
    ]
