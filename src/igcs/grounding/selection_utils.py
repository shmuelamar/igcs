from collections import deque
from typing import Iterable

from igcs.entities import Doc, Selection


def selections_to_chars(
    selections: list[Selection] | Selection,
) -> list[tuple[int, int]]:
    if isinstance(selections, Selection):
        selections = [selections]
    return [(s.doc_id, pos) for s in selections for pos in range(s.start_pos, s.end_pos)]


def chars_to_selections(
    char_indices: Iterable[tuple[int, int]], docs: list[Doc]
) -> list[Selection]:
    indices = deque(sorted(char_indices))

    spans = []
    cur_span = None
    while indices:
        doc_id, idx = indices.popleft()

        # init span start
        if cur_span is None:
            cur_span = [doc_id, idx, idx + 1]  # doc_id, start, end
        # if consecutive - add to current span (end is exclusive)
        elif cur_span[0] == doc_id and cur_span[2] == idx:
            cur_span[2] += 1
        # not consecutive
        else:
            spans.append(cur_span)
            cur_span = None

    if cur_span is not None:
        spans.append(cur_span)

    return [
        Selection(
            doc_id=doc_id,
            start_pos=start,
            end_pos=end,
            content=docs[doc_id].text[start:end],
            metadata={"mode": "exact_match"},
        )
        for doc_id, start, end in spans
    ]


def intersect_selections(
    selection_a: list[Selection], selection_b: list[Selection], docs: list[Doc]
) -> list[Selection]:
    a_indices = selections_to_chars(selection_a)
    b_indices = selections_to_chars(selection_b)
    intersect = set(a_indices) & set(b_indices)
    return chars_to_selections(intersect, docs)
