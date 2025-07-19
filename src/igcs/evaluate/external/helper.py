import logging
from operator import attrgetter

from igcs.entities import Doc, ModelPrediction, Selection, SelectionSample
from igcs.evaluate import evaluation_functions
from igcs.grounding import tokenization

logger = logging.getLogger(__name__)


def align_model_prediction(
    predictions: list[ModelPrediction], samples: list[SelectionSample]
) -> list[ModelPrediction]:
    id2sample = {s.id: s for s in samples}
    sample_id_order = [s.id for s in samples]
    assert len(id2sample) == len(samples), "not unique"

    # sort predictions by original order
    predictions = sorted(predictions, key=lambda p: sample_id_order.index(p["sample_id"]))
    aligned_predictions = []
    for pred in predictions:
        docs = id2sample[pred["sample_id"]].docs
        selections = [
            s if isinstance(s, Selection) else Selection(**s)
            for s in pred["grounded_selection"] or []
        ]

        aligned_predictions.append(
            {
                "sample_id": pred["sample_id"],
                "grounded_selection": [
                    s.dict() for s in align_selections_to_sentences(selections, docs)
                ],
            }
        )
    return aligned_predictions


def align_selections_to_sentences(selections: list[Selection], docs: list[Doc]) -> list[Selection]:
    aligned_selections = []
    for selection in selections:
        # ignore hallucinations
        if selection.doc_id == -1:
            continue

        doc = docs[selection.doc_id]
        overlapping_sents = get_overlapping_sentences(selection, doc)
        logger.debug(f"found {len(overlapping_sents)} overlapping sentences for a selection")
        aligned_selections += overlapping_sents

    aligned_selections.sort(key=attrgetter("doc_id", "end_pos", "start_pos"))
    seen = set()
    dedup_selections = []
    for aligned_sel in aligned_selections:
        sel_key = (aligned_sel.doc_id, aligned_sel.start_pos, aligned_sel.end_pos)
        if sel_key not in seen:
            dedup_selections.append(aligned_sel)
            seen.add(sel_key)
    return dedup_selections


def get_overlapping_sentences(
    selection: Selection, doc: Doc, overlap_threshold: float = 1e-7
) -> list[Selection]:
    """Finds the enclosing sentences inside the doc"""

    doc0 = doc.model_copy(update={"id": 0})
    selection_token_ids = tokenization.selections_to_token_ids(
        [selection.model_copy(update={"doc_id": 0})], docs=[doc0]
    )[1]

    overlapping_sentences = []
    for sent_id, sentence in enumerate(doc.sentences()):
        # case1: sentence and selection do not overlap
        if selection.end_pos <= sentence.start_pos or selection.start_pos >= sentence.end_pos:
            continue

        # case2: selection contains sentence - include it
        if selection.start_pos <= sentence.start_pos and sentence.end_pos <= selection.end_pos:
            overlapping_sentences.append((sent_id, sentence))
            continue

        # case3: some of the sentence overlap with some of the selection
        # fixing doc_id to 0 for the tokenization work
        sentence_token_ids = tokenization.selections_to_token_ids(
            [sentence.model_copy(update={"doc_id": 0})], docs=[doc0]
        )[1]
        if (
            evaluation_functions.iou_score(selection_token_ids, sentence_token_ids)
            >= overlap_threshold
        ):
            overlapping_sentences.append((sent_id, sentence))

    assert not overlapping_sentences or [sid for sid, _ in overlapping_sentences] == list(
        range(overlapping_sentences[0][0], overlapping_sentences[-1][0] + 1)
    ), f"found non-consecutive sentences = {overlapping_sentences}"

    return [sent for _, sent in overlapping_sentences]


def selections_to_text(
    selections: list[Selection] | SelectionSample | ModelPrediction,
) -> str:
    if not selections:
        return ""

    if isinstance(selections, list) and isinstance(selections[0], Selection):
        pass
    elif isinstance(selections, SelectionSample):
        assert len(selections.selections) == 1
        selections = selections.selections[0].selections
    elif isinstance(selections, dict) and "grounded_selection" in selections:
        selections = [
            s if isinstance(s, Selection) else Selection(**s)
            for s in selections["grounded_selection"] or []
        ]
    else:
        raise ValueError(f"Unknown type: {selections!r}")

    return " ".join([s.content for s in selections if s.doc_id != -1])
