import logging
import re
from typing import Callable, List, Sequence

from fuzzysearch import Match, find_near_matches

from igcs.entities import Doc, Selection
from igcs.grounding.completion_parsers import parse_selection
from igcs.grounding.tokenization import Token, spacy_tokenize, translate_tokens

MatcherType = Callable[[str, int | None, list[Doc], Callable], Selection | None]
NormalizerType = Callable[[str], str] | None

logger = logging.getLogger(__name__)


def identity(x):
    return x


def normalize(s: str) -> str:
    return re.sub(r"\s", " ", s.lower())


def exact_match(
    span: str,
    hint_doc_id: int | None,
    docs: list[Doc],
    normalizer: NormalizerType = None,
) -> Selection | None:
    normalizer = normalizer or identity

    # sort docs such that the first doc_id is the hinted doc
    if hint_doc_id is not None:
        docs = docs.copy()
        docs.sort(key=lambda doc: doc.id != hint_doc_id)

    span_text = normalizer(span)
    docs_text = [normalizer(doc.text) for doc in docs]

    total_count = sum(doc_text.find(span_text) >= 0 for doc_text in docs_text)
    for doc, doc_text in zip(docs, docs_text):
        pos = doc_text.find(span_text)
        if pos >= 0:
            return Selection(
                doc_id=doc.id,
                start_pos=pos,
                end_pos=pos + len(span_text),
                content=doc.text[pos : pos + len(span_text)],  # use original text
                metadata={"total_count": total_count},
            )

    # no match
    return None


def fuzzy_match(
    span: str,
    hint_doc_id: int | None,
    docs: List[Doc],
    normalizer: NormalizerType = None,
    *,
    tokenizer: Callable[[str], list[Token]] = spacy_tokenize,
    max_dist_abs: int = 10,
    max_dist_rel: float = 0.15,
) -> Selection | None:
    normalizer = normalizer or identity

    # sort docs such that the first doc_id is the hinted doc
    if hint_doc_id is not None:
        docs = docs.copy()
        docs.sort(key=lambda doc: doc.id != hint_doc_id)

    span_tokens = tokenizer(normalizer(span))
    span_tokens_text = [token.text for token in span_tokens]
    docs_tokens = [tokenizer(normalizer(doc.text)) for doc in docs]
    docs_tokens_text = [[token.text for token in doc_tokens] for doc_tokens in docs_tokens]

    all_matches = []
    # the maximum distance determine by both absolute and relative max distances
    max_dist = min(int(len(span_tokens_text) * max_dist_rel), max_dist_abs)

    if max_dist == 0:
        return None

    for doc, doc_tokens, doc_tokens_text in zip(docs, docs_tokens, docs_tokens_text, strict=True):
        matches = fuzzy_match_sequence(span_tokens_text, doc_tokens_text, max_dist=max_dist)
        # filter out empty matches
        all_matches += [(doc, doc_tokens, match) for match in matches if match.end > match.start]

    if not all_matches:
        return None

    # sort all matches and choose the first by:
    all_matches.sort(
        key=lambda match: (
            match[2].dist,  # minimum edit distance
            match[0].id != hint_doc_id,  # doc_id as hinted
            match[0].id,  # doc_id number
            match[2].end,  # end position of the match
            match[2].start,  # start position of the match
        )
    )
    match_doc, match_doc_tokens, match = all_matches[0]

    # prepare match
    start_pos, end_pos = translate_tokens(match_doc_tokens, match.start, match.end)

    return Selection(
        doc_id=match_doc.id,
        start_pos=start_pos,
        end_pos=end_pos,
        content=match_doc.text[start_pos:end_pos],  # use original text
        metadata={"total_count": len(all_matches), "best_dist": match.dist},
    )


def fuzzy_match_sequence(sub: list[str], seq: list[str], max_dist: int) -> list[Match]:
    try:
        return find_near_matches(sub, seq, max_l_dist=max_dist)
    except IndexError:
        logger.warning(f"Index out of range when doing fuzzy match on: {sub[:10]}...")
        return []


GROUNDING_MODES: dict[str, tuple[MatcherType, NormalizerType]] = {
    "exact_match": (exact_match, identity),
    "normalized_match": (exact_match, normalize),
    "fuzzy_match": (fuzzy_match, identity),
}
ALL_GROUNDING_MODES = ("exact_match", "normalized_match", "fuzzy_match")


def ground_selections(
    selections: list[str | None] | list[tuple[str, int] | None] | list[Selection] | None,
    docs: list[Doc],
    modes: Sequence[str] | None = None,
    max_dist_rel: float | None = None,
    max_dist_abs: int | None = None,
) -> list[Selection] | None:
    """Grounds selections to documents"""
    if selections is None:
        return None
    if (
        selections
        and isinstance(selections, (list, tuple))
        and isinstance(selections[0], Selection)
    ):
        return selections

    modes = modes or ALL_GROUNDING_MODES
    grounded_selections = []
    for selection in selections:
        if selection is None:
            continue

        grounded_selections.append(
            ground_selection(
                selection, docs, modes=modes, max_dist_rel=max_dist_rel, max_dist_abs=max_dist_abs
            )
        )
    return grounded_selections


def ground_selection(
    span: str | tuple[str, int],
    docs: List[Doc],
    modes: Sequence[str] | None = None,
    max_dist_rel: float | None = None,
    max_dist_abs: int | None = None,
) -> Selection:
    """Grounds selection to a document by applying strict to fuzzy matchers."""
    modes = modes or ALL_GROUNDING_MODES

    # the IGCS formatted model hints on the predicted doc_id which helps grounding
    if (
        isinstance(span, tuple)
        and len(span) == 2
        and isinstance(span[0], str)
        and isinstance(span[1], int)
    ):
        span, hint_doc_id = span
    else:
        hint_doc_id = None

    if not isinstance(span, str):
        span = str(span)

    # empty
    if not span.strip():
        return Selection(
            doc_id=-1,
            start_pos=-1,
            end_pos=-1,
            content=str(span),
            metadata={"total_count": 0, "mode": "whitespace"},
        )

    for mode in modes:
        match_fn, normalizer = GROUNDING_MODES[mode]

        kwargs = {}
        if mode == "fuzzy_match":
            if max_dist_abs is not None:
                kwargs["max_dist_abs"] = max_dist_abs
            if max_dist_rel is not None:
                kwargs["max_dist_rel"] = max_dist_rel

        match = match_fn(span, hint_doc_id, docs, normalizer, **kwargs)
        if match:
            match.metadata["mode"] = mode
            return match

    # hallucination
    return Selection(
        doc_id=-1,
        start_pos=-1,
        end_pos=-1,
        content=span,
        metadata={"total_count": 0, "mode": "hallucination"},
    )
