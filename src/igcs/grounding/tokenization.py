import bisect
import logging
from functools import cache
from typing import Protocol

import spacy

from igcs.entities import Doc, Selection

logger = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_SPACY_MODEL_VERSION = "3.7.1"


class Token(Protocol):
    """Grounding expect textual tokens with character-level index in the source docs"""

    text: str
    idx: int


@cache
def load_spacy_model(model: str, version: str | None):
    try:
        return spacy.load(model)
    except OSError:
        model_id = model if version is None else f"{model}-{version}"
        logger.info(f"Installing spaCy model {model_id}...")
        spacy.cli.download(model_id, direct=bool(version))
    return spacy.load(model)


# @lru_cache(maxsize=1_000_000)
def spacy_tokenize(
    text: str,
    model: str = DEFAULT_SPACY_MODEL,
    model_version: str | None = DEFAULT_SPACY_MODEL_VERSION,
    ignore_tags=("SP",),
):
    """Tokenize text into tokens using spacy, by default ignore white-spaces tokens"""
    nlp = load_spacy_model(model, model_version)
    try:
        tokens = nlp(text)
    except UnicodeEncodeError:
        tokens = nlp(text.encode("utf-8", errors="ignore").decode("utf-8"))

    return [token for token in tokens if token.tag_ not in ignore_tags and not token.is_space]


def translate_tokens(doc_tokens: list[Token], start: int, end: int) -> tuple[int, int]:
    last_token = end - 1  # non-inclusive python style indexing: [1,2,3,4][1:3] == [2,3]
    start = doc_tokens[start].idx
    end = doc_tokens[last_token].idx + len(doc_tokens[last_token])
    return start, end


def token_ids_to_selections(token_ids: list[tuple[int, int]], docs: list[Doc]) -> list[Selection]:
    token_ids.sort()
    docs_tokens = [spacy_tokenize(doc.text) for doc in docs]
    raw_selections = []

    stack = list(token_ids)
    while stack:
        doc_id, tok_id = stack.pop(0)
        raw_selections.append([(doc_id, docs_tokens[doc_id][tok_id])])

        if not stack:
            break

        while stack and doc_id == stack[0][0] and tok_id + 1 == stack[0][1]:
            # add the next and progress
            doc_id, tok_id = stack.pop(0)
            raw_selections[-1].append((doc_id, docs_tokens[doc_id][tok_id]))

    # now convert tokens into spans
    selections = []
    for raw_selection in raw_selections:
        first_doc_id, first_token = raw_selection[0]
        last_doc_id, last_token = raw_selection[-1]
        assert first_doc_id == last_doc_id

        start_pos = first_token.idx
        end_pos = last_token.idx + len(last_token.text)
        selections.append(
            Selection(
                doc_id=first_doc_id,
                start_pos=start_pos,
                end_pos=end_pos,
                content=docs[first_doc_id].text[start_pos:end_pos],
            )
        )
    return selections


def selections_to_token_ids(
    selections: list[Selection], docs: list[Doc], tokenizer=spacy_tokenize
) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokenizes selections into token ids"""
    docs_tokens = [tokenizer(doc.text) for doc in docs]
    docs_tokens_start_indices = [[token.idx for token in doc_tokens] for doc_tokens in docs_tokens]
    docs_tokens_end_indices = [
        [token.idx + len(token.text) for token in doc_tokens] for doc_tokens in docs_tokens
    ]
    for doc, dt in zip(docs, docs_tokens_end_indices):
        dt[-1] = len(doc.text)

    token_ids_selections = set()
    for sel in selections:
        doc_id = sel.doc_id
        if doc_id == -1 or not sel.content:
            continue

        doc_tokens = docs_tokens[doc_id]

        # find start token, round down if not a full token is selected
        start_token_idx = bisect.bisect_left(docs_tokens_start_indices[doc_id], sel.start_pos)
        if start_token_idx > 0 and (
            start_token_idx == len(doc_tokens) or doc_tokens[start_token_idx].idx > sel.start_pos
        ):
            start_token_idx -= 1

        # find end token, round up if not a full token is selected
        end_token_idx = bisect.bisect_right(docs_tokens_end_indices[doc_id], sel.end_pos)
        end_token = doc_tokens[end_token_idx - 1]  # -1 as end is exclusive
        if end_token_idx < len(doc_tokens) and end_token.idx + len(end_token) < sel.end_pos:
            end_token_idx += 1

        if start_token_idx == end_token_idx:
            end_token_idx += 1

        assert end_token_idx > 0, f"empty selection for {sel.dict()}"
        assert start_token_idx < end_token_idx, f"{sel.dict()}"
        sel_token_ids = range(start_token_idx, end_token_idx)
        token_ids_selections |= {(doc_id, token_id) for token_id in sel_token_ids}

    token_ids_selections = sorted(token_ids_selections)
    tokens = [docs_tokens[doc_id][tok_id].text for doc_id, tok_id in token_ids_selections]
    return tokens, token_ids_selections
