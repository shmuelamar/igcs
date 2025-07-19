import logging

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from trl import SFTTrainer

from igcs.entities import Doc, Selection, SelectionSample
from igcs.utils import prompting

EMPTY_SELECTION_DOC_ID = 999
EMPTY_SELECTION_MESSAGE = (
    "Found no relevant content matching the instruction on the given documents."
)

logger = logging.getLogger(__name__)


class IGCSTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def get_model_sep_tokens(model_name: str = "llama-3") -> tuple[str, str]:
    if "llama-3" in model_name.lower():
        return "<|reserved_special_token_249|>", "<|reserved_special_token_250|>"
    else:
        raise NotImplementedError("only support Llama-3 by now")


def get_model_sep_tokens2(igcs_format: bool) -> tuple[str, str]:
    if igcs_format:
        return "<|reserved_special_token_247|>", "<|reserved_special_token_248|>"
    return "<b>", "</b>"


def selections_to_encoded_answer(
    selections: list[Selection], doc_id_sep: str, span_sep: str
) -> str:
    # Negative example - output a span marking no selection
    if not selections:
        return f"{EMPTY_SELECTION_DOC_ID}{doc_id_sep}{EMPTY_SELECTION_MESSAGE}{span_sep}"

    text = []
    for selection in selections:
        text.append(f"{selection.doc_id}{doc_id_sep}{selection.content}{span_sep}")
    return "".join(text)


def igcs_format_sample(sample: SelectionSample | dict, model_name: str):
    from igcs.datasets.prepare_igcs_datasets import get_merged_selections

    doc_id_sep, span_sep = get_model_sep_tokens(model_name)
    if isinstance(sample, dict):
        sample = SelectionSample(**sample)

    prompt = prompting.format_prompt(sample)
    merged_selections = get_merged_selections(sample, mode="merge")
    answer = selections_to_encoded_answer(
        merged_selections, doc_id_sep=doc_id_sep, span_sep=span_sep
    )
    conv = prompting.get_conversation(prompt=prompt, answer=answer)
    return {"conversations": conv}


def make_allowed_tokens_fn(
    docs: list[Doc] | list[str] | str,
    tokenizer: PreTrainedTokenizerFast,
    only_doc_id: bool = False,
):
    doc_id_sep, span_sep = get_model_sep_tokens()
    doc_id_sep_token_id = tokenizer.encode(doc_id_sep, add_special_tokens=False)[0]
    span_sep_token_id = tokenizer.encode(span_sep, add_special_tokens=False)[0]

    # precomputed lists of token ids
    end_generation_token_ids = [span_sep_token_id, tokenizer.eos_token_id]
    allowed_doc_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(1000)])
    all_token_ids_without_doc_id = list(range(len(tokenizer.vocab)))
    all_token_ids_without_doc_id.remove(doc_id_sep_token_id)

    if docs and isinstance(docs, list):
        doc_id_to_text = [d.text for d in docs]
    elif isinstance(docs, str):
        doc_id_to_text = [docs]
    else:
        doc_id_to_text = docs

    doc_id_to_tokens = [np.array(tokenizer.encode(txt)) for txt in doc_id_to_text]
    if not only_doc_id:
        doc_id_to_tokens = [np.concatenate(doc_id_to_tokens)]

    def get_allowed_tokens(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        # 1. "" -> allow all digits
        if not len(input_ids):
            return allowed_doc_ids

        tail = input_ids[-1].item()
        # 2. "9" -> allow only doc_id_sep
        if tail in allowed_doc_ids:
            return [doc_id_sep_token_id]

        # 3. "9<D>" -> allow * except for the <D> seperator
        if tail == doc_id_sep_token_id:
            return all_token_ids_without_doc_id

        # 4. "9<D>word<S>" -> allow only doc_ids numbers (like 1.)
        if tail == span_sep_token_id:
            return allowed_doc_ids

        # 5. "9<D>word" -> allow any existing suffix + <S> + terminators
        doc_id_sep_last_pos = (input_ids == doc_id_sep_token_id).nonzero()[0][-1]
        span_prefix = input_ids[doc_id_sep_last_pos + 1 :].cpu().numpy()
        assert len(span_prefix)

        if only_doc_id:
            last_doc_id_text = tokenizer.convert_tokens_to_ids([input_ids[doc_id_sep_last_pos - 1]])
            try:
                last_doc_id = int(last_doc_id_text)
            except ValueError:
                logger.debug(f"invalid doc id found - {last_doc_id_text}")
                last_doc_id = 0
        else:
            last_doc_id = 0

        return (
            list(find_next_tokens(doc_id_to_tokens[last_doc_id], search_sequence=span_prefix))
            + end_generation_token_ids
        )

    return get_allowed_tokens


def find_next_tokens(doc_token_ids: np.array, search_sequence: np.array) -> set[int]:
    next_tokens = set()

    for start in np.where(doc_token_ids == search_sequence[0])[0]:
        end = start + len(search_sequence)
        if (doc_token_ids[start:end] == search_sequence).all() and end < len(doc_token_ids):
            next_tokens.add(doc_token_ids[end].item())

    return next_tokens


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    ds, ss = get_model_sep_tokens()
    fn = make_allowed_tokens_fn(docs="Given the cat sat on the cat head", tokenizer=tokenizer)
    input_ids = tokenizer.encode(f"0{ds} cat", return_tensors="pt", add_special_tokens=False)[0]
    constrained_tok_ids = fn(1, input_ids)
    print(constrained_tok_ids)
    print(tokenizer.convert_ids_to_tokens(constrained_tok_ids))
