import logging
from functools import cache

import pandas as pd
import tiktoken
from transformers import AutoTokenizer

from igcs.datasets import get_dataset_file
from igcs.datasets.igcsbench_datasets import DATASETS
from igcs.llms import gpt4
from igcs.utils import ioutils, log

logger = logging.getLogger(__name__)


@cache
def get_tokenizer(name: str):
    if "/" in name:
        logger.info(f"loading huggingface tokenizer from {name}")
        return AutoTokenizer.from_pretrained(
            name,
            padding_side="left",
        )
    try:
        return tiktoken.get_encoding(name)
    except ValueError:
        return tiktoken.get_encoding("cl100k_base")


def tiktoken_count_tokens(messages: list[dict], model: str) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = get_tokenizer(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = get_tokenizer("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return tiktoken_count_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return tiktoken_count_tokens(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_tokens(conversation: list[dict], model: str):
    tokenizer = get_tokenizer(model)
    if hasattr(tokenizer, "apply_chat_template"):
        chat_text = tokenizer.apply_chat_template(conversation, tokenize=False)
        return len(tokenizer.encode(chat_text))
    return tiktoken_count_tokens(conversation, model)


def main(model: str):
    data_tokens = []
    for ds in DATASETS:
        for split in ["train", "dev", "test"]:
            fname = get_dataset_file(f"{ds.name}/{split}", is_prompt=True)[0]
            if not fname.exists():
                continue
            logger.info(f"processing {fname}")

            for sample in ioutils.jsonl_load(fname):
                tokens_count = count_tokens(sample["conversations"], model)
                data_tokens.append(
                    {
                        "dataset": ds.name,
                        "split": split,
                        "sample_id": sample["sample_id"],
                        "tokens_count": tokens_count,
                    }
                )

    stats_df = pd.DataFrame(data_tokens)
    print("\n\nper split:\n")
    print(stats_df[["split", "tokens_count"]].groupby("split").sum())
    print("\n\nper dataset:\n")
    print(stats_df[["dataset", "tokens_count"]].groupby("dataset").sum())
    print("\n\ntest only:\n")
    print(stats_df.query("split == 'test'")[["dataset", "tokens_count"]].groupby("dataset").sum())


if __name__ == "__main__":
    log.init()
    main(gpt4.GPT4_MODEL)
