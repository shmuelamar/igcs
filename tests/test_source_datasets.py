import random
from collections import Counter

import tiktoken
from rich.console import Console

from igcs.datasets.igcsbench_datasets import DATASETS
from igcs.utils import prompting


def test_create_all_datasets_sanity():
    # calling prepare dataset verify each sample integrity internally with assertions
    console = Console(width=128)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    assert len(DATASETS)

    total_tokens = {}
    all_total_counts = Counter()
    for dataset in DATASETS:
        total_counts = Counter()
        ds = dataset()

        for split, samples in ds.prepare_dataset().items():
            total_counts.update(
                sel.metadata["total_count"]
                for sample in samples
                for ref in sample.selections
                for sel in ref.selections
            )
            assert len(samples)
            assert split in ["train", "dev", "test"]

            console.print(ds.name)
            if split == "test":
                total_tokens[ds.name] = sum(
                    len(tokenizer.encode(prompting.format_prompt(sample))) for sample in samples
                )
                random.seed(42)
                console.print(random.sample(samples, 3))
        all_total_counts += total_counts
        console.print(total_counts.most_common())

    console.print(all_total_counts.most_common())
    console.print(f"Total tokens {total_tokens}")
