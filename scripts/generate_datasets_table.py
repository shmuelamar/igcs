from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from igcs import configs
from igcs.datasets import get_dataset_file
from igcs.entities import Doc, Selection, SelectionSample
from igcs.grounding import tokenization
from igcs.grounding.tokenization import selections_to_token_ids
from igcs.utils import ioutils


@lru_cache
def tokenize_text(txt: str) -> list[str]:
    return tokenization.spacy_tokenize(txt, ignore_tags=())


def get_word_count(s: str | list[str] | list[Selection] | list[Doc]):
    if not s:
        return 0

    if isinstance(s, (str, Doc, Selection)):
        s = [s]

    # handle docs
    if isinstance(s[0], Doc):
        s = [d.text for d in s]

    # handle Selections
    if isinstance(s[0], Selection):
        s = [sel.content for sel in s]

    return sum(len(tokenize_text(si)) for si in s)


def avg(lst: list) -> float:
    if not lst:
        return 0
    return sum(lst) / len(lst)


def selection_dist(sel_a: Selection, sel_b: Selection, docs) -> float:
    assert sel_a.doc_id == sel_b.doc_id
    _, token_ids_a = selections_to_token_ids([sel_a], docs)
    _, token_ids_b = selections_to_token_ids([sel_b], docs)
    doc_tokens = get_word_count(docs[sel_a.doc_id].text)

    min_token_a = min(t for d, t in token_ids_a)
    max_token_a = max(t for d, t in token_ids_a)
    min_token_b = min(t for d, t in token_ids_b)
    max_token_b = max(t for d, t in token_ids_b)

    return min(abs(min_token_a - max_token_b), abs(max_token_a - min_token_b)) / doc_tokens


def compute_connectivity(
    sel_idx: int, selections: list[Selection], docs: list[Doc]
) -> float | None:
    # no close spans - cannot compute connectivity
    if len(selections) < 2:
        return None

    # distance between two selections is 1 if they are on other docs, otherwise it the number of tokens separating
    # between them as a ratio between the document total tokens.
    sel = selections[sel_idx]
    return min(
        [
            (
                selection_dist(sel, other_sel, docs) / len(docs[sel.doc_id].text)
                if sel.doc_id == other_sel.doc_id
                else 1.0
            )
            for other_sel_idx, other_sel in enumerate(selections)
            if sel_idx != other_sel_idx
        ]
    )


def get_stats(sample: SelectionSample) -> dict:
    stats = {
        "sources_words_count": get_word_count(sample.docs),
        "selections_words_counts": [get_word_count(ref.selections) for ref in sample.selections],
        "selection_words_counts": [
            [get_word_count(s) for s in ref.selections] for ref in sample.selections
        ],
        "selections_counts": [len(ref.selections) for ref in sample.selections],
    }
    stats["avg_selections_count"] = avg(stats["selections_counts"])
    stats["avg_selections_words_count"] = avg(stats["selections_words_counts"])
    stats["avg_selection_words_count"] = avg(
        [s for ref in stats["selection_words_counts"] for s in ref]
    )
    stats["median_selection_words_count"] = avg(
        [np.median(ref or [0.0]) for ref in stats["selection_words_counts"]]
    )
    scatteredness = [
        [
            compute_connectivity(sel_idx, ref.selections, sample.docs)
            for sel_idx, _ in enumerate(ref.selections)
        ]
        for ref in sample.selections
    ]
    avg_scatterness = avg([avg([si for si in s if si is not None]) for s in scatteredness])
    stats["avg_src_coverage"] = avg(
        [
            len({sel.doc_id for sel in ref.selections}) / len(sample.docs)
            for ref in sample.selections
        ]
    )
    stats["scatteredness"] = scatteredness
    stats["avg_scatteredness"] = avg_scatterness
    stats["avg_compression"] = stats["avg_selections_words_count"] / stats["sources_words_count"]
    stats["avg_density"] = avg(
        [
            sum([x**2 for x in ref]) / (cnt or 1.0)
            for ref, cnt in zip(
                stats["selection_words_counts"],
                stats["selections_words_counts"],
                strict=True,
            )
        ]
    )

    if not sample.instruction_context:
        term = ""
    elif sample.source_dataset == "AspectNews":
        term = "One"  # single word aspect name
    else:
        for key in [
            "claim",
            "query",
            "aspect",
            "aspect_label",
            "instruction",
            "argument",
        ]:
            try:
                term = sample.instruction_context[key]
                break
            except KeyError:
                continue
        else:
            raise ValueError(sample)

    stats["primary_term_words_count"] = get_word_count(term)
    stats["empty_selection"] = all((not ref.selections) for ref in sample.selections)
    return stats


def load_all_datasets():
    datas = []
    for split in ["train", "dev", "test"]:
        fname = get_dataset_file(f"all/{split}")[0]
        print(f"loading data from {fname}")

        for row in ioutils.jsonl_load(fname):
            datas.append(
                {
                    "id": row["id"],
                    "source_dataset": row["source_dataset"],
                    "split": split,
                    "type": row["type"],
                    "sample": SelectionSample(**row),
                    **{f"ctx.{k}": v for k, v in (row["instruction_context"] or {}).items()},
                }
            )

        majority_fname = (
            configs.ROOT_DIR
            / "igcs-dataset-majority"
            / split
            / f"{split}-ReverseInstructions.jsonl.gz"
        )
        for row in ioutils.jsonl_load(majority_fname):
            datas.append(
                {
                    "id": row["id"],
                    "source_dataset": row["source_dataset"] + "-Majority",
                    "split": split,
                    "type": row["type"],
                    "sample": SelectionSample(**row),
                    **{f"ctx.{k}": v for k, v in (row["instruction_context"] or {}).items()},
                }
            )

    return pd.DataFrame(datas)


def compute_stats_dataframe(sample_size: int | None = None):
    src_df = load_all_datasets()
    if sample_size:
        src_df = src_df.sample(sample_size, random_state=42).reset_index(drop=True)

    all_stats_df = pd.DataFrame(get_stats(s) for s in tqdm(src_df["sample"]))
    return pd.concat([src_df, all_stats_df], axis=1)


def main():
    df = compute_stats_dataframe(sample_size=1000)

    relevant_fields = [
        "sources_words_count",
        "avg_selections_count",
        "avg_selections_words_count",
        "avg_selection_words_count",
        "median_selection_words_count",
        "avg_src_coverage",
        "avg_density",
        "avg_scatteredness",
        "avg_compression",
        "primary_term_words_count",
        "empty_selection",
    ]

    def _capitalize_name(col):
        return " ".join([w.capitalize() for w in col.split("_")])

    mean_df = df[relevant_fields + ["source_dataset"]].groupby("source_dataset").agg(np.mean)
    mean_df = mean_df.round(2)
    mean_df.rename(columns=_capitalize_name, inplace=True)
    mean_df.index.name = _capitalize_name(mean_df.index.name)
    mean_df.index = mean_df.index.to_series().apply(_capitalize_name)
    mean_df.to_csv("datasets_stats.csv")
    print(mean_df.to_latex())


if __name__ == "__main__":
    # #Instances
    # Sources #Words
    # Selection #Words
    # Multi-documents? - bool
    #
    # Granularity - spans / sentences
    # Empty Selections - bool
    main()
