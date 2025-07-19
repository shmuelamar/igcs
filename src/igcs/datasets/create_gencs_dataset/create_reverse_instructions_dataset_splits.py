import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from igcs import configs
from igcs.cseval import score_selection
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample
from igcs.grounding.tokenization import selections_to_token_ids, token_ids_to_selections
from igcs.utils import ioutils, log

# split train / dev / test
SPLITS_RATIO = (0.7, 0.1, 0.2)

logger = logging.getLogger(__name__)


def get_sample_ila(references: list[SelectionGroup], docs: list[Doc]):
    n_refs = len(references)
    pair_scores = {}
    for i in range(n_refs):
        for j in range(i + 1, n_refs):
            selections_i = references[i]
            selections_j = references[j]
            pair_scores[(selections_i.id, selections_j.id)] = score_selection(
                selections_i.selections, selections_j.selections, docs
            )["f1"]

    ila_score = sum(pair_scores.values()) / len(pair_scores)
    return ila_score, pair_scores


def filter_ri_dataset(
    ri_df: pd.DataFrame,
    min_ila_score: float | None = None,
    min_selection_votes: int = 0,
    skip_empty: bool = True,
    skip_all_hallucination: bool = False,
    k_samples_per_document_set: int | None = None,
    filter_hallucinations: bool = True,
    selected_models: list[str] | None = None,
    num_annotations_per_sample: int = 3,
) -> pd.DataFrame:
    stats = defaultdict(int)
    all_filtered_samples = []
    for samples in ri_df["samples"]:
        filtered_samples = []
        for raw_sample in samples:
            sample = SelectionSample(**raw_sample)
            # sort by annotator model_id
            sample.selections.sort(key=lambda x: x.id)

            # 1. always append negative samples
            if sample.instruction_context["is_negative"]:
                filtered_samples.append(sample.model_dump())
                continue

            # 2. for every selection group, filter hallucinations and drop the selection group
            #    if it is solely from hallucinated selected spans.
            if filter_hallucinations:
                filtered_selections_groups = []
                for selections in sample.selections:
                    selections.selections = [s for s in selections.selections if s.doc_id >= 0]

                    if selections.selections or skip_all_hallucination:
                        filtered_selections_groups.append(selections)
                    else:
                        stats["empty_selections"] += 1

                sample.selections = filtered_selections_groups

            # 3. filter less than 3/3 answers
            if len(sample.selections) < num_annotations_per_sample:
                stats["filtered.partial_annotators"] += 1
                continue

            # 4. filter disperse selections
            if min_ila_score is not None:
                ila_score, pair_scores = get_sample_ila(sample.selections, sample.docs)
                logger.info(f"ILA score: {ila_score:.3f}, pairwise scores: {pair_scores}")
                if ila_score < min_ila_score:
                    stats["filtered.low_agreement"] += 1
                    continue

            merged_selections = merge_sample_selections(
                sample, min_selection_votes, selected_models
            )
            if not merged_selections:
                if skip_empty:
                    logger.info("skipping empty merged selection - not enough votes")
                    stats["filtered.empty_merge"] += 1
                    continue
                else:
                    logger.info("found empty merged selection - not enough votes")

            sample.selections.append(SelectionGroup(id="Merged", selections=merged_selections))

            filtered_samples.append(sample.model_dump())

        # Take upto k filtered samples per document set, by default take all
        if k_samples_per_document_set:
            filtered_samples = filtered_samples[:k_samples_per_document_set]
        all_filtered_samples.append(filtered_samples)

    logger.info(f"filtering stats: {stats}")
    ri_df["filtered_samples"] = all_filtered_samples
    return ri_df


def merge_sample_selections(
    sample: SelectionSample, min_selection_votes: int, selected_models: list[str] | None = None
) -> list[Selection]:
    sel_indices = [
        selections_to_token_ids(ref.selections, sample.docs)[1]
        for ref in sample.selections
        if selected_models is None or ref.id in selected_models
    ]
    tokens_freqs = Counter(idx for indices in sel_indices for idx in indices)
    selected_tokens = [token for token, freq in tokens_freqs.items() if freq >= min_selection_votes]

    merged_selections = token_ids_to_selections(selected_tokens, sample.docs)

    scores = []
    for candidate in sample.selections:
        scores.append(
            round(
                score_selection(merged_selections, candidate.selections, sample.docs)["f1"],
                3,
            )
        )

    max_len = max(len(candidate.selections) for candidate in sample.selections)
    logger.info(
        f"merged {len(merged_selections)}/{max_len} selections f1 scores: {scores}. avg: {sum(scores) / len(scores)}"
    )
    return merged_selections


def split_ri_dataset(ri_df: pd.DataFrame) -> dict[str, list[dict]]:
    """load RI dataset (Reverse Instructions)"""

    size = len(ri_df)
    train_size = int(size * SPLITS_RATIO[0])
    dev_size = int(size * SPLITS_RATIO[1])
    test_size = size - train_size - dev_size

    logger.info(
        f"splitting {size} samples into 3 sets: train size {train_size}, dev size {dev_size}, test size {test_size}"
    )
    # test size inferred from left

    ri_train_and_dev_df, ri_test_df = train_test_split(
        ri_df, test_size=test_size, random_state=42, stratify=ri_df["type"]
    )
    ri_train_df, ri_dev_df = train_test_split(
        ri_train_and_dev_df,
        train_size=train_size,
        test_size=dev_size,
        random_state=42,
        stratify=ri_train_and_dev_df["type"],
    )

    # sanity checks
    assert len(ri_test_df) + len(ri_dev_df) + len(ri_train_df) == len(ri_df)
    assert ri_train_df["id"].is_unique
    assert ri_dev_df["id"].is_unique
    assert ri_test_df["id"].is_unique
    assert set(ri_train_df["id"]) & set(ri_dev_df["id"]) == set()
    assert set(ri_train_df["id"]) & set(ri_test_df["id"]) == set()
    assert set(ri_dev_df["id"]) & set(ri_test_df["id"]) == set()

    logger.info("passed sanity checks for contamination of splits and uniqueness")

    split2samples = {}
    for split, split_df in [
        ("train", ri_train_df),
        ("dev", ri_dev_df),
        ("test", ri_test_df),
    ]:
        split_samples = []
        # Every row is actually multiple samples as we create multiple instructions per document set.
        for batch_samples in split_df["filtered_samples"]:
            split_samples += batch_samples

        logger.info(f"found {len(split_samples)} samples for {split} split")
        split2samples[split] = split_samples
    return split2samples


def save_ri_split(outdir: str | Path, split, samples: list[dict]):
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = Path(outdir) / f"{split}.jsonl.gz"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"saving {len(samples)} samples of {split} split to {outfile}")
    ioutils.jsonl_dump(samples, outfile)


def main(
    infile: Path,
    outdir: Path,
    min_selection_votes: int,
    skip_empty: bool,
    k_samples_per_document_set: int | None = None,
    filter_hallucinations: bool = True,
    selected_models: list[str] | None = None,
    num_annotations_per_sample: int = 3,
):
    logger.info("loading annotated dataset")
    ri_df = pd.read_json(infile, lines=True)
    assert ri_df["id"].is_unique

    ri_df = filter_ri_dataset(
        ri_df,
        min_ila_score=None,
        min_selection_votes=min_selection_votes,
        skip_empty=skip_empty,
        k_samples_per_document_set=k_samples_per_document_set,
        filter_hallucinations=filter_hallucinations,
        selected_models=selected_models,
        num_annotations_per_sample=num_annotations_per_sample,
    )
    split2samples = split_ri_dataset(ri_df)
    for split, samples in split2samples.items():
        save_ri_split(outdir, split, samples)


if __name__ == "__main__":  # pragma: nocover
    log.init()
    # original Union mode
    main(
        infile=configs.ANNOTATED_RI_FNAME,
        outdir=configs.RI_DATASET_DIR,
        min_selection_votes=0,
        skip_empty=True,
    )
