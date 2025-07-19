import logging
from pathlib import Path
from typing import Literal

from igcs import configs
from igcs.datasets import prepare_igcs_datasets
from igcs.datasets.create_gencs_dataset import (
    create_reverse_instructions_dataset_splits,
)
from igcs.utils import log

logger = logging.getLogger(__name__)


def create_dataset(
    *,
    dataset_name: str,
    min_selection_votes: int,
    skip_empty: bool,
    prompts_merge_mode: Literal["best", "merge", "model:GPT4"],
    paraphrase: bool,
    k_samples_per_document_set: int | None,
    filter_hallucinations: bool,
    selected_models: list[str] | None,
    annotation_fname: Path = configs.ANNOTATED_RI_FNAME,
    num_annotations_per_sample: int = 3,
):
    ri_dataset_dir = configs.RI_DATASET_DIR / dataset_name
    dataset_dir = configs.ROOT_DIR / f"igcs-dataset-{dataset_name}"

    if paraphrase:
        paraphrase_fname = configs.SRC_DATASETS_DIR / "paraphrases.json"
    else:
        paraphrase_fname = None

    ri_dataset_dir.mkdir(exist_ok=True, parents=True)

    create_reverse_instructions_dataset_splits.main(
        infile=annotation_fname,
        outdir=ri_dataset_dir,
        min_selection_votes=min_selection_votes,
        skip_empty=skip_empty,
        k_samples_per_document_set=k_samples_per_document_set,
        filter_hallucinations=filter_hallucinations,
        selected_models=selected_models,
        num_annotations_per_sample=num_annotations_per_sample,
    )
    prepare_igcs_datasets.main(
        ri_dirname=ri_dataset_dir,
        outdir=dataset_dir,
        prompts_merge_mode=prompts_merge_mode,
        paraphrase_fname=paraphrase_fname,
        verify_hallucinations=not filter_hallucinations,
    )


def create_all_datasets():
    create_dataset(
        dataset_name="majority",
        min_selection_votes=2,
        skip_empty=False,
        prompts_merge_mode="merge",
        paraphrase=False,
        k_samples_per_document_set=None,
        filter_hallucinations=True,
        selected_models=None,
    )

    create_dataset(
        dataset_name="union",
        min_selection_votes=1,
        skip_empty=False,
        prompts_merge_mode="merge",
        paraphrase=False,
        k_samples_per_document_set=None,
        filter_hallucinations=True,
        selected_models=None,
    )

    create_dataset(
        dataset_name="gpt4",
        min_selection_votes=1,  # not applicable
        skip_empty=False,
        prompts_merge_mode="model:GPT4",
        paraphrase=False,
        k_samples_per_document_set=None,
        filter_hallucinations=True,
        selected_models=None,
    )

    # Additional controls for the datasets
    create_dataset(
        dataset_name="intersect",
        min_selection_votes=3,
        skip_empty=False,
        prompts_merge_mode="merge",
        paraphrase=False,
        k_samples_per_document_set=None,
        filter_hallucinations=True,
        selected_models=None,
    )

    create_dataset(
        dataset_name="baseline",
        min_selection_votes=1,  # not applicable
        skip_empty=False,
        prompts_merge_mode="model:GPT4",
        paraphrase=False,
        k_samples_per_document_set=1,
        filter_hallucinations=False,
        selected_models=None,  # not applicable
    )

    create_dataset(
        dataset_name="baseline-k5",
        min_selection_votes=1,  # not applicable
        skip_empty=False,
        prompts_merge_mode="model:GPT4",
        paraphrase=False,
        k_samples_per_document_set=5,
        filter_hallucinations=False,
        selected_models=None,  # not applicable
    )

    # K samples per document set ablation
    for k in [1, 2, 3, 4]:
        create_dataset(
            dataset_name=f"k{k}-union",
            min_selection_votes=1,
            skip_empty=False,
            prompts_merge_mode="merge",
            paraphrase=False,
            k_samples_per_document_set=k,
            filter_hallucinations=True,
            selected_models=None,
        )

    create_dataset(
        dataset_name="union-gpt4-gemini",
        min_selection_votes=1,
        skip_empty=False,
        prompts_merge_mode="merge",
        paraphrase=False,
        k_samples_per_document_set=None,
        filter_hallucinations=True,
        selected_models=["GPT4", "Gemini-1.5"],
    )

    # single Step with single question baseline
    create_dataset(
        dataset_name="single-step",
        min_selection_votes=1,  # not applicable
        skip_empty=False,
        prompts_merge_mode="model:GPT4",
        paraphrase=False,
        k_samples_per_document_set=1,
        filter_hallucinations=True,
        selected_models=None,  # not applicable
        num_annotations_per_sample=1,  # we only have a single annotation
        annotation_fname=configs.ANNOTATED_RI_DIR
        / "annotated_ri_samples_single_step_n500_k1.jsonl.gz",
    )


if __name__ == "__main__":  # pragma: nocover
    log.init()
    create_all_datasets()
