import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from igcs import configs
from igcs.datasets.igcsbench_datasets import DATASETS, ReverseInstructionsDataset
from igcs.entities import Doc, Selection, SelectionGroup, SelectionSample
from igcs.evaluate import evaluation_functions
from igcs.grounding.tokenization import selections_to_token_ids
from igcs.utils import ioutils, log, prompting
from igcs.utils.prompting import SYSTEM_ROLE, selections_to_answer

logger = logging.getLogger(__name__)


def get_prompt_item(
    sample_id: str, instruction: str, answer: str, system: str | None = None
) -> dict:
    # conversation format
    role_key = "role"
    content_key = "content"
    msgs_key = "conversations"

    messages = []
    if system:
        messages.append({role_key: "system", content_key: system})

    messages += [
        {role_key: "user", content_key: instruction},
        {role_key: "assistant", content_key: answer},
    ]
    return {"sample_id": sample_id, msgs_key: messages}


def get_best_selection(
    selections: list[list[Selection] | SelectionGroup], docs: list[Doc], sample_id: str
):
    if selections and isinstance(selections[0], SelectionGroup):
        id2name = [s.id for s in selections]
        selections = [s.selections for s in selections]
    else:
        id2name = list(range(len(selections)))

    # first, filter hallucinations and empty matches
    selections = [[s for s in sels if s.doc_id >= 0 and s.content] for sels in selections]

    # convert matches to char-level indices
    sel_indices = [selections_to_token_ids(sels, docs)[1] for sels in selections]

    # build similarity matrix between answers
    similarity = np.zeros((len(selections), len(selections)))
    for i, sels_i in enumerate(sel_indices):
        for j, sels_j in enumerate(sel_indices[i:], start=i):
            ij_f1 = evaluation_functions.unigram_score(sels_i, sels_j)["f1"]
            similarity[i, j] = ij_f1
            similarity[j, i] = ij_f1

    # find best answer based on similarity
    avg_sim_scores = similarity.sum(axis=0) / len(similarity)
    best_answer_idx = avg_sim_scores.argmax()
    best_selection = selections[best_answer_idx]
    logger.info(
        f"average similarities: {avg_sim_scores.round(3)}. "
        f"{sample_id} best answer - {id2name[best_answer_idx]} with {len(best_selection)} selections"
    )
    return best_selection


def get_merged_selection_merged(sample: SelectionSample, model: str = "Merged") -> list[Selection]:
    # special case on ReverseInstruction dataset - force negative samples to be empty,
    # use the Merged selection for training (i.e. answer) proposes.
    if sample.source_dataset == ReverseInstructionsDataset.name:
        if sample.is_negative:
            merged_selections = []
        else:
            merged_selections = [ref.selections for ref in sample.selections if ref.id == model][0]
    elif len(sample.selections) > 1:
        sels = [s for s in sample.selections if s.id != "Merged"]
        merged_selections = get_best_selection(sels, sample.docs, sample.id)
    else:
        merged_selections = sample.selections[0].selections

    return merged_selections


def get_merged_selection_best(sample: SelectionSample) -> list[Selection]:
    # special case on ReverseInstruction dataset - force negative samples to be empty,
    # use the Merged selection for training (i.e. answer) proposes.
    if sample.source_dataset == ReverseInstructionsDataset.name and sample.is_negative:
        return []
    elif len(sample.selections) > 1:
        sels = [s for s in sample.selections if s.id != "Merged"]
        merged_selections = get_best_selection(sels, sample.docs, sample.id)
    else:
        merged_selections = sample.selections[0].selections
    return merged_selections


def get_merged_selections(sample: SelectionSample, mode: str) -> list[Selection]:
    if mode == "best":
        return get_merged_selection_best(sample)
    elif mode == "merge":
        return get_merged_selection_merged(sample)
    elif mode.startswith("model:"):
        model = mode.removeprefix("model:")
        return get_merged_selection_merged(sample, model=model)
    else:
        raise ValueError(f"unknown mode {mode}")


def prepare_prompts(
    samples: list[SelectionSample],
    mode: Literal["best", "merge", "model:GPT4"],
    paraphrases: dict[str, str] | None,
) -> list[dict]:
    prompts = []
    for sample in samples:
        # these samples have no paraphrases
        if paraphrases and sample.id in paraphrases:
            input_prompt = prompting.format_prompt(sample, instruction=paraphrases[sample.id])
        else:
            input_prompt = prompting.format_prompt(sample)
        merged_selections = get_merged_selections(sample, mode=mode)

        expected_answer = selections_to_answer(merged_selections)

        prompt_dict = get_prompt_item(
            sample.id,
            instruction=input_prompt,
            answer=expected_answer,
            system=SYSTEM_ROLE,
        )
        prompts.append(prompt_dict)
    return prompts


def main(
    ri_dirname: Path,
    outdir: Path,
    prompts_merge_mode: Literal["best", "merge", "model:GPT4"],
    paraphrase_fname: Path | None,
    verify_hallucinations: bool = True,
):
    prompts_outdir = outdir / "prompts"
    prompts_outdir.mkdir(exist_ok=True, parents=True)
    split2dir = {
        "train": outdir / "train",
        "dev": outdir / "dev",
        "test": outdir / "test",
    }
    prompt_split2dir = {
        "train": prompts_outdir / "train",
        "dev": prompts_outdir / "dev",
        "test": prompts_outdir / "test",
    }

    if paraphrase_fname:
        logger.info(f"loading paraphrase data from: {paraphrase_fname}")
        with open(paraphrase_fname) as fp:
            paraphrases = json.load(fp)
    else:
        paraphrases = None

    for dataset in DATASETS:
        ds = dataset()
        if isinstance(ds, ReverseInstructionsDataset):
            ds.dirname = ri_dirname

        logger.info(f"creating dataset {ds.name}")

        for split, samples in ds.prepare_dataset(verify_hallucinations).items():
            logger.info(f"creating samples for {ds.name} split {split} with {len(samples)} samples")
            fname = split2dir[split] / f"{split}-{ds.name}.jsonl.gz"
            fname.parent.mkdir(parents=True, exist_ok=True)
            raw_samples = [s.dict() for s in samples]
            ioutils.jsonl_dump(raw_samples, fname)

            # per dataset prompts
            logger.info(f"creating prompts for {ds.name} split {split}")
            prompts = prepare_prompts(samples, mode=prompts_merge_mode, paraphrases=paraphrases)
            prompts_fname = prompts_outdir / split / f"prompts-{split}-{ds.name}.jsonl.gz"
            prompts_fname.parent.mkdir(parents=True, exist_ok=True)
            ioutils.jsonl_dump(prompts, prompts_fname)

    for split, dirname in split2dir.items():
        all_split_samples = [
            sample for fname in dirname.glob("*.jsonl.gz") for sample in ioutils.jsonl_load(fname)
        ]
        logger.info(f"found {len(all_split_samples)} for all {split} datasets")
        ioutils.jsonl_dump(all_split_samples, outdir / f"{split}-all.jsonl.gz")

        # all datasets prompts
        logger.info(f"creating prompts for all-datasets split {split}")

        all_prompts_samples = [
            sample
            for fname in prompt_split2dir[split].glob("*.jsonl.gz")
            for sample in ioutils.jsonl_load(fname)
        ]

        all_prompts_fname = prompts_outdir / f"prompts-{split}-all.jsonl.gz"
        all_prompts_fname.parent.mkdir(parents=True, exist_ok=True)
        ioutils.jsonl_dump(all_prompts_samples, all_prompts_fname)


if __name__ == "__main__":  # pragma: nocover
    log.init()
    main(
        outdir=configs.IGCS_DATA_DIR,
        ri_dirname=configs.RI_DATASET_DIR,
        prompts_merge_mode="merge",
        paraphrase_fname=None,
    )
