import hashlib
import logging
import os
from glob import glob
from pathlib import Path

import pandas as pd

from igcs import configs, cseval
from igcs.predict import get_canonic_fname
from igcs.utils import log
from igcs.utils.diskcache import disk_cache
from scripts.compute_overall_scores_and_ci import compute_overall_scores_with_ci

logger = logging.getLogger(__name__)


def parse_fname(fname: str):
    # test-AspectNews-trainer_models_igcs-ri-only-v2.jsonl.gz
    _, dataset_name, model_name_and_ext = os.path.basename(fname).split("-", 2)
    model_name = model_name_and_ext.removesuffix(".jsonl.gz")
    # dataset_file = configs.TEST_DIR / f"test-{dataset_name}.jsonl.gz"
    return f"{dataset_name}/test", dataset_name, model_name


COMMON_MODEL_PREFIX = "trainer_models_llama-3_"
NAME_TO_MODEL = {
    # Qwen Models
    "Qwen2.5-0.5B": "Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B": "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen_Qwen2.5-3B-Instruct",
    "Qwen2.5-7B": "Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-0.5B-ICL": "ICL-Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-ICL": "ICL-Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-ICL": "ICL-Qwen_Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-ICL": "ICL-Qwen_Qwen2.5-7B-Instruct",
    # SmolLM2 Trained Models
    "SmolLM2-135M-Union": "trainer_models_smollm2_smollm2-135M-rionly-union",
    "SmolLM2-360M-Union": "trainer_models_smollm2_smollm2-360M-rionly-union",
    "SmolLM2-1.7B-Union": "trainer_models_smollm2_smollm2-1.7B-rionly-union",
    "SmolLM2-135M-Majority": "trainer_models_smollm2_smollm2-135M-rionly-majority",
    "SmolLM2-360M-Majority": "trainer_models_smollm2_smollm2-360M-rionly-majority",
    "SmolLM2-1.7B-Majority": "trainer_models_smollm2_smollm2-1.7B-rionly-majority",
    # SmolLM2 Prompt Models
    "SmolLM2-135M": "HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-360M": "HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-1.7B": "HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "SmolLM2-135M-ICL": "ICL-HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-360M-ICL": "ICL-HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-ICL": "ICL-HuggingFaceTB_SmolLM2-1.7B-Instruct",
    # Qwen Trained Models
    "RIOnly-Majority-Qwen-0.5b": "trainer_models_qwen2.5-0.5b_qwen2.5-0.5b-rionly-majority",
    "RIOnly-Majority-Qwen-1.5b": "trainer_models_qwen2.5-1.5b_qwen2.5-1.5b-rionly-majority",
    "RIOnly-Majority-Qwen-3b": "trainer_models_qwen2.5-3b_qwen2.5-3b-rionly-majority",
    "RIOnly-Majority-Qwen-7b": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-majority",
    "RIOnly-Union-Qwen-0.5b": "trainer_models_qwen2.5-0.5b_qwen2.5-0.5b-rionly-union",
    "RIOnly-Union-Qwen-1.5b": "trainer_models_qwen2.5-1.5b_qwen2.5-1.5b-rionly-union",
    "RIOnly-Union-Qwen-3b": "trainer_models_qwen2.5-3b_qwen2.5-3b-rionly-union",
    "RIOnly-Union-Qwen-7b": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union",
    # Gemma Models
    "gemma3-1b": "google_gemma-3-1b-it",
    "gemma3-1b-ICL": "ICL-google_gemma-3-1b-it",
    # Gemma Trained Models
    "RIOnly-Union-gemma3-1b": "trainer_models_gemma-3-1b_gemma-3-1b-rionly-union",
    # Zero Shot Models (from APIs)
    "Claude3-Opus": "Claude3-Opus",
    "GPT4": "GPT4",
    "GPT4-ICL": "ICL-GPT4",
    "llama-3-405B": "llama3-405B",
    "llama-3-70B": "llama3-70B",
    "llama-3-70B-ICL": "ICL-llama3-70B",
    "Llama-3-8B": "meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL": "ICL-meta-llama_Meta-Llama-3-8B-Instruct",
    # RI Models
    "RIOnly-Union": f"{COMMON_MODEL_PREFIX}rionly-union-retrain",
    "RIOnly-Majority": f"{COMMON_MODEL_PREFIX}rionly-majority-retrain",
    # LOO Vanilla Models
    "LOO-EVR": f"{COMMON_MODEL_PREFIX}loo-ub-scifact-dsfix",
    "LOO-ASPSEL": f"{COMMON_MODEL_PREFIX}loo-ub-openasp-dsfix",
    "LOO-EXTABS": f"{COMMON_MODEL_PREFIX}loo-ub-aspectnews-dsfix",
    "LOO-ARG": f"{COMMON_MODEL_PREFIX}loo-ub-debatesum-dsfix",
    "SCOnly": f"{COMMON_MODEL_PREFIX}igcs-sconly-ub-dsfix",
    # LOO+RI Models
    "LOO-RI-Majority-EVR": f"{COMMON_MODEL_PREFIX}loo-ub-ri-majority-scifact-dsfix",
    "LOO-RI-Union-ASPSEL": f"{COMMON_MODEL_PREFIX}loo-ub-ri-union-openasp-dsfix",
    "LOO-RI-Union-EXTABS": f"{COMMON_MODEL_PREFIX}loo-ub-ri-union-aspectnews-dsfix",
    "LOO-RI-Union-ARG": f"{COMMON_MODEL_PREFIX}loo-ub-ri-union-debatesum",
    "Full-Union": f"{COMMON_MODEL_PREFIX}igcs-full-union-ub-dsfix",
    "Full-Majority": f"{COMMON_MODEL_PREFIX}igcs-full-majority-ub-dsfix",
    # SFT Vanilla Models
    "SFT-EVR": f"{COMMON_MODEL_PREFIX}sft-ub-scifact",
    "SFT-EXTABS": f"{COMMON_MODEL_PREFIX}sft-ub-aspectnews",
    "SFT-ARG": f"{COMMON_MODEL_PREFIX}sft-ub-debatesum-dsfix",
    # SFT+RI Models
    "SFT-RI-Union-EVR": f"{COMMON_MODEL_PREFIX}sft-ub-ri-union-scifact",
    "SFT-RI-Union-ARG": f"{COMMON_MODEL_PREFIX}sft-debatesum-ri-union-dsfix",
    "SFT-RI-Union-EXTABS": f"{COMMON_MODEL_PREFIX}sft-ub-ri-union-aspectnews",
    "SFT-RI-Majority-EVR": f"{COMMON_MODEL_PREFIX}sft-ub-ri-majority-scifact",
    "SFT-RI-Majority-ARG": f"{COMMON_MODEL_PREFIX}sft-ub-ri-majority-debatesum",
    "SFT-RI-Majority-EXTABS": f"{COMMON_MODEL_PREFIX}sft-ub-ri-majority-aspectnews",
    # Llama-3 GenCS Ablation Models
    "Llama-3-8B-GenCS-Baseline": f"{COMMON_MODEL_PREFIX}rionly-baseline",
    "Llama-3-8B-GenCS-Single-Step": f"{COMMON_MODEL_PREFIX}rionly-single-step",
    "Llama-3-8B-GenCS-gpt4": f"{COMMON_MODEL_PREFIX}igcs-rionly-gpt4",
    "Llama-3-8B-GenCS-gpt4-gemini": f"{COMMON_MODEL_PREFIX}rionly-union-gpt4-gemini",
    "Llama-3-8B-GenCS-k1": f"{COMMON_MODEL_PREFIX}rionly-union-k1",
    "Llama-3-8B-GenCS-k2": f"{COMMON_MODEL_PREFIX}rionly-union-k2",
    "Llama-3-8B-GenCS-k3": f"{COMMON_MODEL_PREFIX}rionly-union-k3",
    "Llama-3-8B-GenCS-k4": f"{COMMON_MODEL_PREFIX}rionly-union-k4",
    # Qwen GenCS Ablation Models
    "Qwen2.5-7B-GenCS-Baseline": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-baseline",
    "Qwen2.5-7B-GenCS-Single-Step": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-single-step",
    "Qwen2.5-7B-GenCS-gpt4": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-gpt4",
    "Qwen2.5-7B-GenCS-gpt4-gemini": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union-gpt4-gemini",
    "Qwen2.5-7B-GenCS-k1": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union-k1",
    "Qwen2.5-7B-GenCS-k2": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union-k2",
    "Qwen2.5-7B-GenCS-k3": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union-k3",
    "Qwen2.5-7B-GenCS-k4": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union-k4",
    "Qwen2.5-7B-intersect": "trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-intersect",
    # Qwen GenCS Ablation Models
    "SmolLM2-GenCS-Baseline": "trainer_models_smollm2_smollm2-1.7B-rionly-baseline",
    "SmolLM2-1.7B-GenCS-Single-Step": "trainer_models_smollm2_smollm2-1.7B-rionly-single-step",
    "SmolLM2-GenCS-gpt4": "trainer_models_smollm2_smollm2-1.7B-rionly-gpt4",
    "SmolLM2-GenCS-k1": "trainer_models_smollm2_smollm2-1.7B-rionly-union-k1",
    # Prompt Robustness ICL (Llama3 and Qwen2.5) x (P0 and P1) x (N1 and N3 and RND)
    "Llama-3-8B-ICL-N1": "ICL-N1-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL-N3": "ICL-N3-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL-RND": "ICL-RND-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL-P1-N1": "ICL-P1-N1-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL-P1-N3": "ICL-P1-N3-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL-P1-RND": "ICL-P1-RND-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-ICL-P1": "ICL-P1-meta-llama_Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B-ICL-N1": "ICL-N1-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-ICL-N3": "ICL-N3-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-ICL-RND": "ICL-RND-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-ICL-P1-N1": "ICL-P1-N1-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-ICL-P1-N3": "ICL-P1-N3-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-ICL-P1-RND": "ICL-P1-RND-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-0.5B-ICL-N1": "ICL-N1-Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-ICL-N1": "ICL-N1-Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-ICL-N1": "ICL-N1-Qwen_Qwen2.5-3B-Instruct",
    # Qwen & SmolLM2 all sizes ICL Prompt Variants
    "Qwen2.5-0.5B-ICL-P1": "ICL-P1-Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-0.5B-ICL-P2": "ICL-P2-Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-ICL-P1": "ICL-P1-Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-1.5B-ICL-P2": "ICL-P2-Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-ICL-P1": "ICL-P1-Qwen_Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-ICL-P2": "ICL-P2-Qwen_Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-ICL-P1": "ICL-P1-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-ICL-P2": "ICL-P2-Qwen_Qwen2.5-7B-Instruct",
    "SmolLM2-135M-ICL-P1": "ICL-P1-HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-135M-ICL-P2": "ICL-P2-HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-360M-ICL-P1": "ICL-P1-HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-360M-ICL-P2": "ICL-P2-HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-ICL-P1": "ICL-P1-HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "SmolLM2-1.7B-ICL-P2": "ICL-P2-HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "SmolLM2-135M-ICL-N1": "ICL-N1-HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-360M-ICL-N1": "ICL-N1-HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-ICL-N1": "ICL-N1-HuggingFaceTB_SmolLM2-1.7B-Instruct",
    # Prompt Robustness ZS
    "Llama-3-8B-P1": "P1-meta-llama_Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-P2": "P2-meta-llama_Meta-Llama-3-8B-Instruct",
    "Qwen2.5-0.5B-P1": "P1-Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-0.5B-P2": "P2-Qwen_Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-P1": "P1-Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-1.5B-P2": "P2-Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-P1": "P1-Qwen_Qwen2.5-3B-Instruct",
    "Qwen2.5-3B-P2": "P2-Qwen_Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-P1": "P1-Qwen_Qwen2.5-7B-Instruct",
    "Qwen2.5-7B-P2": "P2-Qwen_Qwen2.5-7B-Instruct",
    "SmolLM2-135M-P1": "P1-HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-135M-P2": "P2-HuggingFaceTB_SmolLM2-135M-Instruct",
    "SmolLM2-360M-P1": "P1-HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-360M-P2": "P2-HuggingFaceTB_SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-P1": "P1-HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "SmolLM2-1.7B-P2": "P2-HuggingFaceTB_SmolLM2-1.7B-Instruct",
    # Prompt Robustness RiOnly
    # Llama-3
    "RIOnly-Union-P1": f"P1-{COMMON_MODEL_PREFIX}rionly-union-retrain",
    "RIOnly-Union-P2": f"P2-{COMMON_MODEL_PREFIX}rionly-union-retrain",
    "RIOnly-Majority-P1": f"P1-{COMMON_MODEL_PREFIX}rionly-majority-retrain",
    "RIOnly-Majority-P2": f"P2-{COMMON_MODEL_PREFIX}rionly-majority-retrain",
    # Qwen2.5 Prompt Variant RIOnly
    "RIOnly-Majority-Qwen-0.5b-P1": "P1-trainer_models_qwen2.5-0.5b_qwen2.5-0.5b-rionly-majority",
    "RIOnly-Majority-Qwen-0.5b-P2": "P2-trainer_models_qwen2.5-0.5b_qwen2.5-0.5b-rionly-majority",
    "RIOnly-Majority-Qwen-1.5b-P1": "P1-trainer_models_qwen2.5-1.5b_qwen2.5-1.5b-rionly-majority",
    "RIOnly-Majority-Qwen-1.5b-P2": "P2-trainer_models_qwen2.5-1.5b_qwen2.5-1.5b-rionly-majority",
    "RIOnly-Majority-Qwen-3b-P1": "P1-trainer_models_qwen2.5-3b_qwen2.5-3b-rionly-majority",
    "RIOnly-Majority-Qwen-3b-P2": "P2-trainer_models_qwen2.5-3b_qwen2.5-3b-rionly-majority",
    "RIOnly-Majority-Qwen-7b-P1": "P1-trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-majority",
    "RIOnly-Majority-Qwen-7b-P2": "P2-trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-majority",
    "RIOnly-Union-Qwen-0.5b-P1": "P1-trainer_models_qwen2.5-0.5b_qwen2.5-0.5b-rionly-union",
    "RIOnly-Union-Qwen-0.5b-P2": "P2-trainer_models_qwen2.5-0.5b_qwen2.5-0.5b-rionly-union",
    "RIOnly-Union-Qwen-1.5b-P1": "P1-trainer_models_qwen2.5-1.5b_qwen2.5-1.5b-rionly-union",
    "RIOnly-Union-Qwen-1.5b-P2": "P2-trainer_models_qwen2.5-1.5b_qwen2.5-1.5b-rionly-union",
    "RIOnly-Union-Qwen-3b-P1": "P1-trainer_models_qwen2.5-3b_qwen2.5-3b-rionly-union",
    "RIOnly-Union-Qwen-3b-P2": "P2-trainer_models_qwen2.5-3b_qwen2.5-3b-rionly-union",
    "RIOnly-Union-Qwen-7b-P1": "P1-trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union",
    "RIOnly-Union-Qwen-7b-P2": "P2-trainer_models_qwen2.5-7b_qwen2.5-7b-rionly-union",
    "SmolLM2-135M-Union-P1": "P1-trainer_models_smollm2_smollm2-135M-rionly-union",
    "SmolLM2-135M-Union-P2": "P2-trainer_models_smollm2_smollm2-135M-rionly-union",
    "SmolLM2-360M-Union-P1": "P1-trainer_models_smollm2_smollm2-360M-rionly-union",
    "SmolLM2-360M-Union-P2": "P2-trainer_models_smollm2_smollm2-360M-rionly-union",
    "SmolLM2-1.7B-Union-P1": "P1-trainer_models_smollm2_smollm2-1.7B-rionly-union",
    "SmolLM2-1.7B-Union-P2": "P2-trainer_models_smollm2_smollm2-1.7B-rionly-union",
    "SmolLM2-135M-Majority-P1": "P1-trainer_models_smollm2_smollm2-135M-rionly-majority",
    "SmolLM2-135M-Majority-P2": "P2-trainer_models_smollm2_smollm2-135M-rionly-majority",
    "SmolLM2-360M-Majority-P1": "P1-trainer_models_smollm2_smollm2-360M-rionly-majority",
    "SmolLM2-360M-Majority-P2": "P2-trainer_models_smollm2_smollm2-360M-rionly-majority",
    "SmolLM2-1.7B-Majority-P1": "P1-trainer_models_smollm2_smollm2-1.7B-rionly-majority",
    "SmolLM2-1.7B-Majority-P2": "P2-trainer_models_smollm2_smollm2-1.7B-rionly-majority",
}
assert len(set(NAME_TO_MODEL.values())) == len(NAME_TO_MODEL), "duplicates found"

DATASET_LIST = [
    "AspectNews",
    "DebateSum",
    "EvidenceDetection",
    "OpenAsp",
    "SaliencyDetection",
    "SciFact",
    "ReverseInstructions",
    "Overall",
]


@disk_cache(cache_dir=configs.CACHE_DIR / "eval-results")
def _perform_eval(
    *,
    dataset_file,
    fname: str,
    eval_outfile: str,
    is_single_doc,
    model_fullname,
    is_fuzzy,
    dataset_name,
    model_codename,
    model_shortname,
    _content_hash: str,
):
    (
        eval_df,
        eval_results,
        external_scores,
        main_external_score,
        raw_scores,
    ) = cseval.run_cs_eval(
        references=dataset_file,
        predictions=Path(fname),
        quiet=True,
        allow_partial=False,
        output=Path(eval_outfile),
        multi_selections=is_single_doc,
        ignore_parsing_errors=False,
        fuzz=is_fuzzy,
    )
    print(model_fullname, dataset_name, eval_results)
    hallucinations_rate = eval_results["hallucinations_rate"]
    eval_results = {
        (
            f"ext-{metric}" if metric in ["f1", "f1-ci", "iou", "precision", "recall"] else metric
        ): score
        for metric, score in eval_results.items()
    }

    if main_external_score is not None:
        print(f"External main score: {main_external_score:.2f} - {external_scores}")

    grounded_selection = (
        eval_df.copy()["grounded_selection"]
        .apply(lambda s: [si.model_dump() for si in s] if s else s)
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "model_codename": model_codename,
        "model": model_shortname,
        "model_fullname": model_fullname,
        "is_fuzzy": is_fuzzy,
        "is_single_doc": is_single_doc,
        "dataset_name": dataset_name,
        "error": False,
        "orig_score": main_external_score,
        "hal": round((hallucinations_rate or 0.0) * 100, 2),
        "err": eval_results["parsing_errors_ratio"],
        "ext-f1-scores": raw_scores["f1"],
        "grounded_selections": grounded_selection,
        **eval_results,
        **{metric: round(score, 2) for metric, score in (external_scores or {}).items()},
    }


def main(indir: str | Path, model_list: list[str] | None = None):
    indir = Path(indir)
    fnames = glob(str(indir / "test-*.jsonl.gz"))
    fnames.sort(key=lambda x: x[::-1])

    model_to_name = {}
    model_to_idx = {}
    model_codename_list = []
    if model_list is None:
        model_list = []
        for i, (model_shortname, model_fullname) in enumerate(NAME_TO_MODEL.items()):
            model_list.append(model_fullname)
            model_list.append(f"SD-{model_fullname}")
            model_to_name[model_fullname] = model_shortname
            model_to_name[f"SD-{model_fullname}"] = model_shortname
            model_codename_list.append(f"{model_shortname}-fuzz-sd")
            model_codename_list.append(f"{model_shortname}-fuzz-md")
            model_codename_list.append(f"{model_shortname}-norm-sd")
            model_codename_list.append(f"{model_shortname}-norm-md")
            model_to_idx[model_fullname] = i
            model_to_idx[f"SD-{model_fullname}"] = i

    models_results = []

    for fname in fnames:
        logger.debug(f"parsing file {fname}")
        dataset_file, dataset_name, model_fullname = parse_fname(fname)

        if model_fullname not in model_list or dataset_name not in DATASET_LIST:
            logger.debug(f"skipping {model_fullname} {dataset_file}")
            continue

        model_shortname = model_to_name[model_fullname]
        is_single_doc = (
            "SD-" in model_fullname
        )  # single_doc outputs multiple selections - one per doc

        for is_fuzzy in [True, False]:
            model_codename = f'{model_shortname}-{"fuzz" if is_fuzzy else "norm"}-{"sd" if is_single_doc else "md"}'
            logger.info(f"model_name {model_fullname} on {dataset_file}")
            eval_outfile = configs.EVAL_RESULTS_DIR / get_canonic_fname(
                "test", dataset_name, model_fullname
            )
            try:
                with open(fname, "rb") as fp:
                    content_hash = hashlib.sha256(fp.read()).hexdigest()
                models_results.append(
                    _perform_eval(
                        dataset_file=dataset_file,
                        fname=str(fname),
                        eval_outfile=str(eval_outfile),
                        is_single_doc=is_single_doc,
                        model_fullname=model_fullname,
                        is_fuzzy=is_fuzzy,
                        dataset_name=dataset_name,
                        model_codename=model_codename,
                        model_shortname=model_shortname,
                        _content_hash=content_hash,
                    )
                )

            except Exception as exc:
                logger.exception(f"cannot process file {fname} {exc!r}")
                models_results.append(
                    {
                        "model_codename": model_codename,
                        "model": model_shortname,
                        "model_fullname": model_fullname,
                        "is_fuzzy": is_fuzzy,
                        "is_single_doc": is_single_doc,
                        "dataset_name": dataset_name,
                        "f1": -1.0,
                        "ext-f1-ci": -1.0,
                        "ext-f1": -1.0,
                        "ext-f1-scores": [],
                        "grounded_selections": [],
                        "error": True,
                    }
                )

    model2ci = compute_overall_scores_with_ci(models_results, alpha=0.05)
    for model, overall_scores in model2ci.items():
        models_results.append(
            {
                "model_codename": model,
                "dataset_name": "Overall",
                "error": False,
                **{metric: round(score, 2) for metric, score in overall_scores["Overall"].items()},
            }
        )

    df = pd.DataFrame(models_results)

    # save results
    df.to_csv(configs.ROOT_DIR / "agg_results" / "models_results_concat.csv", index=False)
    df.to_json(
        configs.ROOT_DIR / "agg_results" / "models_results_concat.json", indent=2, orient="records"
    )

    # make final df with column per dataset and row per model
    included_cols = [
        col
        for col in [
            "model_codename",
            "f1",
            "precision",
            "recall",
            "rouge1",
            "rouge2",
            "rouge2-p",
            "rouge2-r",
            "rougeL",
            "hal",
            "err",
            "ext-f1",
            "ext-f1-ci",
            "ext-precision",
            "ext-recall",
            "avg_hal",
            "avg_err",
        ]
        if col in df.columns
    ]
    ds_dfs = []
    for ds_name in sorted(set(df["dataset_name"]), key=DATASET_LIST.index):
        ds_df = df[df["dataset_name"] == ds_name][included_cols].copy().set_index("model_codename")
        ds_df.dropna(axis=1, how="all", inplace=True)

        ds_df.columns = [(ds_name, str(c)) for c in ds_df.columns]

        ds_dfs.append(ds_df)

    merged_df = pd.concat(ds_dfs, axis=1).copy()
    merged_df.columns = pd.MultiIndex.from_tuples(
        merged_df.transpose().index, names=["Dataset", "Metrics"]
    )
    merged_df["sort_index"] = merged_df.index.map(model_codename_list.index)
    merged_df.sort_values(["sort_index"], inplace=True)
    merged_df.pop("sort_index")
    merged_df.to_excel(configs.ROOT_DIR / "agg_results" / "models_results.xlsx")


if __name__ == "__main__":
    log.init()
    main(indir=configs.PREDICTIONS_DIR)
