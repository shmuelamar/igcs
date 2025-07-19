import json
from collections import defaultdict

import numpy as np
from scipy.stats import bootstrap
from tqdm import tqdm

from igcs import configs
from igcs.datasets.igcsbench_datasets import (
    AspectNewsDataset,
    DebateSumDataset,
    OpenAspDataset,
    ReverseInstructionsDataset,
    SciFactDataset,
    SparkEvidenceDataset,
    SparkSaliencyDataset,
)

EXPECTED_DATASETS = [
    DebateSumDataset.name,
    AspectNewsDataset.name,
    SciFactDataset.name,
    OpenAspDataset.name,
    SparkSaliencyDataset.name,
    SparkEvidenceDataset.name,
]


def parse_row(row: dict):
    model = row["model_codename"]
    dataset = row["dataset_name"]

    # extract token-level F1 metrics (Extractive F1)
    ext_f1 = row["ext-f1"]
    ext_precision = row["ext-precision"]
    ext_recall = row["ext-recall"]

    # extract metric averages
    if dataset in [
        DebateSumDataset.name,
        AspectNewsDataset.name,
        SciFactDataset.name,
        OpenAspDataset.name,
    ]:
        orig_score = row["orig_score"]
        orig_precision = row["rouge2-p"] if "rouge2-p" in row else row["precision"]
        orig_recall = row["rouge2-r"] if "rouge2-r" in row else row["recall"]

    # these dataset original metric is the token-level F1 metric
    elif dataset in [SparkEvidenceDataset.name, SparkSaliencyDataset.name]:
        orig_score = ext_f1
        orig_precision = ext_precision
        orig_recall = ext_recall
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    ext_all_scores = row["ext-f1-scores"]

    return {
        "dataset": dataset,
        "orig_score": orig_score,
        "orig_precision": orig_precision,
        "orig_recall": orig_recall,
        "ext_f1": ext_f1,
        "ext_precision": ext_precision,
        "ext_recall": ext_recall,
        "all_scores": ext_all_scores,
        "err": row["err"],
        "hal": row["hal"],
    }, model


def macro_avg(*arrs, axis=-1):
    per_task_mean_f1 = np.asarray([x.mean(axis=axis) for x in arrs])
    overall_mean = per_task_mean_f1.mean(axis=0)
    return overall_mean


def bootstrap_macro_avg(data: list[np.array], alpha: float):
    ci_res = bootstrap(
        data,
        macro_avg,
        vectorized=True,
        confidence_level=1 - alpha,
        random_state=42,
        n_resamples=10_000,
    )
    f1_value = macro_avg(*data)
    return ci_res, f1_value


def compute_overall_scores_with_ci(data: list[dict], alpha=0.05):
    model2scores = defaultdict(list)
    # parse the scores into normalized format
    for row in data:
        model = row["model_codename"]
        dataset = row["dataset_name"]
        if row["error"]:
            print(f"Skipping {model} {dataset} because of error")
            continue

        # not participating in the benchmark overall average
        if dataset == ReverseInstructionsDataset.name:
            continue

        metrics, model = parse_row(row)
        model2scores[model].append(metrics)

    model2ci = {}

    # verify which models have full datasets and omit the ones who don't
    for model, model_scores in tqdm(list(model2scores.items())):
        dataset_names = {s["dataset"] for s in model_scores}
        assert len(dataset_names) == len(model_scores), f"non unique datasets in model {model}"

        if missing_datasets := (set(EXPECTED_DATASETS) ^ dataset_names):
            if model.endswith("-sd") and missing_datasets <= {
                AspectNewsDataset.name,
                DebateSumDataset.name,
            }:
                md_completed_scores = [
                    s
                    for s in model2scores[model.replace("-sd", "-md")]
                    if s["dataset"] in missing_datasets
                ]
                if len(md_completed_scores) == len(missing_datasets):
                    print(f"completed {model} with MD dataset")
                    model_scores += md_completed_scores
                else:
                    print(f"WARN: {model} has no MD or SD datasets results")
                    continue
            else:
                print(
                    f"Skipping {model} because of missing/extra datasets: {','.join(missing_datasets)}"
                )
                continue

        assert len(model_scores) == 6
        model_scores.sort(key=lambda x: EXPECTED_DATASETS.index(x["dataset"]))

        # compute overall per dataset confidence interval (\alpha=0.05)
        datasets_cis = {}
        for dataset_scores in model_scores:
            ci_res, f1_value = bootstrap_macro_avg(
                [np.array(dataset_scores["all_scores"])], alpha=alpha
            )

            # Sanity that we compute average correctly
            assert round(f1_value * 100, 2) == round(dataset_scores["ext_f1"], 2)
            datasets_cis[dataset_scores["dataset"]] = {
                "f1_value": f1_value,
                "ext_f1_ci_low": ci_res.confidence_interval.low,
                "ext_f1_ci_high": ci_res.confidence_interval.high,
                "ext_f1_se": ci_res.standard_error,
                "ext_f1_ci": max(
                    f1_value - ci_res.confidence_interval.low,
                    ci_res.confidence_interval.high - f1_value,
                ),
            }

        all_ds_scores = [np.array(dataset_scores["all_scores"]) for dataset_scores in model_scores]
        ci_res, f1_value = bootstrap_macro_avg(all_ds_scores, alpha=alpha)

        # sanity check that macro average is correct
        f1_val_rounded = round(f1_value * 100, 1)
        f1_mean_rounded = round(np.mean([s["ext_f1"] for s in model_scores]), 1)
        try:
            assert (
                f1_val_rounded == f1_mean_rounded
            ), f"scores differ: {f1_val_rounded} != {f1_mean_rounded}"
        except AssertionError as exc:
            print(f"WARN: {model} Overall F1 scores not the same: {exc}")

        datasets_cis["Overall"] = {
            "ext-f1": f1_value * 100,
            "ext-precision": float(np.mean([m["ext_precision"] for m in model_scores])),
            "ext-recall": float(np.mean([m["ext_recall"] for m in model_scores])),
            "ext-f1-ci": max(
                f1_value - ci_res.confidence_interval.low,
                ci_res.confidence_interval.high - f1_value,
            )
            * 100,
            "f1": np.mean([m["orig_score"] for m in model_scores]),
            "precision": np.mean([m["orig_precision"] for m in model_scores]),
            "recall": np.mean([m["orig_recall"] for m in model_scores]),
            # extra values for debugging
            "ext_f1_ci_low": ci_res.confidence_interval.low,
            "ext_f1_ci_high": ci_res.confidence_interval.high,
            "ext_f1_se": ci_res.standard_error,
            "avg_hal": np.mean([m["hal"] for m in model_scores]),
            "avg_err": np.mean([m["err"] for m in model_scores]),
        }

        model2ci[model] = datasets_cis

    return model2ci


def compute_ci_loo():
    with open(configs.ROOT_DIR / "agg_results" / "models_results_concat.json") as fp:
        data = json.load(fp)

    ds2model = {
        "AspectNews": "LOO-EXTABS-fuzz-md",
        "DebateSum": "LOO-ARG-fuzz-md",
        "SciFact": "LOO-EVR-fuzz-sd",
        "OpenAsp": "LOO-ASPSEL-fuzz-sd",
        "SaliencyDetection": "SCOnly-fuzz-sd",
        "EvidenceDetection": "SCOnly-fuzz-sd",
    }
    rows = [
        r
        for r in data
        if any(m == r["model_codename"] and ds == r["dataset_name"] for ds, m in ds2model.items())
    ]

    for row in rows:
        if row["model_codename"] == "LOO-EVR-fuzz-sd":
            # row['ext-precision'] = -1000
            # row['ext-recall'] = -1000
            # row['ext-f1'] = -1000
            # row['precision'] = -1000
            # row['recall'] = -1000
            # row['f1'] = -1000
            row["orig_precision"] = -1000
            row["orig_recall"] = -1000
            row["orig_f1"] = -1000

    assert len(rows) == 6
    for row in rows:
        row["model_codename"] = "LOO-fuzz-sd"

    print(compute_overall_scores_with_ci(rows))


def main():
    with open(configs.ROOT_DIR / "agg_results" / "models_results_concat.json") as fp:
        data = json.load(fp)
    model2ci = compute_overall_scores_with_ci(data, alpha=0.05)
    with open(configs.ROOT_DIR / "agg_results" / "confidence_intervals.json", "w") as fp:
        json.dump(model2ci, fp, indent=2)


if __name__ == "__main__":
    main()
