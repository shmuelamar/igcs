import itertools
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import scipy.stats
from tqdm import tqdm

from igcs import configs
from igcs.evaluate.evaluation_functions import mean_confidence_interval
from igcs.evaluate.external import eval_external
from igcs.utils import log

logger = logging.getLogger(__name__)

RESULTS_FNAME = configs.ROOT_DIR / "agg_results" / "models_results_concat.json"


def load_models_data(models_codenames: list[str], key: str):
    with open(RESULTS_FNAME) as fp:
        data = json.load(fp)

    ds2models = defaultdict(list)
    for row in data:
        if row["model_codename"] in models_codenames:
            ds2models[row["dataset_name"]].append((row["model_codename"], row[key]))

    single_doc_dataset = ["DebateSum", "AspectNews"]
    models_md_codenames = [x.removesuffix("-sd") + "-md" for x in models_codenames]
    for row in data:
        row_model = row["model_codename"]
        if row["dataset_name"] in single_doc_dataset and row_model in models_md_codenames:
            ds2models[row["dataset_name"]].append((row["model_codename"], row[key]))
    return dict(ds2models)


def load_models_data_2(models_codenames: list[str]):
    with open(RESULTS_FNAME) as fp:
        data = json.load(fp)

    ds2models = defaultdict(dict)
    for row in data:
        if row["model_codename"] in models_codenames:
            ds2models[row["dataset_name"]][row["model_codename"]] = row

    single_doc_dataset = ["DebateSum", "AspectNews"]
    for row in data:
        row_model = row["model_codename"]
        row_model_sd = row_model.removesuffix("-md") + "-sd"
        dataset = row["dataset_name"]

        if ds2models[dataset].get(row_model_sd):
            continue

        if dataset in single_doc_dataset and row_model_sd in models_codenames:
            ds2models[dataset][row_model_sd] = row

    # filter empty datasets - such as RI
    return {k: v for k, v in ds2models.items() if v}


def main_wilcoxon_token_level_f1():
    models = [
        "RIOnly-Majority-fuzz-sd",
        "RIOnly-Union-fuzz-sd",
        "Llama-3-8B-fuzz-sd",
    ]
    ds2models = load_models_data(models, key="ext-f1-scores")

    for dataset_name, models_scores in ds2models.items():
        print()
        print(f"=======================")
        print(f"=== {dataset_name} ===")
        for (model_a, scores_a), (model_b, scores_b) in itertools.combinations(models_scores, 2):
            scores_a = np.asarray(scores_a)
            scores_b = np.asarray(scores_b)

            mean_a, ci_a = mean_confidence_interval(scores_a * 100.0, confidence=0.9)
            mean_b, ci_b = mean_confidence_interval(scores_b * 100.0, confidence=0.9)
            res = scipy.stats.wilcoxon(
                scores_a, scores_b, zero_method="pratt", alternative="greater"
            )
            if res.pvalue >= 0.01:
                prefix = "NOT SIGNIFICANT"
            else:
                prefix = "SIGNIFICANT"

            if mean_a - ci_a > mean_b + ci_b:
                prefix_abs = "SIG_ABS"
            else:
                prefix_abs = "NOT_SIG_ABS"
            print(
                f"--- {prefix_abs} {prefix} {model_a}(f1={mean_a:.2f}±{ci_a:.1f}) vs {model_b}(f1={mean_b:.2f}±{ci_b:.1f}) - {res.pvalue.round(7)} ---"
            )


def calc_statistic(a_data, b_data, dataset_name: str):
    if dataset_name == "DebateSum":
        primary_score = "rouge2"
    else:
        primary_score = "f1"

    score_a = eval_external(dataset_name, "test", list(a_data))
    score_b = eval_external(dataset_name, "test", list(b_data))

    diff = score_b[primary_score] - score_a[primary_score]
    return diff


def main_permutation(model_a_codename, model_b_codename, p_value_under=0.05, n_resamples=100):
    logger.info("loading data")
    ds2models = load_models_data_2([model_a_codename, model_b_codename])
    logger.info("done loading data")

    detailed_results = {}
    for dataset_name, models_preds in reversed(ds2models.items()):
        if dataset_name != "AspectNews":
            continue
        try:
            model_a_pred_data = models_preds[model_a_codename]["grounded_selections"]
            model_b_pred_data = models_preds[model_b_codename]["grounded_selections"]
        except KeyError:
            print(f"skipping not found dataset - {dataset_name}")
            continue

        if calc_statistic(model_a_pred_data, model_b_pred_data, dataset_name) <= 0:
            print("skipping significant test - model inferior to the baseline")
            continue

        logger.info(f"comparing {dataset_name}")
        res = scipy.stats.permutation_test(
            (model_a_pred_data, model_b_pred_data),
            vectorized=False,
            statistic=lambda a, b: calc_statistic(a, b, dataset_name=dataset_name),
            alternative="greater",
            permutation_type="samples",
            random_state=42,
            n_resamples=n_resamples,
        )

        # compute result
        pval = res.pvalue
        if pval < p_value_under:
            conclusion = f"significant (p={pval:.4f} < {p_value_under:.3f})"
        else:
            conclusion = f"insignificant (p={pval:.4f} > {p_value_under:.3f})"
        statistic = res.statistic
        print(
            f"comparing {model_a_codename} < {model_b_codename} on dataset {dataset_name}: {conclusion} (diff={statistic:.4f})"
        )

        detailed_results[dataset_name] = res
    logger.info(detailed_results)


def main(n_resamples=1000):
    futs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        baseline_model = "Llama-3-8B-ICL-fuzz-sd"
        for generic_tl_model in ["RIOnly-Union-fuzz-sd", "RIOnly-Majority-fuzz-sd"]:
            futs.append(
                executor.submit(
                    main_permutation, baseline_model, generic_tl_model, n_resamples=n_resamples
                )
            )

        for tl_model in [
            "LOO-EVR-fuzz-sd",
            "LOO-ASPSEL-fuzz-sd",
            "LOO-EXTABS-fuzz-md",
            "LOO-ARG-fuzz-md",
            "SCOnly-fuzz-sd",
        ]:
            futs.append(
                executor.submit(main_permutation, baseline_model, tl_model, n_resamples=n_resamples)
            )

        # supervised
        for sup_baseline_model in ["SFT-EVR-fuzz-sd", "SFT-EXTABS-fuzz-md", "SFT-ARG-fuzz-md"]:
            for sup_tl_model in [
                "SFT-RI-Union-EVR-fuzz-sd",
                "SFT-RI-Union-ARG-fuzz-md",
                "SFT-RI-Union-EXTABS-fuzz-md",
                "SFT-RI-Majority-EVR-fuzz-sd",
                "SFT-RI-Majority-ARG-fuzz-sd",
                "SFT-RI-Majority-EXTABS-fuzz-sd",
                "SCOnly-fuzz-sd",
            ]:
                futs.append(
                    executor.submit(
                        main_permutation, sup_baseline_model, sup_tl_model, n_resamples=n_resamples
                    )
                )

        for fut in tqdm(as_completed(futs), total=len(futs)):
            fut.result()


if __name__ == "__main__":
    log.init()
    main()
