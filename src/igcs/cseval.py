import argparse
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from igcs import grounding
from igcs.datasets import get_dataset_file
from igcs.datasets.igcsbench_datasets import ReverseInstructionsDataset
from igcs.entities import Doc, Selection, SelectionSample
from igcs.evaluate import evaluation_functions
from igcs.evaluate.external import eval_external
from igcs.grounding import completion_parsers, tokenization
from igcs.grounding.tokenization import spacy_tokenize
from igcs.utils import ioutils

logger = logging.getLogger("igcs.cseval")


def load_eval_data(ref_file, pred_file, allow_partial: bool):
    samples = [SelectionSample(**s) for s in ioutils.jsonl_load(ref_file)]
    ref_df = pd.DataFrame({"sample": samples})
    ref_df["sample_id"] = ref_df["sample"].apply(lambda s: str(s.id))

    pred_df = pd.read_json(pred_file, lines=True)
    # ensure all ids are strings and not casted to ints
    pred_df["sample_id"] = pred_df["sample_id"].apply(str)

    if not ref_df["sample_id"].is_unique or any(ref_df["sample_id"].isna()):
        raise ValueError(f"{ref_file} contains duplicate or empty sample_ids.")

    if not pred_df["sample_id"].is_unique or any(pred_df["sample_id"].isna()):
        raise ValueError(f"{pred_file} contains duplicate or empty sample_ids.")

    ref_ids = set(ref_df["sample_id"])
    pred_ids = set(pred_df["sample_id"])

    if pred_ids < ref_ids:
        if allow_partial:
            logger.warning(
                f"Partial predictions, missing {len(ref_ids - pred_ids)}/{len(ref_ids)}"
                f"sample_ids from the prediction file"
            )
        else:
            raise ValueError(
                "The prediction file must have exactly the same sample_ids as the reference file"
            )
    elif ref_ids != pred_ids:
        raise ValueError("Encountered prediction sample_ids not in the reference file")

    eval_df = pred_df.set_index("sample_id").join(ref_df.set_index("sample_id"))

    # paranoid check, shouldn't really happen.
    if not (set(eval_df.index) == ref_ids or allow_partial and set(eval_df.index) <= ref_ids):
        raise ValueError("Something bad happened")

    return eval_df


def parse_single_selections(
    selection: str | list[str] | None,
) -> list[str] | list[tuple[str, int]] | None:
    # we keep intact already parsed selection
    return parse_json_selections(selection)


def parse_selections(
    selection: str | list[str] | None, multi_selections: bool
) -> list[str | None] | list[tuple[str, int] | None] | list[Selection] | None:
    # return empty selection as is
    if selection is None or (isinstance(selection, list) and not selection):
        return selection

    # passthrough already grounded selection
    if (
        isinstance(selection, list)
        and isinstance(selection[0], dict)
        and {"doc_id", "start_pos", "end_pos", "content"} <= set(selection[0].keys())
    ):
        return [Selection(**s) for s in selection]

    if multi_selections:
        return parse_multi_selections(selection)
    return parse_single_selections(selection)


def parse_json_selections(selection: str) -> list[str] | None:
    parsed_selection = completion_parsers.parse_selection(selection)

    # try crop selection if prefixed by some text (e.g. "Here is the selection:")
    start = selection.find("[")
    end = selection.rfind("]")
    if parsed_selection is None and 0 <= start < end:
        parsed_selection = completion_parsers.parse_selection(selection[start : end + 1])

    # try to add "] to a string in case of premature ending of the json array
    if parsed_selection is None:
        if selection.strip().endswith('"'):
            suffix = '"]'
        else:
            suffix = "]"
        parsed_selection = completion_parsers.parse_selection(selection + suffix)

    if parsed_selection is None:
        logger.debug(f"Cannot parse selection: {selection!r}")
    return parsed_selection


def parse_multi_selections(
    selections: list[str],
) -> list[str | None] | list[tuple[str, int] | None] | None:
    all_selections = []
    for selection in selections:
        parsed_selection = parse_single_selections(selection)
        if parsed_selection and isinstance(parsed_selection, (list, tuple)):
            all_selections += [str(s) for s in parsed_selection]
        else:
            all_selections.append(None)
    return all_selections


def score_selection(
    ref_selections: list[Selection], pred_selections: list[Selection], docs: list[Doc]
):
    assert all(s.doc_id >= 0 for s in ref_selections), "hallucinations in ref selection"

    ref_tokens, ref_token_ids = tokenization.selections_to_token_ids(ref_selections, docs)
    if pred_selections is None:
        pred_selections = []
    pred_tokens, pred_token_ids = tokenization.selections_to_token_ids(pred_selections, docs)

    scores = evaluation_functions.unigram_score(ref_token_ids, pred_token_ids)
    scores["iou"] = evaluation_functions.iou_score(ref_token_ids, pred_token_ids)
    scores["ref_tokens"] = ref_tokens
    scores["ref_token_ids"] = ref_token_ids
    scores["pred_tokens"] = pred_tokens
    scores["pred_token_ids"] = pred_token_ids
    return scores


def score_selection_by_sample(
    pred_selections: list[Selection],
    sample: SelectionSample,
    max_by_key: Literal["iou", "f1"] = "f1",
):
    all_scores = []
    for selection_group in sample.selections:
        score = score_selection(
            ref_selections=selection_group.selections,
            pred_selections=pred_selections,
            docs=sample.docs,
        )
        score["ref_id"] = selection_group.id
        all_scores.append(score)

    best = max(all_scores, key=lambda s: s[max_by_key])
    return best


def save_eval_df(df: pd.DataFrame, outfile: str | Path):
    for_save = df.copy()
    for_save.reset_index(inplace=True)
    for_save.pop("sample")
    for_save["grounded_selection"] = for_save["grounded_selection"].apply(
        lambda s: [si.dict() for si in s] if s else s
    )
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    for_save.to_json(outfile, orient="records", lines=True)


def parse_args(argv: None | list[str] = None):
    file_format_help = (
        "Every line must be a json object with these keys - {keys}. "
        "`selection` may be a raw string that will be attempt to parse into a list of strings or an "
        "already parsed list of strings."
        "Other keys are permitted and will be copied to the output file."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--references",
        help=f"JSON-Lines file with reference data, optionally gzipped-compressed. {file_format_help.format(keys='`guid`, `documents` and `selection`')}",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-p",
        "--predictions",
        help=f"JSON-Lines file with prediction data, optionally gzipped-compressed. {file_format_help.format(keys='`guid` and `selection`')}",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output as JSON-Lines file were per sample results are saved.",
        required=False,
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--allow-partial",
        help="Allow partial evaluation of prediction file.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--ignore-parsing-errors",
        help="Ignore parsing errors and report metrics only on successfully parsed spans.",
        action="store_true",
        required=False,
        default=False,
    )
    return parser.parse_args(argv)


def run_cs_eval(
    references: Path | str,
    predictions: Path,
    output: Path | None = None,
    allow_partial: bool = False,
    quiet: bool = False,
    ignore_parsing_errors: bool = False,
    multi_selections: bool = False,
    fuzz: bool = True,
):
    references_file, dataset_name, split_name = get_dataset_file(references)

    # Step1: load files into guid and verify completeness
    eval_df = load_eval_data(references_file, predictions, allow_partial)

    # Step2: parse selections
    eval_df["parsed_selection"] = eval_df["selection"].apply(
        lambda s: parse_selections(s, multi_selections)
    )

    # Step3: ground selections
    eval_df["grounded_selection"] = eval_df.apply(
        lambda row: grounding.ground_selections(
            row["parsed_selection"],
            row["sample"].docs,
            modes=None if fuzz else ("exact_match", "normalized_match"),
        ),
        axis=1,
    )

    # Step4.1: score selections
    scores = eval_df.apply(
        lambda row: score_selection_by_sample(
            pred_selections=row["grounded_selection"],
            sample=row["sample"],
        ),
        axis=1,
    )

    # Step4.2: Eval on per dataset metric - if one of the reference is one of our datasets
    if dataset_name and dataset_name != ReverseInstructionsDataset.name:
        external_scores = eval_external(
            dataset_name,
            "test",
            predictions=eval_df.reset_index().to_dict(orient="records"),
        )
        main_external_score = (
            external_scores["f1"] if "f1" in external_scores else external_scores["rouge2"]
        )
    else:
        external_scores = None
        main_external_score = None

    scores = pd.json_normalize(scores).set_index(eval_df.index)
    eval_df = pd.concat((eval_df, scores), axis=1)

    # Step5: save
    if output:
        save_eval_df(eval_df, output)

    def _count_errors(lst):
        if not isinstance(lst, list):
            return 1.0
        if len(lst) == 0:
            return 0.0
        return sum(i is None for i in lst) / len(lst)

    total_samples = len(eval_df)
    parsing_errors = eval_df["parsed_selection"].apply(_count_errors).sum()
    if ignore_parsing_errors:
        logger.warning(f"ignoring parsing errors of {parsing_errors}/{total_samples} results")
        eval_df = eval_df[~eval_df["parsed_selection"].isna()]

    eval_results = {}
    raw_scores = {}
    for col in ["f1", "iou", "precision", "recall"]:
        metric_raw_scores = eval_df[col].fillna(0).copy()
        raw_scores[col] = metric_raw_scores.sort_index().tolist()
        mean, ci = evaluation_functions.mean_confidence_interval(metric_raw_scores, confidence=0.9)
        eval_results[col] = mean.round(4) * 100
        if col == "f1":
            eval_results["f1-ci"] = ci.round(4) * 100
            # eval_results["selection-token-size"] = (
            #     eval_df["grounded_selection"]
            #     .apply(
            #         lambda x: sum(
            #             [len(spacy_tokenize(s.content)) for s in (x or []) if s.doc_id >= 0]
            #         )
            #     )
            #     .mean()
            #     .round(1)
            # )

    all_selection_modes = (
        eval_df["grounded_selection"]
        .explode()
        .apply(lambda s: s.metadata["mode"] if isinstance(s, Selection) else None)
        .value_counts()
    )

    if all_selection_modes.sum() > 0:
        eval_results["hallucinations_rate"] = (
            all_selection_modes.get("hallucination", 0) / all_selection_modes.sum()
        )
    else:
        eval_results["hallucinations_rate"] = None

    if not quiet:
        print("Evaluation Results:")
        for k, v in eval_results.items():
            print(f"{k}: {v}")

        print("\n---")
        print(f"Parsing errors: {parsing_errors} - {100 * parsing_errors / total_samples:.2f}%")
        print(all_selection_modes)
        print(f"External main score - {main_external_score}.")
        print(f"detailed external metrics: {external_scores}")
    eval_results["parsing_errors"] = parsing_errors
    eval_results["parsing_errors_ratio"] = parsing_errors / len(eval_df)
    eval_results["parsing_modes"] = all_selection_modes.to_dict()
    return eval_df, eval_results, external_scores, main_external_score, raw_scores


def main(argv: None | list[str] = None):
    args = parse_args(argv)
    run_cs_eval(
        args.references,
        args.predictions,
        args.output,
        args.allow_partial,
        ignore_parsing_errors=args.ignore_parsing_errors,
    )


if __name__ == "__main__":  # pragma: nocover
    main()
