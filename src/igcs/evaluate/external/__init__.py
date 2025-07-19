"""We build upon the original evaluation scripts of IGCS-Bench Datasets, copied with
minimal or no adaptation."""

import json
import tempfile

import numpy as np
import pandas as pd
from nltk import word_tokenize

from igcs.datasets import load_dataset
from igcs.datasets.igcsbench_datasets import (
    AspectNewsDataset,
    DebateSumDataset,
    OpenAspDataset,
    ReverseInstructionsDataset,
    SciFactDataset,
    SparkEvidenceDataset,
    SparkSaliencyDataset,
)
from igcs.entities import Doc, ModelPrediction
from igcs.evaluate.evaluation_functions import score_rouge, unigram_score
from igcs.evaluate.external import aspectnews_evaluate, scifact_rationale_selection
from igcs.evaluate.external.convert_to_scifact_format import convert_to_scifact_format
from igcs.evaluate.external.helper import (
    align_model_prediction,
    align_selections_to_sentences,
    selections_to_text,
)
from igcs.evaluate.external.openasp_eval import calculate_openasp_scores
from igcs.utils import ioutils


def eval_openasp_external(
    dataset: str, split_name: str, predictions: list[ModelPrediction]
) -> dict[str, float]:
    ref_samples = load_dataset(dataset, split_name)
    aligned_predictions = align_model_prediction(predictions, samples=ref_samples)
    return calculate_openasp_scores(ref_outputs=ref_samples, model_outputs=aligned_predictions)


def pred_to_sents(pred: ModelPrediction, docs: list[Doc]) -> list[tuple[int, int]]:
    sent_ids = []
    for sel in pred["grounded_selection"] or []:
        doc = docs[sel["doc_id"]]
        selected_sentences_ids = [
            sent_id
            for sent_id, (sent_start, sent_end) in enumerate(doc.get_sentences_boundaries())
            if sel["start_pos"] == sent_start and sel["end_pos"] == sent_end
        ]
        assert len(selected_sentences_ids) == 1
        sent_ids.append((doc.id, selected_sentences_ids[0]))
    return sent_ids


def eval_aspectnews_external(
    dataset: str, split_name: str, predictions: list[ModelPrediction]
) -> dict[str, float]:
    assert split_name == "test", "only test supported"
    ref_samples = load_dataset(dataset, split_name)
    id2sample = {s.id: s for s in ref_samples}
    assert len(id2sample) == len(ref_samples)

    aligned_predictions = align_model_prediction(predictions, samples=ref_samples)
    ds = AspectNewsDataset()

    ref_df = pd.concat([pd.read_csv(ds.dirname / name) for name in ds.splits_fnames["test"]])
    aspectnews_preds = []
    for pred in aligned_predictions:
        pred_docs = id2sample[pred["sample_id"]].docs
        assert len(pred_docs) == 1
        sent_ids = pred_to_sents(pred, pred_docs)
        assert all(doc_id == 0 for doc_id, _ in sent_ids)
        sent_ids = [sid for _, sid in sent_ids]
        doc_sents = pred_docs[0].sentences()
        # text = [doc_sents[sid].content for sid in sent_ids]
        doc_text = [s.content for s in doc_sents]
        aspectnews_preds.append({"ids": sent_ids, "text": doc_text})

    with (
        tempfile.NamedTemporaryFile(suffix=".csv") as ref_file,
        tempfile.NamedTemporaryFile(suffix=".json") as preds_file,
    ):
        ref_df.to_csv(ref_file.name, index=False)
        with open(preds_file.name, "w") as fp:
            json.dump(aspectnews_preds, fp)
        f1, precision, recall, spearman_corr, rouges, max_f1 = aspectnews_evaluate.evaluate_output(
            annotations_filename=ref_file.name,
            model_outputs_filename=preds_file.name,
        )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "max_f1": max_f1,
        **rouges,
    }


def eval_debatesum_external(
    dataset: str, split_name: str, predictions: list[ModelPrediction]
) -> dict[str, float]:
    ref_samples = load_dataset(dataset, split_name)

    scores = []
    id2sample = {s.id: s for s in ref_samples}
    assert len(id2sample) == len(ref_samples)
    for pred in predictions:
        target_text = selections_to_text(id2sample[pred["sample_id"]])
        prediction_text = selections_to_text(pred)
        scores.append(score_rouge(target_text, prediction_text, detailed=True))

    return {metric: 100 * np.mean([s[metric] for s in scores]) for metric in scores[0].keys()}


def eval_spark_external(
    dataset: str, split_name: str, predictions: list[ModelPrediction]
) -> dict[str, float]:
    ref_samples = load_dataset(dataset, split_name)

    scores = []
    id2sample = {s.id: s for s in ref_samples}
    assert len(id2sample) == len(ref_samples)
    for pred in predictions:
        target_tokens = word_tokenize(selections_to_text(id2sample[pred["sample_id"]]))
        prediction_tokens = word_tokenize(selections_to_text(pred))
        scores.append(unigram_score(target_tokens, prediction_tokens))

    return {metric: np.mean([s[metric] for s in scores]) * 100 for metric in scores[0].keys()}


def eval_scifact_external(
    dataset: str, split_name: str, predictions: list[ModelPrediction]
) -> dict[str, float]:
    # SciFact dev == our test as they did not release test gold references
    assert split_name == "test", f"not implemented for split_name={split_name}"

    ref_samples = load_dataset(dataset, split_name)
    aligned_predictions = align_model_prediction(predictions, samples=ref_samples)
    scifact_format_predictions = convert_to_scifact_format(
        ref_outputs=ref_samples, model_outputs=aligned_predictions
    )

    ds = SciFactDataset()

    with tempfile.NamedTemporaryFile(suffix=".jsonl") as rationale_selection_file:
        ioutils.jsonl_dump(scifact_format_predictions, rationale_selection_file.name)
        corpus_fname = str(ds.dirname / "corpus.jsonl")
        dataset_fname = ds.dirname / ds.splits_fnames["test"]

        scores = scifact_rationale_selection.eval_predictions(
            corpus=corpus_fname,
            dataset=dataset_fname,
            rationale_selection=rationale_selection_file.name,
        )
    return {metric: val * 100 for metric, val in scores.items()}


DATASET2EVAL_FN = {
    SciFactDataset.name: eval_scifact_external,
    OpenAspDataset.name: eval_openasp_external,
    AspectNewsDataset.name: eval_aspectnews_external,
    DebateSumDataset.name: eval_debatesum_external,
    SparkEvidenceDataset.name: eval_spark_external,
    SparkSaliencyDataset.name: eval_spark_external,
    ReverseInstructionsDataset.name: lambda *_, **__: {"f1": 0.0},
}


def eval_external(
    dataset: str, split_name: str, predictions: list[ModelPrediction]
) -> dict[str, float]:
    return DATASET2EVAL_FN[dataset](dataset, split_name, predictions)
