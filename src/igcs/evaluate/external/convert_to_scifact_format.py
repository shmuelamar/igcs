import argparse
import json
import logging
import os
from pathlib import Path

from igcs.entities import Doc, ModelPrediction, SelectionSample

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)


def convert_selection_to_evidence(curr_selection, docs_metadata: list[Doc]):
    relevant_doc_metadata = docs_metadata[curr_selection["doc_id"]]
    # add zero as the idx of the first chunk
    assert relevant_doc_metadata.chunks_pos[0] != 0

    selected_sentences_ids = [
        sent_id
        for sent_id, (sent_start, sent_end) in enumerate(
            relevant_doc_metadata.get_sentences_boundaries()
        )
        if curr_selection["start_pos"] == sent_start and curr_selection["end_pos"] == sent_end
    ]
    assert (
        len(selected_sentences_ids) == 1
    ), "every span must be aligned to sentence boundary and consist of single sentence"
    selected_sentences_ids = selected_sentences_ids[0]
    return relevant_doc_metadata.metadata["doc_id"], selected_sentences_ids


def convert_to_scifact_format(
    ref_outputs: list[SelectionSample], model_outputs: list[ModelPrediction]
):
    claims = []
    for ref_output in ref_outputs:
        curr_model_output = [elem for elem in model_outputs if elem["sample_id"] == ref_output.id]
        assert len(curr_model_output) == 1, "either no sample_id found or more than one found"
        curr_model_output = curr_model_output[0]
        evidence_dict = {}
        for sel in curr_model_output["grounded_selection"]:
            doc_id, sent_id = convert_selection_to_evidence(sel, ref_output.docs)
            evidence_dict.setdefault(doc_id, []).append(sent_id)

        curr_prediction_dict = {
            "claim_id": int(ref_output.id.split(".")[-1]),
            "evidence": {
                doc_id: sorted(set(sent_ids)) for doc_id, sent_ids in evidence_dict.items()
            },
        }
        claims.append(curr_prediction_dict)
    claims.sort(key=lambda x: x["claim_id"])
    return claims


def main(args):
    logging.info("Starting conversion...")
    ref_path = args.reference_path
    gen_path = args.prediction_path
    outdir = args.outdir
    with open(ref_path, "r") as f1:
        ref_outputs = [json.loads(line) for line in f1.readlines()]
    with open(gen_path, "r") as f1:
        model_outputs = [json.loads(line) for line in f1.readlines()]

    scifact_format_predictions = convert_to_scifact_format(ref_outputs, model_outputs)

    # create outdir if doesn't exist
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    outdir_SciFact = os.path.join(outdir, "converted_SciFact.jsonl")
    with open(outdir_SciFact, "w") as f1:
        for pred in scifact_format_predictions:
            # Convert the dictionary to a JSON string
            json_string = json.dumps(pred)
            # Write the JSON string to the file with a newline character
            f1.write(json_string + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument(
        "--reference-path",
        type=str,
        required=True,
        help="path to jsonl with reference answers",
    )
    argparser.add_argument(
        "--prediction-path",
        type=str,
        required=True,
        help="path to jsonl with predictions",
    )
    argparser.add_argument(
        "--outdir",
        type=str,
        default="converted_to_SciFact",
        help="path to outdir where the converted predictions will be saved.",
    )
    args = argparser.parse_args()
    main(args)
