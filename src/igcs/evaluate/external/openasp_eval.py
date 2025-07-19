from igcs.entities import ModelPrediction, SelectionSample


def calculate_openasp_scores(
    ref_outputs: list[SelectionSample], model_outputs: list[ModelPrediction]
) -> dict[str, float]:
    num_correct, num_predicted, num_gold = 0, 0, 0
    for ref_output in ref_outputs:
        assert len(ref_output.selections) == 1, "either no selections found or more than one found"
        curr_model_output = [elem for elem in model_outputs if elem["sample_id"] == ref_output.id]
        assert len(curr_model_output) == 1, "either no sample_id found or more than one found"
        curr_model_output = curr_model_output[0]

        # make sure all ref and generated instances align with the reference chunks
        all_chunks = [
            (elem.id, chunk_id) for elem in ref_output.docs for chunk_id in elem.chunks_pos
        ]  # (doc_id, sentence_start_idx)
        all_chunks += [
            (elem.id, 0) for elem in ref_output.docs
        ]  # add also the starting sentence (idx=0)
        ref_selected_sentences = [
            (elem.doc_id, elem.start_pos) for elem in ref_output.selections[0].selections
        ]  # (doc_id, sentence_start_idx)
        if not curr_model_output["grounded_selection"]:  # no selections found, sometimes upon error
            model_selected_sentences = []
        else:
            model_selected_sentences = [
                (elem["doc_id"], elem["start_pos"])
                for elem in curr_model_output["grounded_selection"]
            ]  # (doc_id, sentence_start_idx)
        assert all(
            elem in all_chunks for elem in ref_selected_sentences
        ), "misalignment between selected reference sentences and all_chunks"
        assert all(
            elem in all_chunks for elem in model_selected_sentences
        ), "misalignment between selected model sentences and all_chunks"

        num_gold += len(set(ref_selected_sentences))
        num_predicted += len(set(model_selected_sentences))
        num_correct += len(set(ref_selected_sentences).intersection(set(model_selected_sentences)))

    # calculate recall, precision and F-1
    if num_gold == 0:
        raise ValueError()
    if num_predicted == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_correct / num_predicted
    # if num_gold == 0:
    #     recall = 1
    # else:
    recall = num_correct / num_gold
    F1 = 2 * ((precision * recall) / (precision + recall)) if precision > 0 or recall > 0 else 0.0
    return {"precision": 100 * precision, "recall": 100 * recall, "f1": 100 * F1}
