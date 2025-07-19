"""This file used for generating a 1-step pipeline in Section 6.2 and Table 5 in the paper."""

import ast
import concurrent
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import igcs.datasets.create_gencs_dataset.response_parser
from igcs import predict
from igcs.configs import ANNOTATED_RI_DIR, RAW_TRAIN_DATASETS_DIR
from igcs.datasets.create_gencs_dataset.annotate_dataset import (
    SYSTEM_ROLE_MANAGER,
    format_annotation_guidelines,
    format_doc,
    ground_selections,
    load_df,
    make_selection_samples,
    parse_generated_instructions,
    serialize_df,
)
from igcs.entities import Doc, Selection
from igcs.predict import LLM_MODEL
from igcs.utils import log

logger = logging.getLogger("igcs.create_gencs_dataset.annotate_dataset_single_step")

ANNOTATION_INSTRUCTIONS_SINGLE_STEP = """
You are a manager of a {software|publishing} firm and you are required to train the best students on how to perform {code|content} selection from given sources.
Write %(num_instructions)s short instructions for selecting {code|content} from the given {file|document}(s) and select {code|content} from the {source file|document}(s) that matches the instruction, to challenge students and train them on how to select relevant {code|content} based on diverse instructions.

Guidelines:
  1. You must keep the %(num_instructions)s instructions short and concise - as a single sentence.
  2. Instructions must start with the words "Select {code|content}" as they are always for selecting {code|content} from the {file|document}(s) and never for writing a new {code|text} nor paraphrasing the original content.
  3. Instructions should not be too specific that hint on the answer and not too vague that cannot be fulfilled.
  4. Output the exact verbatim {code|text phrases} from the {source file|document}(s). Do not change spaces or punctuation, do not fix typos and avoid any other changes to the {code|content} you select.
  5. Follow the instructions as closely as possible and pay careful attention to them.
  6. Output format must be a two level nested list, the first level is the instruction and the second level is the multiple {code|content} selections copied from the original {source file|document}(s).

The {source-code|document}(s):
""".strip()

ANNOTATION_NEG_INSTRUCTIONS_SINGLE_STEP = """
You are a manager of a {software|publishing} firm and you are required to train the best students on how to perform {code|content} selection from given sources.
Write %(num_instructions)s short instructions that have no matching {code|content} from the given {file|document}(s) to challenge students and train them to avoid selecting {code|content} when none matches the instruction.

Guidelines:
  1. You must keep the %(num_instructions)s instructions short and concise - as a single sentence.
  2. Instructions must start with the words "Select {code|content}" as they are always for selecting {code|content} from the {file|document}(s) and never for writing a new {code|text} nor paraphrasing the original content.
  3. Instructions should not be too specific that hint on the answer and not too vague that cannot be fulfilled.
  4. Write the instructions as a numbered list.

The {source-code|document}(s):
""".strip()


def sample2tagging_prompt_single_step(
    is_code: bool, is_negative: bool, docs: list[Doc], num_instructions: int
) -> tuple[str, str]:
    # create the prompts
    if is_negative:
        guidelines_template = ANNOTATION_NEG_INSTRUCTIONS_SINGLE_STEP
    else:
        guidelines_template = ANNOTATION_INSTRUCTIONS_SINGLE_STEP

    prompt = [
        format_annotation_guidelines(guidelines_template, is_code)
        % {"num_instructions": num_instructions}
    ]
    prompt += format_doc(docs, is_code)
    system_role = format_annotation_guidelines(SYSTEM_ROLE_MANAGER, is_code)
    return "".join(prompt), system_role


def extra_parser(
    raw_instructions_and_selection: str, num_instructions: int
) -> dict[str, list[str]]:
    try:
        resp = json.loads(raw_instructions_and_selection, strict=False)
    except (ValueError, TypeError, AttributeError):
        try:
            resp = ast.literal_eval(raw_instructions_and_selection)
        except (ValueError, TypeError, AttributeError, SyntaxError):
            raise ValueError(f"Invalid: {raw_instructions_and_selection}")

    if (
        len(resp) == 2
        and isinstance(resp, list)
        and len(resp[0]) == len(resp[1]) == num_instructions
    ):
        instructions, selections = resp[0], resp[1]
        return dict(zip(instructions, selections))

    # single instruction special case
    if (
        num_instructions == 1
        and len(resp) == 2
        and isinstance(resp, list)
        and len(resp[0]) == num_instructions
    ):
        instruction, selections = resp[0][0], resp[1]
        assert isinstance(instruction, str)
        if selections and isinstance(selections[0], list):
            selections = [si for s in selections for si in s]
        assert isinstance(selections, list) and all(isinstance(s, str) for s in selections)
        return {instruction: selections}

    raise ValueError(f"Unknown format: {raw_instructions_and_selection}")


def single_step_generate_instruction(
    sample: dict, model: LLM_MODEL, num_instructions: int
) -> tuple[dict[str, list[Selection]] | None, dict]:
    logger.info(
        f'generating {num_instructions} instruction(s) and selection(s) for {sample["id"]} with {model}'
    )
    uid = sample["id"]
    is_code = bool(sample["type"] == "code")
    is_negative = bool(sample["is_negative"])
    docs = sample["documents"]

    # create prompt chat
    prompt, system = sample2tagging_prompt_single_step(is_code, is_negative, docs, num_instructions)

    # model answer
    raw_instructions_and_selection = predict.completion(prompt=prompt, system=system, model=model)

    # parse answer
    is_error = False
    instruction2raw_selections = None

    if not is_negative:
        try:
            instruction2raw_selections = (
                igcs.datasets.create_gencs_dataset.response_parser.parse_multi_selections(
                    raw_instructions_and_selection,
                    num_instructions,
                )
            )
            assert instruction2raw_selections is not None
        except (ValueError, TypeError, AttributeError, AssertionError):
            try:
                instruction2raw_selections = extra_parser(
                    raw_instructions_and_selection, num_instructions
                )
            except ValueError:
                is_error = True
    else:
        instructions = parse_generated_instructions(raw_instructions_and_selection, uid)
        assert all(isinstance(i, str) for i in instructions)
        instruction2raw_selections = {i: [] for i in instructions}

    # detailed debug info
    debug_info = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "completion": raw_instructions_and_selection,
        "instruction2raw_selections": deepcopy(instruction2raw_selections),
        "is_error": is_error,
    }

    if instruction2raw_selections and not len(instruction2raw_selections) == num_instructions:
        raise ValueError(
            f"{uid} - got response with wrong number of instructions - {instruction2raw_selections!r}"
        )

    if instruction2raw_selections is not None:
        # ground selections to the documents
        instruction2selections = ground_selections(instruction2raw_selections, docs)
    else:
        instruction2selections = {}

    return instruction2selections, debug_info


def main(
    input_file: str | Path,
    output_fname: str | Path,
    num_instructions: int,
    n_jobs: int = 1,
    debug: bool = True,
):
    input_file = Path(input_file)
    # input_file.parent.mkdir(parents=True, exist_ok=True)

    # step1 - load raw, un-annotated data
    df = load_df(input_file)

    model: LLM_MODEL = "GPT4"

    all_results = {}
    all_instructions = {}
    all_debug_infos = {}

    # Single Step
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        future2sample_id = {
            pool.submit(
                single_step_generate_instruction,
                sample=sample,
                model=model,
                num_instructions=num_instructions,
            ): sample_id
            for sample_id, (_, sample) in enumerate(df.iterrows())
        }
        for future in concurrent.futures.as_completed(future2sample_id):
            sample_id = future2sample_id[future]
            instruction2selections, debug_info = future.result()

            if debug:
                all_debug_infos[sample_id] = debug_info

            if debug_info["is_error"]:
                logger.info(f"error in selection {sample_id}: {debug_info['completion']!r}")
                all_results[sample_id] = None
                all_instructions[sample_id] = None
                continue

            # if instruction2selections is None:
            #     logger.info(f"no instructions for sample {sample_id}")
            #     all_instructions[sample_id] = None
            # else:
            all_results[sample_id] = {model: instruction2selections}
            all_instructions[sample_id] = list(instruction2selections.keys())

            logger.info(f"finished {sample_id}")

    df["selection_results"] = [x for _, x in sorted(all_results.items(), key=lambda x: x[0])]
    df["generated_instructions"] = [
        x for _, x in sorted(all_instructions.items(), key=lambda x: x[0])
    ]

    # step 4 - output debug information
    if debug:
        df["debug_info"] = [x for _, x in sorted(all_debug_infos.items(), key=lambda x: x[0])]

    df["samples"] = df.apply(make_selection_samples, axis=1)
    total_rows = len(df)
    df = df[df["selection_results"].apply(bool)]

    # serialize and save
    logger.info(f"saving {len(df)}/{total_rows} records")
    df.sort_values(["id"], inplace=True)
    serialize_df(df)
    df.to_json(output_fname, lines=True, orient="records")
    logger.info(f"saved annotated data to {output_fname}")


if __name__ == "__main__":  # pragma: nocover
    log.init()
    num_instructions = 1
    main(
        input_file=RAW_TRAIN_DATASETS_DIR / "sampled_ri_dataset_for_annotation_n500.jsonl.gz",
        output_fname=ANNOTATED_RI_DIR
        / f"annotated_ri_samples_single_step_n500_k{num_instructions}.jsonl.gz",
        num_instructions=num_instructions,
        n_jobs=1,
    )
