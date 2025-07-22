import functools
import gc
import json
import logging
import os
from pathlib import Path

import gradio as gr
import tiktoken
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from igcs import grounding
from igcs.entities import Doc, Selection
from igcs.utils import log
from igcs.utils.diskcache import disk_cache

try:
    import spaces
except ModuleNotFoundError:
    spaces = lambda: None
    spaces.GPU = lambda fn: fn

logger = logging.getLogger("igcs-demo")

_EXAMPLES_DIR = Path(__file__).parent

# In this simulation, we store only a single document although multi-document is possible.
# taken from https://en.wikipedia.org/wiki/Barack_Obama
with open(_EXAMPLES_DIR / "barack_obama_wiki.txt", encoding="utf8") as fp:
    DEFAULT_TEXT = fp.read().strip()

# This is the global doc in this demo
DEFAULT_PROMPTS = (
    "Select content that details Obama's initiatives",
    "Select content that discusses Obama's personal life",
    "Select content that details Obama's education",
    "Select content with Obama's financial data",
)

# see src/igcs/prompting.py for more info
PROMPT_TEMPLATE = (
    "Given the following document(s), {selection_instruction}. "
    "Output the exact text phrases from the given document(s) as a valid json array of strings. Do not change the copied text.\n\n"
    "Document #0:\n{doc.text}\n"
)

MODELS_LIST = [
    # local models:
    ("====== IGCS Fine-tuned SLMs ======", None),
    ("Qwen2.5-3b-GenCS-union (local)", "shmuelamar/Qwen2.5-3b-GenCS-union"),
    ("====== Zero-shot SLMs ======", None),
    ("Qwen/Qwen2.5-3B-Instruct (local)", "Qwen/Qwen2.5-3B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct (local)", "Qwen/Qwen2.5-7B-Instruct"),
    # API-based OpenRouter models:
    ("====== API-based Models ======", None),
    ("qwen/qwen3-14b (API)", "api:qwen/qwen3-14b:free"),
    ("moonshotai/kimi-k2 (API)", "api:moonshotai/kimi-k2:free"),
    ("deepseek/deepseek-chat-v3-0324 (API)", "api:deepseek/deepseek-chat-v3-0324:free"),
    ("meta-llama/llama-3.3-70b-instruct (API)", "api:meta-llama/llama-3.3-70b-instruct:free"),
    ("meta-llama/llama-3.1-405b-instruct (API)", "api:meta-llama/llama-3.1-405b-instruct:free"),
]
DEFAULT_MODEL = MODELS_LIST[1][1]
MAX_INPUT_TOKENS = 4500
MAX_PROMPT_TOKENS = 256


@spaces.GPU
def completion(prompt: str, model, tokenizer):
    # tokenize
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # MPS (on Mac) requires manual attention mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)

    logger.info(f"generating completion with model_id: {model.name_or_path} and prompt: {prompt!r}")
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        # eos_token_id=[tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]],
        do_sample=False,
        top_k=None,
        top_p=None,
        temperature=None,
    )
    return tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True)


def completion_openrouter(prompt: str, model_id: str):
    logger.info(f"calling openrouter with model_id: {model_id} and prompt: {prompt!r}")
    client = load_openrouter_client()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content


# @functools.cache
def load_model_and_tokenizer(model_id: str):
    logger.info(f"loading local model and tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    logger.info(f"done loading {model_id}")
    return model, tokenizer


def load_openrouter_client():
    logger.info(f"connecting to OpenRouter")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )


@disk_cache(cache_dir=_EXAMPLES_DIR / "models-cache")
def get_completion_cache(*, prompt: str, model_id: str) -> str:
    return get_completion(prompt=prompt, model_id=model_id)


@functools.lru_cache(maxsize=2048)
def get_completion(*, prompt: str, model_id: str):
    if model_id.startswith("api:"):
        return completion_openrouter(prompt, model_id.removeprefix("api:"))
    else:
        model, tokenizer = load_model_and_tokenizer(model_id)
        resp = completion(prompt, model, tokenizer)

        # cleanup memory
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        return resp


TIKTOKEN_TOKENIZER = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    return len(TIKTOKEN_TOKENIZER.encode(text))


def perform_igcs(
    doc: Doc, selection_instruction: str, model_id: str
) -> tuple[list[Selection] | None, str]:
    logger.info(f"performing selection with {selection_instruction!r} using {model_id!r}")
    prompt = PROMPT_TEMPLATE.format(doc=doc, selection_instruction=selection_instruction)

    # For the example inputs - we cache from disk as they are more popular
    if doc.text == DEFAULT_TEXT and selection_instruction in DEFAULT_PROMPTS:
        logger.info("using disk_cache mode")
        resp = get_completion_cache(prompt=prompt, model_id=model_id)
    else:
        resp = get_completion(prompt=prompt, model_id=model_id)
    logger.info(f"Got response from model: {model_id}: {resp!r}")

    # First, parse the selections as json array of strings
    selection_spans = grounding.parse_selection(resp)

    # Next, ground them to specific character positions in the source documents
    selections = grounding.ground_selections(selection_spans, docs=[doc])
    logger.info(f"model selections: {selections!r}")
    return selections, resp


def convert_selections_to_gradio_highlights(selections, doc) -> list[tuple[str, str | None]]:
    pos = 0
    highlights = []

    # add hallucinations outside the text itself:
    if any(sel.doc_id == -1 for sel in selections):
        highlights.append(
            ("\n\nHallucinated selections (not found in the document):\n\n", "hallucination")
        )
        for sel in selections:
            if sel.doc_id != -1:  # not hallucination
                continue

            highlights.append((sel.content + "\n", "hallucination"))

    selections.sort(key=lambda sel: (sel.end_pos, sel.start_pos))
    for sel in selections:
        if sel.doc_id == -1:
            continue  # hallucination
        if pos < sel.start_pos:
            highlights.append((doc.text[pos : sel.start_pos], None))  # outside selection
        elif pos >= sel.end_pos:
            continue  # two selections overlap - we only display the first.

        highlights.append(
            (doc.text[sel.start_pos : sel.end_pos], sel.metadata["mode"])
        )  # the selection
        pos = sel.end_pos

    if pos + 1 < len(doc.text):
        highlights.append((doc.text[pos:], None))  # end of the text

    return highlights


def process_igcs_request(selection_instruction: str, model_id: str, doc_data: list[dict]):
    if model_id is None:
        raise gr.Error("Please select a valid model from the list.")

    doc_text = "".join(
        [doc["token"] for doc in doc_data if doc["class_or_confidence"] != "hallucination"]
    )

    if count_tokens(doc_text) > MAX_INPUT_TOKENS:
        raise gr.Error(
            f"File too large! currently only up-to {MAX_INPUT_TOKENS} tokens are supported"
        )

    if count_tokens(selection_instruction) > MAX_PROMPT_TOKENS:
        raise gr.Error(f"Prompt is too long! only supports up-to {MAX_PROMPT_TOKENS} tokens.")

    # Perform content selection
    # TODO: cache examples
    doc = Doc(id=0, text=doc_text)
    selections, model_resp = perform_igcs(doc, selection_instruction, model_id)

    if selections is None:
        raise gr.Error(
            "Cannot parse selections, model response is invalid. please try another instruction or model."
        )

    # Post-process selections for display as highlighted spans
    highlights = convert_selections_to_gradio_highlights(selections, doc)
    selections_text = json.dumps([s.model_dump(mode="json") for s in selections], indent=2)
    return highlights, model_resp, selections_text


def get_app() -> gr.Interface:
    with gr.Blocks(title="Instruction-guided content selection", theme="ocean", head="") as app:
        with gr.Row():
            gr.Markdown(
                "Instruction-guided content selection Demo. Check the [GitHub repo](https://github.com/shmuelamar/igcs) and our [paper]() for more info."
            )
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=300):
                prompt_text = gr.Dropdown(
                    label="Content Selection Instruction:",
                    info='Choose an existing instruction or write a short one, starting with "Select content" or "Select code".',
                    value=DEFAULT_PROMPTS[0],
                    choices=DEFAULT_PROMPTS,
                    multiselect=False,
                    allow_custom_value=True,
                )
            with gr.Column(scale=1, min_width=200):
                model_selector = gr.Dropdown(
                    label="Choose a Model",
                    info="Choose a model from the predefined list below.",
                    value=DEFAULT_MODEL,
                    choices=MODELS_LIST,
                    multiselect=False,
                    allow_custom_value=False,
                )

        with gr.Row():
            submit_button = gr.Button("Submit", variant="primary")
            upload_button = gr.UploadButton("Upload a text or code file", file_count="single")
            reset_button = gr.Button("Default text")

        with gr.Row():
            with gr.Accordion("Detailed response", open=False):
                model_resp_text = gr.Code(
                    label="Model's raw response",
                    interactive=False,
                    value="No response yet",
                    lines=5,
                    language="json",
                )
                model_selections_text = gr.Code(
                    label="Grounded selections",
                    interactive=False,
                    value="No response yet",
                    lines=10,
                    language="json",
                )

        with gr.Row():
            highlighted_text = gr.HighlightedText(
                label="Selected Content",
                value=[(DEFAULT_TEXT, None), ("", "exact_match")],
                combine_adjacent=False,
                show_legend=True,
                interactive=False,
                color_map={
                    "exact_match": "lightgreen",
                    "normalized_match": "green",
                    "fuzzy_match": "yellow",
                    "hallucination": "red",
                },
            )

        def upload_file(filepath):
            with open(filepath, "r", encoding="utf8") as fp:
                text = fp.read().strip()

            if count_tokens(text) > MAX_INPUT_TOKENS:
                raise gr.Error(
                    f"File too large! currently only up-to {MAX_INPUT_TOKENS} tokens are supported"
                )

            return [(text, None), ("", "exact_match")]

        def reset_text(*args):
            return [(DEFAULT_TEXT, None), ("", "exact_match")]

        upload_button.upload(upload_file, upload_button, outputs=[highlighted_text])
        submit_button.click(
            process_igcs_request,
            inputs=[prompt_text, model_selector, highlighted_text],
            outputs=[highlighted_text, model_resp_text, model_selections_text],
        )
        reset_button.click(reset_text, reset_button, outputs=[highlighted_text])

    return app


if __name__ == "__main__":
    log.init()
    logger.info("starting app")
    app = get_app()
    app.queue()
    app.launch()
    logger.info("done")
