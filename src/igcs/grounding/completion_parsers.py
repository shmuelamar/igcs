"""Set of utilities for parsing outputs of LLMs into structured data."""

import ast
import json
import logging
import re

logger = logging.getLogger(__name__)

PREAMBLES = ("assistant\n\nSelected Content:", "Selected Content:")
EXPECTED_CHARACTERS_BY_PRESTRING_VALUE = {
    "[": (",", "]"),
    "]": ("[", ","),
    "{": (":",),
    "}": (",", "{", "]"),
    ":": (",", "}"),
}


def parse_selection(selection: str) -> list[str] | None:
    """Parses text as JSON array of string by attempting to remove prefix text,
    crop json inside ``` code snippet and fix unescaped quotes.
    """
    selection = remove_preamble(selection)
    selection = crop_text(selection).strip()

    try:
        return parse_json(selection)
    except (
        ValueError,
        KeyError,
        AttributeError,
        SyntaxError,
        TypeError,
        MemoryError,
        RecursionError,
    ):
        return None


def remove_preamble(selection: str):
    for preamble in PREAMBLES:
        if selection.startswith(preamble):
            selection = selection[len(preamble) :].strip()
    return selection


def crop_text(text: str) -> str:
    """Crops text if it is encapsulated inside a code snippet"""

    preamble = re.compile(r"^\s*```(\w+)?\s", flags=re.M)
    text = text.strip()
    cnt = len(preamble.findall(text))
    if (match := preamble.search(text)) and text.endswith("```") and cnt == 1:
        return text[match.end() : -len("```")]

    return text


def parse_json(text: str) -> dict | list:
    """parses json output, attempting to fix unescaped quotes"""

    try:
        return json.loads(text, strict=False)
    except ValueError:
        pass

    for fixer_fn in [
        fix_unescaped_quotes,
        fix_curly_braces,
        lambda t: fix_unescaped_quotes(fix_curly_braces(t)),
    ]:
        try:
            fixed_text = fixer_fn(text)
        except (KeyError, AttributeError):
            fixed_text = text
        else:
            try:
                return json.loads(fixed_text, strict=False)
            except ValueError:
                continue

    return ast.literal_eval(fixed_text)


def fix_curly_braces(text: str) -> str:
    fixed_text = re.sub(r"^(\s*)\{", r"\1[", text, flags=re.MULTILINE)
    fixed_text = re.sub(r'^(\s*)"instruction": ?', r"\1", fixed_text, flags=re.MULTILINE)
    return re.sub(r"}(,?)$", r"]\1", fixed_text, flags=re.MULTILINE)


def fix_unescaped_quotes(raw: str) -> str:
    """Attempts to fix badly escape json quotes"""
    # adapted from https://stackoverflow.com/a/75526674/7438048
    in_string = False
    output = []
    nesting_stack = []
    for index, character in enumerate(raw):
        if character == '"' and raw[index - 1] != "\\":
            if in_string:
                first_nonwhite_character_ahead = re.search(r"\S", raw[index + 1 :]).group()
                if (
                    nesting_stack
                    and first_nonwhite_character_ahead
                    in EXPECTED_CHARACTERS_BY_PRESTRING_VALUE.get(nesting_stack[-1], ())
                ):  # (",", "]", "}", ":"):
                    in_string = False
                else:
                    output.append("\\")
            else:
                in_string = True
        else:
            if not in_string:
                if character.strip() != "" and character not in (",",):
                    nesting_stack.append(character)
        output.append(character)
    return "".join(output)
