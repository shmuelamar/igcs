import re

import markdown_to_json

from igcs.grounding.completion_parsers import crop_text, parse_json


def json_to_multi_selection(
    json_data: list[list[str]] | list[tuple[str, list[str]]],
    num_instructions: int = 5,
) -> dict[str, list[str]] | None:
    """Converts raw json objects into the selection format"""
    if not isinstance(json_data, list):
        # cannot understand json schema
        return None

    # type1: list of [inst1, [sel1, sel2...], inst2, [sel1, sel2, ...]]
    if (
        len(json_data) % num_instructions == 0
        and all(isinstance(item, str) for item in json_data[::2])
        and all(isinstance(item, list) for item in json_data[1::2])
    ):
        selections = dict(zip(json_data[::2], json_data[1::2]))

    # all the rest of the types accept only list of 5 elements
    elif len(json_data) != num_instructions:
        return None

    # type2: list of [key, value] where value is a list of strings - convert to dict
    elif all(len(lst) == 2 for lst in json_data) and all(
        isinstance(lst, list) and isinstance(lst[1], list) for lst in json_data
    ):
        selections = dict(json_data)

    # type3: list of lists of selections, were the first is the instruction
    elif (
        all(isinstance(lst, list) for lst in json_data)
        and all(len(lst) >= 1 for lst in json_data)
        and all(
            isinstance(lst[0], str) and lst[0].strip("\"'").startswith("Select")
            for lst in json_data
        )
    ):
        selections = {lst[0]: lst[1:] for i, lst in enumerate(json_data)}

    # type4: list of dicts of format {content:str, instruction:str}
    elif (
        all(isinstance(x, dict) for x in json_data)
        and all("content" in x for x in json_data)
        and all("instruction" in x for x in json_data)
    ):
        selections = {dct["instruction"]: dct["content"] for dct in json_data}

    # type5: list of lists of selections omitting instructions (we number them between 0-4)
    else:
        selections = {i: lst for i, lst in enumerate(json_data)}

    return {str(inst): sels for inst, sels in selections.items()}


def parse_multi_selections(
    selection_text: str, num_instructions: int = 5
) -> dict[str, list[str]] | None:
    """Parses selections output into instruction -> list of selections"""

    if selection_text.startswith("<error "):
        raise ValueError(f"Selection Error - {selection_text}")

    # crop selection if inside
    selection_text = crop_text(selection_text)

    # type1: try parse as valid json
    try:
        json_data = parse_json(selection_text)
        return json_to_multi_selection(json_data, num_instructions)
    except (ValueError, SyntaxError, MemoryError, RecursionError):

        # type2: json newline list
        try:
            parsed_selections = parse_nested_lists_as_json(selection_text)
        except (ValueError, IndexError):

            # type3: nested Markdown list
            parsed_selections = parse_markdown_nested_list(selection_text)

    if len(parsed_selections) == 1 and len(list(parsed_selections.values())[0]) == num_instructions:
        parsed_selections = dict(list(parsed_selections.values())[0])
    elif len(parsed_selections) != num_instructions:
        return None

    for inst, selections in parsed_selections.items():
        if isinstance(selections, str):
            parsed_selections[inst] = [selections]

    parsed_selections = {
        str(inst): [clean_text(s) for s in selections if s is not None]
        for inst, selections in parsed_selections.items()
    }
    return parsed_selections


def parse_markdown_nested_list(text: str) -> dict[str, list[str]]:
    """Parsers Markdown list of lists. e.g.:

    Input example:

        1. Instruction A:
          - "selection 1"
          - "selection 2"
          - "selection 3"
        2. **Instruction B:
          - "selection 4"
          - "selection 5"
          - "selection 6"

    Output example:

        {"Instruction A": ["selection 1", "selection 2", "selection3"],
         "**Instruction B": ["selection 4", "selection5", "selection6"]}
    """
    # markdown_to_json works better with headers rather than numbered lists (i.e. `#` vs 1.)
    text = re.sub(r"^(- (\*\*)?)?\d+\.", r"#", text, flags=re.MULTILINE)

    # fix incorrect indentation of 2nd level lists
    text = re.sub(r"^ {4}([*-]) +", r"  \1 ", text, flags=re.MULTILINE)
    return dict(markdown_to_json.dictify(text))


def parse_nested_lists_as_json(selections: str) -> dict[str, list[str]]:
    """Parses possibly ill-delimited list of lists in json format"""
    data: list[tuple[str, list[str]]] = []

    # find max level
    level = 0
    max_level = 0
    for line in selections.strip().splitlines():
        line = line.strip()
        if line == "[":
            level += 1
            max_level = max(level, max_level)
        elif line == "]" or line == "],":
            level -= 1

    if max_level not in [2, 3]:
        raise ValueError(f"cannot parse selections level is {max_level}: {selections}")

    # nested mode
    if max_level == 3:
        level = 0
        for line in selections.strip().splitlines():
            line = line.strip()

            # control states
            if not line:
                continue
            elif line == "[":
                level += 1
                continue
            elif line == "]" or line == "],":
                level -= 1
                continue

            # content
            if level < max_level:
                level1_data = line.rstrip(",").strip('"')
                data.append((level1_data, []))
            elif level == max_level:
                level2_data = line.rstrip(",").strip('"')
                data[-1][1].append(level2_data)

    # max_level == 2
    else:
        level = 0
        start = True
        for line in selections.strip().splitlines():
            line = line.strip()

            # control states
            if not line:
                continue
            elif line == "[":
                level += 1
                continue
            elif line == "]" or line == "],":
                start = True
                level -= 1
                continue

            # content
            if start:
                level1_data = line.rstrip(",").strip('"')
                data.append((level1_data, []))
                start = False
            else:
                level2_data = line.rstrip(",").strip('"')
                data[-1][1].append(level2_data)

    return dict(data)


def clean_text(text: str | list[str]) -> str:
    """Strips whitespaces and remove enclosing quotes, colon or dot"""
    if isinstance(text, list):
        text = "\n".join(text)

    cleaned = text.strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    if cleaned.endswith(":") or cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()

    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()

    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1].strip()

    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    return cleaned
