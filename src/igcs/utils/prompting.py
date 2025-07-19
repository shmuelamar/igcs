import json
from typing import Literal

from _operator import attrgetter

from igcs.datasets.igcsbench_datasets import (
    AspectNewsDataset,
    DebateSumDataset,
    OpenAspDataset,
    ReverseInstructionsDataset,
    SciFactDataset,
    SparkEvidenceDataset,
    SparkSaliencyDataset,
)
from igcs.entities import (
    Selection,
    SelectionSample,
)

SYSTEM_ROLE = "You are a helpful assistant."


def get_output_format(
    selection_type: Literal["sentences", "text phrases"],
    source_type: Literal["documents", "document", "{doc_type}(s)", "abstract(s)"],
):
    # We name spans text phrases as spans are not that clear for LLMs from our tests.
    return (
        f"Output the exact {selection_type} from the given {source_type} as a valid json array of strings. "
        f"Do not change the copied text."
    )


INSTRUCTIONS = {
    OpenAspDataset.name: (
        'Given the following news articles on the topic "{title}", extract all sentences related to "{aspect_label}". '
        + get_output_format(selection_type="sentences", source_type="documents")
    ),
    AspectNewsDataset.name: (
        "Given the following news article, select at least 1 and at most 3 sentences that are the most relevant to the {topic}'s {aspect_description}. "
        + get_output_format(selection_type="sentences", source_type="document")
    ),
    SciFactDataset.name: (
        'Given the following abstract document(s) of medical papers, select the sentences that provide either supporting or refuting evidence for the claim: "{claim}". '
        + get_output_format(selection_type="sentences", source_type="abstract(s)")
    ),
    DebateSumDataset.name: (
        'Given the following document, select short and concise text phrases that summarize all the evidence for the argument: "{argument}". '
        + get_output_format(selection_type="text phrases", source_type="document")
    ),
    # Original prompt:  Below are documents on the same topic in different user messages. Please copy exactly salient sub-sentenial spans. Do not change the copied text.
    SparkSaliencyDataset.name: (
        "Given the following documents on the same topic, extract short and concise salient text phrases. "
        + get_output_format(selection_type="text phrases", source_type="documents")
    ),
    # Original prompt: Below are documents on the same topic and a query. Please extract exactly short text spans from the documents that match the information in the query. Separate the spans with a new line and start each span with -.
    SparkEvidenceDataset.name: (
        'Given the following documents on the same topic, extract short and concise text phrases that provide references to the following statement: "{query}". '
        + get_output_format(selection_type="text phrases", source_type="documents")
    ),
    ReverseInstructionsDataset.name: (
        "Given the following {doc_type}(s), {instruction}. "
        + get_output_format(selection_type="text phrases", source_type="{doc_type}(s)")
    ),
}

INSTRUCTIONS2 = {
    OpenAspDataset.name: (
        """Extract all sentences from the provided news articles that directly relate to the aspect specified by "{aspect_label}" while preserving the exact original wording.

Ensure you follow these details:
- Read each news article thoroughly for content related to "{aspect_label}".
- Identify and select sentences that explicitly mention or contextually relate to the specified aspect.
- **Do not rephrase, summarize, or modify any text.** Each extracted sentence must be copied exactly as it appears in the source.
- Begin by outlining your reasoning process—describe your criteria and steps for determining relevance—before presenting your final output. The reasoning steps must occur first, and the extracted sentences (final conclusions) should appear last.

# Steps
1. Analyze each provided news article sentence by sentence.
2. Determine which sentences are relevant to "{aspect_label}" based on explicit mentions or contextual clues.
3. Record your reasoning process briefly (i.e., criteria for selection and decision points) before compiling the final results.
4. Compile all relevant sentences into a list.

# Output Format
Provide your final answer as a valid JSON array of strings. For example:  
["First relevant sentence.", "Second relevant sentence.", "Third relevant sentence."]

# Notes
- The reasoning process must precede the final output and should clearly describe the selection criteria and process.
- Do not alter any of the extracted sentences in any way; maintain the original text in full.
- The final JSON array must be strictly valid without additional commentary or formatting outside of it."""
    ),
    AspectNewsDataset.name: (
        """Select between 1 and 3 sentences from the provided news article that are most relevant to the {topic}'s {aspect_description}.

Read the news article carefully and identify all sentences. Internally analyze each sentence for relevance to the specified topic and aspect. Perform your reasoning steps before arriving at a final conclusion, but only output the final result. Do not modify or alter any of the selected sentences.

# Steps
1. **Parse the Article:** Break the article into individual sentences.
2. **Internal Reasoning:** Evaluate each sentence for its relevance to the {topic}'s {aspect_description} based on the context and details provided.
3. **Selection:** Choose at least 1 and at most 3 sentences that best capture the required information.
4. **Conclusion:** Prepare your final selection after completing your internal reasoning.

# Output Format
Output a valid JSON array of strings containing the exact selected sentence(s). For example: ["Sentence 1", "Sentence 2"]

# Notes
- Do not change or rephrase any of the copied sentence(s).
- Ensure that your internal reasoning process is used to determine the selection, but do not include it in the final output."""
    ),
    SciFactDataset.name: (
        """Extract evidence sentences from provided medical abstracts for the claim.

Additional details:
- Input includes a claim in the form "{claim}" and one or more abstracts of medical papers.
- Your objective is to identify and extract any sentences that directly provide supporting or refuting evidence for the given claim.
- Do not modify, paraphrase, or change the exact text of any sentence extracted from the abstracts.
- Ensure that you examine each sentence methodically and conduct a reasoning process before finalizing the result. The reasoning steps must come before any conclusion, and the final output should present the selected sentences after all reasoning is complete.

Steps:
1. Parse the input to obtain the claim and the provided abstract document(s).
2. Split each abstract into individual sentences.
3. Evaluate each sentence to determine if it offers either supporting or refuting evidence for the claim.
4. Ensure all extracted sentences are unaltered in punctuation, casing, and formatting.
5. Reason through the evidence identification process before arriving at the conclusion.

Output Format:
- Provide your final result as a valid JSON array of strings.
- The JSON array must contain only the exact sentences that offer evidence for the claim.
- No extra commentary, reasoning details, or additional text should be included in the final output.

Notes:
- Ensure that the reasoning process is clearly organized with the reasoning steps listed before the final answer, but only the final JSON array should be output as the conclusion.
- Refrain from including any explanations or reasoning within the final output.
- Follow all instructions closely, ensuring minimal changes to the original text while preserving clarity and precision in the final result."""
    ),
    DebateSumDataset.name: (
        """Summarize Evidence for Argument

Task:  
Given a document and a specific argument represented by "{argument}", scan the document and identify short, concise text phrases that serve as evidence supporting that argument. Your task is to extract these evidence phrases exactly as they appear in the document without any modifications.

Additional Details:  
- Examine the entire document carefully to locate all relevant segments that directly support the argument.  
- Do not paraphrase or change any portion of the selected text.  
- Your response must include a reasoning section that details how you identified and selected the evidence phrases, followed by a conclusion section with the final output.

Steps:  
1. **Reasoning:**  
   - Read through the provided document carefully.  
   - Identify text segments that clearly support the argument "{argument}".  
   - Document your thought process detailing how you determined which phrases were relevant.  
   - **Note:** The reasoning section must come first in your response.

2. **Conclusion:**  
   - List the selected evidence phrases exactly as they appear in the document.  
   - Format your answer as a valid JSON array of strings.  
   - Ensure that the JSON output contains no additional text or modifications to the evidence.

Output Format:  
- The response must have two clearly separated sections:  
  - **Reasoning:** A detailed explanation of the extraction process.  
  - **Conclusion:** A JSON array (e.g., ["Phrase one", "Phrase two", "..."]) containing the exact evidence phrases from the document.

Notes:  
- The reasoning process must precede the final conclusion.  
- Do not alter, reword, or modify any of the text phrases extracted from the document.  
- Ensure that the final JSON output is syntactically valid and consists solely of the evidence phrases."""
    ),
    SparkSaliencyDataset.name: (
        """Extract short and concise salient text phrases from the provided documents.

Read the provided documents on the same topic and extract text segments that clearly represent the key points. Only extract text phrases that are exactly present in the original documents without any modifications. The phrases must be short, concise, and capture the essential information. Do not rephrase or alter the original text in any way.

**Reasoning Steps:**
1. Read and analyze each provided document carefully.
2. Identify text segments that are short, clear, and capture the most salient points.
3. Verify that each extracted phrase appears exactly as in the documents.
4. Perform all reasoning steps before concluding with the final output.

**Output Format:**
- The final answer must be a valid JSON array of strings.
- Each string in the array should be a salient text phrase extracted from the documents.

**Notes:**
- Follow the structured reasoning steps before listing the final result.
- Do not modify, rephrase, or change the original text of any extracted phrase.
- Reasoning must precede conclusions, and final output should be the extracted phrases only, formatted as specified."""
    ),
    SparkEvidenceDataset.name: (
        """Extract short and concise text phrases from provided documents that reference the query.

Additional details:
- You are given multiple documents on the same topic and a specific query placeholder "{query}".
- Your goal is to extract the exact text phrases from these documents that support or provide references to the query.
- Do not alter or modify any copied text; use the text exactly as it appears in the documents.

# Steps
1. Read and comprehend the documents and the query.
2. Identify text segments in the documents that reference or support the query "{query}".
3. Detail your reasoning steps by explaining how you determined which phrases were relevant.
4. Validate that the selected text phrases are exact matches from the source documents.
5. Structure your output so that the detailed reasoning is presented first, followed by the final JSON output.

# Output Format
- Provide the reasoning process in clear, logical steps.
- Conclude with a final answer as a valid JSON array of strings.
- Do not modify or paraphrase any extracted text; include the copied text exactly as in the documents.

Example Output:  
Reasoning: [Explain step-by-step how the relevant text phrases were identified from the documents, ensuring that reasoning appears before the conclusion.]  
Final Answer: ["Extracted text phrase 1", "Extracted text phrase 2"]

# Notes
- Ensure reasoning always appears before any conclusions or final outputs.
- The final output must strictly be a JSON array of strings containing the extracted phrases.
- If no relevant phrases are found in a document, omit that document from your output."""
    ),
    ReverseInstructionsDataset.name: (
        """Extract exact text phrases from the provided {doc_type}(s) as a valid JSON array of strings, preserving the order and the exact content. Do not modify the copied text.

Additional details:
- You are given one or more documents identified as {doc_type}(s) along with an instruction defined by {instruction}. Your task is to locate and extract the exact text phrases specified by the instruction from the provided documents.
- You must include a brief reasoning section before presenting the final answer. In the reasoning section, explain how you identified and verified the exact text phrases, but ensure that all conclusions (the final extracted phrases) appear at the end.
- Do not alter, paraphrase, or make any changes to the original text. Only the precise text from the sources should be copied.

Steps:
1. Read and interpret the {instruction} to determine which exact text phrases must be extracted from the {doc_type}(s).
2. Identify and locate all segments in the {doc_type}(s) that fulfill the criteria specified in the instruction.
3. Verify that the extracted text is unmodified from its original form.
4. Output your reasoning process first, followed by the final result as a JSON array of strings.

Output Format:
- First, include a brief section summarizing your reasoning (separated clearly from the final output).
- The final output must be a valid JSON array of strings containing the exact phrases.
- No additional commentary or annotations should be included with the final JSON output.

Expected reasoning section:
"Identified three phrases in the article containing [SPECIFIC_TERM]. Verified that all text segments match the source text exactly."

Expected final output:
["Exact phrase 1", "Exact phrase 2", "Exact phrase 3"]

Notes:
- Always ensure that the reasoning process is presented before the final JSON result.
- If any examples reverse this order, reverse them back to maintain reasoning first and final conclusions last.
- Avoid any alteration of the original text; do not paraphrase or reformat the copied phrases."""
    ),
}

INSTRUCTIONS3 = {
    OpenAspDataset.name: 'From the provided news articles on the topic "{title}", extract every sentence pertaining to "{aspect_label}" and return them verbatim as a JSON array of strings. Do not modify the text.',
    AspectNewsDataset.name: "From the news article below, pick between 1 and 3 sentences that best address the {topic}’s {aspect_description}. Return them verbatim as a valid JSON array of strings.",
    SciFactDataset.name: 'From the provided medical abstract(s), identify sentences that support or oppose the claim: "{claim}". Return those sentences verbatim as a JSON array of strings.',
    DebateSumDataset.name: 'Given the document below, extract short, concise excerpts that cover every piece of evidence for the argument "{argument}". Return those exact excerpts—unaltered—as a JSON array of strings.',
    SparkSaliencyDataset.name: "Given the following documents on the same topic, extract short, salient phrases exactly as written. Return them as a valid JSON array of strings. Do not alter the copied text.",
    SparkEvidenceDataset.name: 'Given the following documents on the same topic, extract brief, precise text snippets that support the statement "{query}". Output these exact snippets verbatim as a JSON array of strings.',
    ReverseInstructionsDataset.name: "Given the following {doc_type}(s), {instruction}. Return the exact text phrases from those {doc_type}(s) as a valid JSON array of strings, without altering them.",
}


ID2INSTRUCTIONS = (
    INSTRUCTIONS,
    INSTRUCTIONS2,
    INSTRUCTIONS3,
)

ICL_INSTRUCTIONS1 = {
    OpenAspDataset.name: (
        "Given the following news articles on the same topic, extract all sentences related to the given aspect. "
        + get_output_format(selection_type="sentences", source_type="documents"),
        "Title: {title}\nAspect: {aspect_label}",
    ),
    AspectNewsDataset.name: (
        "Given the following news article, select at least 1 and at most 3 sentences that are the most relevant to the given aspect. "
        + get_output_format(selection_type="sentences", source_type="document"),
        "Aspect: {topic}'s {aspect_description}",
    ),
    SciFactDataset.name: (
        'Given the following abstract document(s) of medical papers, select the sentences that provide either supporting or refuting evidence for the given claim". '
        + get_output_format(selection_type="sentences", source_type="abstract(s)"),
        "Claim: {claim}",
    ),
    DebateSumDataset.name: (
        "Given the following document, select short and concise text phrases that summarize all the evidence for the given argument. "
        + get_output_format(selection_type="text phrases", source_type="document"),
        "Argument: {argument}",
    ),
    SparkSaliencyDataset.name: (
        "Given the following documents on the same topic, extract short and concise salient text phrases. "
        + get_output_format(selection_type="text phrases", source_type="documents"),
        "",
    ),
    SparkEvidenceDataset.name: (
        "Given the following documents on the same topic, extract short and concise text phrases that provide references to the given statement. "
        + get_output_format(selection_type="text phrases", source_type="documents"),
        "Statement: {query}",
    ),
    ReverseInstructionsDataset.name: (
        "Given the following {doc_type}(s) and an instruction, select text phrases that match the instruction. "
        + get_output_format(selection_type="text phrases", source_type="{doc_type}(s)"),
        "Instruction: {instruction}",
    ),
}
ICL_INSTRUCTIONS2 = {
    OpenAspDataset.name: (
        "From the provided news articles on the topic, extract every sentence pertaining to the aspect and return them verbatim as a JSON array of strings. Do not modify the text.",
        "Title: {title}\nAspect: {aspect_label}",
    ),
    AspectNewsDataset.name: (
        "From the news article below, pick between 1 and 3 sentences that best address the aspect. Return them verbatim as a valid JSON array of strings.",
        "Aspect: {topic}'s {aspect_description}",
    ),
    SciFactDataset.name: (
        "From the provided medical abstract(s), identify sentences that support or oppose the claim. Return those sentences verbatim as a JSON array of strings.",
        "Claim: {claim}",
    ),
    DebateSumDataset.name: (
        "Given the following document, select short and concise text phrases that summarize all the evidence for the given argument. ",
        "Argument: {argument}",
    ),
    SparkSaliencyDataset.name: (
        "Given the following documents on the same topic, extract short, salient phrases exactly as written. Return them as a valid JSON array of strings. Do not alter the copied text.",
        "",
    ),
    SparkEvidenceDataset.name: (
        "Given the following documents on the same topic, extract brief, precise text snippets that support the statement. Output these exact snippets verbatim as a JSON array of strings.",
        "Statement: {query}",
    ),
    ReverseInstructionsDataset.name: (
        "Given the following {doc_type}(s), follow the given instruction. Extract the exact text phrases and return them as a JSON array of strings, preserving the text unchanged.",
        "Instruction: {instruction}",
    ),
}

ICL_INSTRUCTIONS = (
    ICL_INSTRUCTIONS1,
    ICL_INSTRUCTIONS2,
)

CONCISE_PREFIX = (
    "Given the following document, mark with bold the most important and concise text-phrases"
)
CONCISE_SUFFIX = (
    "Copy the exact selected content and mark with tags %(open_tag)s and %(close_tag)s the most important text-phrases, "
    "there could be more than one. Do not change the copied selection."
)


def format_instruction(
    sample: SelectionSample, instruction: str | None = None, instruction_variant_id: int = 0
) -> str:
    tmpl = ID2INSTRUCTIONS[instruction_variant_id]
    instruction = instruction or tmpl[sample.source_dataset]
    return instruction.format(**sample.instruction_context)


def format_docs(sample: SelectionSample) -> str:
    doc_type = sample.instruction_context.get("doc_type", "Doc").title()

    docs_text = []
    for doc in sample.docs:
        docs_text.append(f"{doc_type} #{doc.id}:\n")
        docs_text.append(doc.text.strip())
        docs_text.append("\n\n")
    return "".join(docs_text)


def format_prompt(
    sample: SelectionSample, instruction: str | None = None, instruction_variant_id: int = 0
) -> str:
    prompt_text = [
        format_instruction(sample, instruction, instruction_variant_id),
        "\n\n",
        format_docs(sample),
    ]
    return "".join(prompt_text)


def get_conversation(
    prompt: str, answer: str | None = None, system: str | None = SYSTEM_ROLE
) -> list[dict]:
    conv = []
    if system is not None:
        conv.append({"role": "system", "content": system})

    conv.append({"role": "user", "content": prompt})

    if answer is not None:
        conv.append({"role": "assistant", "content": answer})
    return conv


def selections_to_answer(selections: list[Selection]):
    answer = json.dumps(
        [
            s.content
            for s in sorted(
                selections,
                key=attrgetter("doc_id", "start_pos", "end_pos"),
            )
            if s.doc_id >= 0  # filter hallucination (doc_id = -1)
        ],
        indent=4,
    )
    return f"Selected Content:\n\n{answer}"


def get_icl_prompt_for_sample(
    sample: SelectionSample, train_samples: list[SelectionSample], prompt_variant: int
) -> str:
    instruction, item_tmpl = ICL_INSTRUCTIONS[prompt_variant][sample.source_dataset]
    prompt_text = [format_instruction(sample, instruction), "\n\n"]

    # Add example -> answer
    is_singular = len(train_samples) == 1
    prompt_text.append(
        f"Below {'is an example' if is_singular else 'are examples'} of an input and the correct selected content:\n\n"
    )
    for train_sample in train_samples:
        prompt_text.append(item_tmpl.format(**train_sample.instruction_context))
        prompt_text.append("\n")
        prompt_text.append("Input Document(s):\n\n")
        prompt_text.append(format_docs(train_sample))
        prompt_text.append("\n")

        assert len(train_sample.selections) == 1, "Can only format single reference samples"
        prompt_text.append(selections_to_answer(train_sample.selections[0].selections))
        prompt_text.append("\n\n")

    # Add the eval sample
    prompt_text.append("--- END OF EXAMPLES ---\n\n")
    prompt_text.append("Now, select content from the below document(s):\n\n")
    prompt_text.append(item_tmpl.format(**sample.instruction_context))
    prompt_text.append("\n")
    prompt_text.append("Input Document(s):\n\n")
    prompt_text.append(format_docs(sample))
    prompt_text.append("You're selected content answer is:\n\n")

    return "".join(prompt_text)


def print_example_icl_prompts():
    from igcs.datasets import load_dataset
    from igcs.predict import get_icl_train_samples

    for ds in INSTRUCTIONS:
        print(f"{ds}")
        data = load_dataset(ds, "test")
        icl_samples = get_icl_train_samples(
            sample_id=data[0].id,
            samples_pool=data,
            icl_num_samples=2,
            single_doc=True,
            seed=42,
        )
        p = get_icl_prompt_for_sample(data[0], train_samples=icl_samples, prompt_variant=1)
        print(p)


if __name__ == "__main__":
    # for ds, inst in INSTRUCTIONS.items():
    #     print(f"{ds}: {inst}")
    print_example_icl_prompts()
