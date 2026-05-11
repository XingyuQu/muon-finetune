"""
Utility functions for HumanEval evaluation with EXACT training format.

This matches the codefeedback training data format in data.py:
- Uses template_wo_input for instruction formatting
- The instruction describes the code completion task
"""
import re
import evaluate as hf_evaluate

# Initialize code_eval metric (required for HumanEval)
try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    print(f"Warning: Could not initialize code_eval metric: {e}")
    compute_ = None


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    """Compute pass@k metric for code evaluation."""
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


# =============================================================================
# HumanEval Final Format (EXACT match with training data.py codefeedback)
# =============================================================================

# This template is EXACTLY the same as template_wo_input in data.py
TRAINING_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def format_prompt(doc: dict) -> str:
    """
    Format HumanEval prompt using EXACT training format (codefeedback).

    Training format from data.py:
        x = template_wo_input.format(instruction=data["query"])
        y = "```python\\n...code...\\n```"

    So we format the HumanEval prompt as an instruction similar to CodeFeedback queries.
    """
    # Build instruction similar to CodeFeedback queries
    instruction = f"""Complete the following Python function:

```python
{doc["prompt"]}```"""

    # Use exact training template
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    return prompt


def post_process_code(text: str) -> str:
    """
    Post-process generated code to extract clean Python code.

    This follows the same logic as humaneval_alpaca (utils.py post_process_code),
    since we now ask the model to generate the ENTIRE complete function.

    Steps:
    1. Extract code from markdown code blocks (if present)
    2. Replace tabs with 4 spaces
    3. Remove docstrings that might be regenerated
    4. Remove empty lines and strip trailing whitespace
    5. Normalize indentation based on the 'def' line position
    """
    # First, try to extract code from markdown code block
    # This handles cases where model outputs: "Here is the solution:\n```python\ndef..."
    if "```python" in text:
        # Extract content between ```python and ```
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end == -1:
            # No closing ```, take everything after ```python
            text = text[start:]
        else:
            text = text[start:end]
    elif "```" in text:
        # Handle generic code blocks without language specifier
        start = text.find("```") + 3
        end = text.find("```", start)
        if end == -1:
            text = text[start:]
        else:
            text = text[start:end]
    # If no code blocks, keep text as-is (model might have output raw code)

    # Replace tabs with 4 spaces
    text = text.replace("\t", "    ")

    # Remove docstrings that might be regenerated
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', "", text, flags=re.DOTALL)

    # Remove empty lines and strip trailing whitespace
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])

    lines = text.split("\n")

    # Calculate leading spaces for each line
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r"^( *)", line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)

    # Find the 'def' line and its indentation level
    try:
        def_line = [i for i, line in enumerate(lines) if "def " in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except (IndexError, ValueError):
        # No def line found, return as-is
        def_line_space = 0

    # Build indentation level mapping
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i

    # Rebuild lines with normalized indentation
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())

    return "\n".join(new_lines)


def build_predictions(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """
    Build predictions for HumanEval.

    The model generates the ENTIRE complete function definition,
    so we just post-process and return the generated code directly.
    """
    results = []
    for resp, doc in zip(resps, docs):
        processed_resps = []
        for r in resp:
            # Post-process the generated code (complete function)
            processed = post_process_code(r)
            processed_resps.append(processed)
        results.append(processed_resps)
    return results
