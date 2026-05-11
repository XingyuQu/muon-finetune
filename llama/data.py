from datasets import load_dataset, Dataset
import typing as tp
import logging

log = logging.getLogger(__name__)


# =============================================================================
# Instruction-tuning Datasets (wizard_lm, codefeedback, meta_math)
# Uses HF Dataset's built-in caching with .filter() and .map()
# =============================================================================

template_with_input = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

template_wo_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

# Shared tokenizer instance for length filtering (lazy loaded)
_shared_tokenizer = None
_shared_tokenizer_model = None

def _get_shared_tokenizer(model_name: str = "meta-llama/Llama-2-7b-hf"):
    """Get shared tokenizer instance for length filtering."""
    global _shared_tokenizer, _shared_tokenizer_model
    if _shared_tokenizer is None or _shared_tokenizer_model != model_name:
        from transformers import AutoTokenizer
        _shared_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _shared_tokenizer_model = model_name
        log.info(f"Loaded tokenizer for length filtering: {model_name}")
    return _shared_tokenizer


def _length_filter_and_split(
    dataset: Dataset,
    tokenizer,
    max_tokens: int,
    train_size: int,
    eval_size: int,
    dataset_name: str,
) -> tp.Tuple[Dataset, Dataset, Dataset]:
    """Common tail: filter by token length, keep only x/y, then split.

    Datasets are already shuffled before being passed in, so the split is order-preserving.
    """
    def length_filter_fn(example):
        text = example["x"] + " " + example["y"]
        return len(tokenizer(text)["input_ids"]) < max_tokens

    dataset = dataset.filter(
        length_filter_fn,
        desc=f"Filtering {dataset_name} (max_tokens={max_tokens})",
    )
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["x", "y"]])
    return _instruction_dataset_split_no_shuffle(dataset, train_size, eval_size)


def _instruction_dataset_split_no_shuffle(
    dataset: Dataset,
    train_size: int,
    eval_size: int,
) -> tp.Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/eval/test sets WITHOUT shuffle.
    Use this when dataset is already shuffled before filtering.

    Args:
        dataset: Filtered and formatted dataset with 'x' and 'y' columns (already shuffled)
        train_size: Number of training samples
        eval_size: Number of eval samples

    Returns:
        (train_set, eval_set, test_set) - test_set is same as eval_set
    """
    total_size = train_size + eval_size

    if len(dataset) < total_size:
        log.warning(f"Dataset has {len(dataset)} samples, less than requested {total_size}")
        total_size = len(dataset)
        train_size = int(total_size * 0.9)
        eval_size = total_size - train_size

    dataset = dataset.select(range(total_size))

    # Split
    train_set = dataset.select(range(train_size))
    eval_set = dataset.select(range(train_size, train_size + eval_size))

    log.info(f"Dataset split: train={len(train_set)}, eval={len(eval_set)}")

    return train_set, eval_set, eval_set


def load_wizardlm(max_tokens: int = 1024, train_size: int = 52000, eval_size: int = 18000, seed: int = 42, model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Load WizardLM Chinese instruction dataset.

    Uses HF Dataset's built-in caching for filter and map operations.
    Note: Shuffle is applied BEFORE filtering to match original behavior.
    """
    log.info("Loading WizardLM dataset...")
    dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split="train")
    tokenizer = _get_shared_tokenizer(model_name)

    # Shuffle FIRST to match original behavior (shuffle before filtering)
    dataset = dataset.shuffle(seed=seed)

    # Filter: remove AI-refusal responses
    def filter_fn(example):
        output_lower = example["output"].lower()
        if "sorry" in output_lower or "as an ai" in output_lower:
            return False
        return True

    dataset = dataset.filter(filter_fn, desc="Filtering WizardLM (AI refusals)")

    # Format: create x, y columns
    def format_fn(example):
        x = template_wo_input.format(instruction=example["instruction"])
        y = example["output"]
        return {"x": x, "y": y}

    dataset = dataset.map(format_fn, desc="Formatting WizardLM")

    return _length_filter_and_split(dataset, tokenizer, max_tokens, train_size, eval_size, "WizardLM")


def load_codefeedback(max_tokens: int = 1024, train_size: int = 100000, eval_size: int = 10000, seed: int = 42, model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Load CodeFeedback instruction dataset.

    Uses HF Dataset's built-in caching for filter and map operations.
    Note: Shuffle is applied BEFORE filtering to match original behavior.
    """
    log.info("Loading CodeFeedback dataset...")
    dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    tokenizer = _get_shared_tokenizer(model_name)

    # Shuffle FIRST to match original behavior (shuffle before filtering)
    dataset = dataset.shuffle(seed=seed)

    # Filter: must contain code block
    def filter_fn(example):
        return "```" in example["answer"]

    dataset = dataset.filter(filter_fn, desc="Filtering CodeFeedback (code blocks)")

    # Format: create x, y columns (truncate to first code block)
    def format_fn(example):
        x = template_wo_input.format(instruction=example["query"])
        # Keep only first code block
        y = "```".join(example["answer"].split("```")[:2]) + "```"
        return {"x": x, "y": y}

    dataset = dataset.map(format_fn, desc="Formatting CodeFeedback")

    return _length_filter_and_split(dataset, tokenizer, max_tokens, train_size, eval_size, "CodeFeedback")


def load_meta_math(max_tokens: int = 512, train_size: int = 100000, eval_size: int = 10000, seed: int = 42, model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Load MetaMathQA dataset (GSM subset only).

    Uses HF Dataset's built-in caching for filter and map operations.
    Note: Shuffle is applied BEFORE filtering to match original behavior.
    """
    log.info("Loading MetaMathQA dataset...")
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    tokenizer = _get_shared_tokenizer(model_name)

    # Shuffle FIRST to match original behavior (shuffle before filtering)
    dataset = dataset.shuffle(seed=seed)

    # Filter: GSM type only
    def filter_fn(example):
        return "GSM" in example["type"]

    dataset = dataset.filter(filter_fn, desc="Filtering MetaMathQA (GSM only)")

    # Format: create x, y columns
    def format_fn(example):
        x = f'Q: {example["query"]}\nA: '
        # Remove "The answer is:" suffix
        y = example["response"].split("\nThe answer is:")[0]
        return {"x": x, "y": y}

    dataset = dataset.map(format_fn, desc="Formatting MetaMathQA")

    return _length_filter_and_split(dataset, tokenizer, max_tokens, train_size, eval_size, "MetaMathQA")


DATASET_MAP = {
    "wizard_lm": load_wizardlm,
    "codefeedback": load_codefeedback,
    "meta_math": load_meta_math,
}
