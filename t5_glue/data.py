from datasets import DatasetDict, load_dataset


# =============================================================================
# GLUE Datasets (mrpc, cola, sst2, qnli, mnli)
# - Uses stratified split for train/eval/test
# =============================================================================

def _glue_split_and_format(dataset, prompt_func, label_map, test_size=0.1, dataset_name="default"):
    """
    Helper function for GLUE datasets:
    - Format data with prompt_func and label_map
    - Split original train into train/eval with stratification
    - Use original validation as test set
    """
    # GLUE's `test*` splits have label=-1 (hidden by design — labels live on the
    # leaderboard server). We don't use them anyway, so drop them before mapping
    # to avoid `KeyError: -1` when looking up label_map[e["label"]].
    dataset = DatasetDict({k: v for k, v in dataset.items() if not k.startswith("test")})
    ds = dataset.map(
        lambda e: {
            "x": prompt_func(e),
            "y": label_map[e["label"]],
        },
        desc=f"Formatting {dataset_name}",
    )

    original_train = ds["train"]
    if "validation" in ds:
        original_validation = ds["validation"]
    else:
        original_validation = ds["validation_matched"]  # mnli

    split_ds = original_train.train_test_split(
        test_size=test_size,
        seed=42,
        stratify_by_column="label"
    )

    return split_ds["train"], split_ds["test"], original_validation


def load_sst2():
    """SST-2: Sentiment classification"""
    dataset = load_dataset("glue", "sst2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive"}

    def prompt(e):
        return f'{instruction}{e["sentence"]}\nresult: '

    return _glue_split_and_format(dataset, prompt, label_map, dataset_name="sst2")


def load_cola():
    """CoLA: Grammaticality classification"""
    dataset = load_dataset("glue", "cola")
    instruction = "classify the grammaticality of the text: "
    label_map = {0: "unacceptable", 1: "acceptable"}

    def prompt(e):
        return f'{instruction}{e["sentence"]}\nresult: '

    return _glue_split_and_format(dataset, prompt, label_map, dataset_name="cola")


def load_mrpc():
    """MRPC: Paraphrase detection"""
    dataset = load_dataset("glue", "mrpc")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent"}

    def prompt(e):
        return f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: '

    return _glue_split_and_format(dataset, prompt, label_map, dataset_name="mrpc")


def load_qnli():
    """QNLI: Question-answering NLI"""
    dataset = load_dataset("glue", "qnli")
    instruction = "classify the semantic similarity of the question and the sentence: "
    label_map = {0: "entailment", 1: "not_entailment"}

    def prompt(e):
        return f'{instruction}{e["question"]}\n{e["sentence"]}\nresult: '

    return _glue_split_and_format(dataset, prompt, label_map, dataset_name="qnli")


def load_mnli():
    """MNLI: Multi-genre NLI (uses validation_matched as test split)"""
    dataset = load_dataset("glue", "mnli")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def prompt(e):
        return f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: '

    return _glue_split_and_format(dataset, prompt, label_map, dataset_name="mnli")


DATASET_MAP = {
    "sst2": load_sst2,
    "cola": load_cola,
    "mrpc": load_mrpc,
    "qnli": load_qnli,
    "mnli": load_mnli,
}
