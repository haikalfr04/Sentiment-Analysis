"""Loading, cleaning and tokenizing the IMDB movie reviews dataset."""

import os
import re

from datasets import DatasetDict, load_dataset, load_from_disk

DATASET_NAME = "stanfordnlp/imdb"
LABELS = ["negative", "positive"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {label: i for i, label in ID2LABEL.items()}

# Sun et al. (2019), "How to Fine-Tune BERT for Text Classification": keeping the
# first 128 and last 382 tokens of a long review works better than plain head
# truncation. That is roughly a 25% / 75% split of the token budget.
HEAD_RATIO = 0.25

# Saved next to a fine-tuned model so inference tokenizes exactly like training.
INFERENCE_CONFIG = "inference_config.json"

_BR_TAG = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove the HTML line breaks IMDB reviews are full of and normalize whitespace."""
    text = _BR_TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _subsample(dataset, n, seed):
    if n is None or n >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(n))


def load_imdb(
    dataset_name: str = DATASET_NAME,
    val_size: float = 0.1,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    """Return cleaned train / validation / test splits.

    IMDB ships only train (25k) and test (25k). A stratified validation split is
    carved out of train so model selection never looks at the test set.
    `dataset_name` may also be a local directory created with `save_to_disk`.
    """
    if os.path.isdir(dataset_name):
        raw = load_from_disk(dataset_name)
    else:
        raw = load_dataset(dataset_name)

    split = raw["train"].train_test_split(test_size=val_size, seed=seed, stratify_by_column="label")
    ds = DatasetDict(
        train=_subsample(split["train"], max_train_samples, seed),
        validation=_subsample(split["test"], max_eval_samples, seed),
        test=_subsample(raw["test"], max_eval_samples, seed),
    )
    return ds.map(lambda batch: {"text": [clean_text(t) for t in batch["text"]]}, batched=True)


def head_tail_truncate(tokenizer, texts: list[str], max_length: int, head_ratio: float = HEAD_RATIO) -> list[str]:
    """Shorten texts longer than `max_length` tokens to their head and tail.

    Works on character offsets, so it is independent of the tokenizer class and
    the returned text stays human-readable (useful for error analysis).
    Requires a fast tokenizer.
    """
    budget = max_length - tokenizer.num_special_tokens_to_add()
    n_head = int(budget * head_ratio)
    n_tail = budget - n_head
    encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True, verbose=False)

    shortened = []
    for text, offsets in zip(texts, encoded["offset_mapping"]):
        if len(offsets) <= budget:
            shortened.append(text)
            continue
        head_end = offsets[n_head - 1][1]
        tail_start = offsets[-n_tail][0]
        shortened.append(text[:head_end] + " " + text[tail_start:])
    return shortened


def encode(tokenizer, texts: list[str], max_length: int, truncation: str = "head"):
    """Tokenize texts using the given truncation strategy ("head" or "head_tail")."""
    if truncation == "head_tail":
        texts = head_tail_truncate(tokenizer, texts, max_length)
    elif truncation != "head":
        raise ValueError(f"Unknown truncation strategy: {truncation!r}")
    # The head_tail split can re-tokenize a word slightly differently at the seam,
    # so plain truncation is still applied as a safety net.
    return tokenizer(texts, truncation=True, max_length=max_length)


def tokenize_dataset(ds: DatasetDict, tokenizer, max_length: int, truncation: str = "head") -> DatasetDict:
    return ds.map(
        lambda batch: encode(tokenizer, batch["text"], max_length, truncation),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
