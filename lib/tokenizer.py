import numpy as np
import tokenizers
import json
import tempfile
import os
from typing import Dict, List, Tuple, Union
import pandas as pd


def train_tokenizer(
    text, vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
):
    """
    Trains a BPE tokenizer on the provided text.

    The tokenizer is trained until vocab_size is reached or no more matches of alteast two tokens can be made.
    There will always be the <UNK> token in the vocabulary, plus any special tokens provided.

    Args:
        text (list of str): The text to train the tokenizer on.
        vocab_size (int): The desired vocabulary size.
        special_tokens (list of str): List of special tokens to include in the vocabulary.

    Returns:
        tokenizers.Tokenizer: The trained tokenizer.

    Raises:
        ValueError: If vocab_size is less than the number of special tokens.
        ValueError: If the text is emtpy or if vocab_size is not a postiive integer

    """
    if vocab_size < len(special_tokens):
        raise ValueError(
            "Vocab size must be greater than the number of special tokens."
        )
    if not isinstance(text, list) or len(text) == 0:
        raise ValueError("Text must be a non-empty list of strings.")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Vocab size must be a positive integer.")

    bpe = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<UNK>"))

    # Preprocessing
    bpe.normalizer = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.NFD(),  # Unicode Normalizer
            tokenizers.normalizers.Lowercase(),
            tokenizers.normalizers.StripAccents(),
        ]
    )
    bpe.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [tokenizers.pre_tokenizers.Metaspace()]
    )

    # Trainer
    bpe_trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    bpe.train_from_iterator(text, trainer=bpe_trainer)

    # Postprocessing
    bpe.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", bpe.token_to_id("[CLS]")),
            ("[SEP]", bpe.token_to_id("[SEP]")),
        ],
    )
    bpe.decoder = tokenizers.decoders.Metaspace(replacement="▁")

    return bpe


def extract_vocab_and_merges(
    tokenizer: tokenizers.Tokenizer,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Given a Tokenizer from the Hugging Face tokenizers library, get the merges performed

    Args:
        tokenizer (Tokenizer): A Hugging Face Tokenizer object.

    Returns:
        vocab (dict): A dictionary mapping tokens to their IDs.
        merges (list): A list of (token1, token2) tuples representing merge rules.
    """
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp_file:
        path = tmp_file.name
        tokenizer.save(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["model"]["vocab"], [
            tuple(merge) for merge in data["model"]["merges"]
        ]

    finally:
        os.remove(path)

def assign_proportionally(current, target, n) -> np.ndarray:
    """
    Assigns n new tokens to the current list of tokens based on the target proportions.
    Args:
        current (list): Current distribution.
        target (list): Desired distribution of tokens in percent.
        n (int): Number of new tokens to assign.
    Returns:
        np.ndarray: Array of indices that can be added to current, while maintaining the target proportions.
    Raises:
        AssertionError: If the lengths of current and target do not match
        AssertionError: If n is not a positive integer.
        AssertionError: If any element in current is negative
        AssertionError: If any element in target is negative.
        AssertionError: If the sum of target does not equal 1. (margin of 1e-6)

    """
    assert len(current) == len(target), "Current and target must have the same length."
    assert n > 0, "n must be a positive integer."
    assert all(x >= 0 for x in current), "Current counts must be non-negative."
    assert all(x >= 0 for x in target), "Target percentages must be non-negative."
    assert abs(sum(target) - 1) < 1e-6, "Target percentages must sum to 1."

    if not isinstance(current, np.ndarray):
        current = np.array(current, dtype=int)
    if not isinstance(target, np.ndarray):
        target = np.array(target, dtype=float)
    if not isinstance(n, int):
        n = int(n)

    target_new = (target * (current.sum() + n)) - current.sum()
    added_indices = np.zeros(len(current), dtype=int)

    for _ in range(n):
        idx = np.argmax(target_new)
        added_indices[idx] += 1
        target_new[idx] -= 1

    return added_indices