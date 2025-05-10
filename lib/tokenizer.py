import numpy as np
import tokenizers
import json
import tempfile
import os
from typing import Dict, List, Tuple
import pandas as pd


def train_tokenizer(text: list[str], vocab_size: int) -> tokenizers.Tokenizer:
    """
    Trains a BPE tokenizer on the provided text.

    The tokenizer is trained until vocab_size is reached or no more matches of alteast two tokens can be made.
    There following special tokens are added, these take up space in the vocab size:
    - <UNK>: Unknown token
    - <PAD>: Padding token
    - <CLS>: Class token
    - <SEP>: Separator token
    - <MASK>: Mask token
    - ▁: Beginning of word token

    Args:
        text (list of str): The text to train the tokenizer on.
        vocab_size (int): The desired vocabulary size.

    Returns:
        tokenizers.Tokenizer: The trained tokenizer.

    Raises:
        ValueError: If vocab_size is less than the number of special tokens (6).
        ValueError: If the text is emtpy or not a list of strings.
        ValueError: If vocab_size is not a positive integer.
    """
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Vocab size must be a positive integer.")
    if vocab_size < 6:
        raise ValueError("Vocab size must be at least 6 to accommodate special tokens.")
    if not isinstance(text, list) or len(text) == 0:
        raise ValueError("Text must be a non-empty list of strings.")

    bpe = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            unk_token="[UNK]",
            padding_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
    )

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
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
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


def assign_proportionally(current: List[int], target: List[int], n: int) -> np.ndarray:
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

def vocab_allocation():
    pass

def tokenizer_from_vocab_and_merges(
    tokenizer_type: str,
    vocab: Dict[str, int],
    merges: List[Tuple[str, str]],
    save_path: str = None
) -> tokenizers.Tokenizer:
    """
    Creates a tokenizer from a vocabulary and merges.
    Args:
        tokenizer_type (str): The type of tokenizer to create. Options are ["bpe"].
        vocab (dict): A dictionary mapping tokens to their IDs.
        merges (list): A list of (token1, token2) tuples representing merge rules.
        save_path (str): Path to save the tokenizer. If None, the tokenizer is not saved.
    Returns:
        tokenizer (Tokenizer): A Hugging Face Tokenizer object.
    Raises:
        ValueError: If the tokenizer type is not supported.
    """
    
    match tokenizer_type.lower():
        case "bpe":
            tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(
                vocab=vocab,
                merges=merges,
                unk_token="[UNK]",
            ))
        case _:
            raise ValueError(f"Tokenizer type {tokenizer_type} not supported.")

    tokenizer.add_special_tokens(["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

    tokenizer.normalizer = tokenizers.normalizers.Sequence(
    [
        tokenizers.normalizers.NFD(),  # Unicode Normalizer
        tokenizers.normalizers.Lowercase(),
        tokenizers.normalizers.StripAccents(),
    ])

    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [tokenizers.pre_tokenizers.Metaspace()]
    )

    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],)

    tokenizer.decoder = tokenizers.decoders.Metaspace(replacement="▁")
    if save_path:
        tokenizer.save(save_path)
    return tokenizer