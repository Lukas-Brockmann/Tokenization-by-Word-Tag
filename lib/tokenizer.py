import numpy as np
import tokenizers
import json
import tempfile
import os
import pandas as pd


def train_tokenizer(text: list[str], vocab_size: int, algorithm: str = "BPE") -> tokenizers.Tokenizer:
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
        ValueError: If the selected algorithm is not supported.
    """
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Vocab size must be a positive integer.")
    if vocab_size < 6:
        raise ValueError("Vocab size must be at least 6 to accommodate special tokens.")
    if not isinstance(text, list) or len(text) == 0:
        raise ValueError("Text must be a non-empty list of strings.")

    match algorithm.lower():
        case "bpe":
            tokenizer = tokenizers.Tokenizer(
                tokenizers.models.BPE(
                    unk_token="[UNK]"
                )
            )
        case "unigram":
            tokenizer = tokenizers.Tokenizer(
                tokenizers.models.Unigram()
            )
        case "wordpiece":
            tokenizer = tokenizers.Tokenizer(
                tokenizers.models.WordPiece(
                    unk_token="[UNK]"
                )
            )
        case _:
            raise ValueError(f"Tokenizer algorithm {algorithm} not supported.")

    # Preprocessing
    tokenizer.normalizer = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.NFD(),  # Unicode Normalizer
            tokenizers.normalizers.Lowercase(),
            tokenizers.normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [tokenizers.pre_tokenizers.Metaspace()]
    )

    # Trainer
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    match algorithm.lower():
        case "bpe":
            trainer = tokenizers.trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=special_tokens,
            )
        case "unigram":
            trainer = tokenizers.trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                unk_token="[UNK]"
            )
        case "wordpiece":
            trainer = tokenizers.trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=2,
                special_tokens=special_tokens,
            )

    tokenizer.train_from_iterator(text, trainer=trainer)

    # Postprocessing
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.decoder = tokenizers.decoders.Metaspace(replacement="▁")

    return tokenizer


def extract_vocab_and_merges(
    tokenizer: tokenizers.Tokenizer,
) -> tuple[dict[str, int], list[tuple[str, str]]]:
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

        if "merges" in data["model"].keys():
            return data["model"]["vocab"], [
                tuple(merge) for merge in data["model"]["merges"]
            ]
        else:
            return data["model"]["vocab"], [tuple(["", ""])]

    finally:
        os.remove(path)


def assign_proportionally(current: dict[str, int], target: dict[str, int], n: int) -> np.ndarray:
    """
    Assigns n new tokens to the current list of tokens based on the target proportions.
    Args:
        current (dict[str, int]): Current distribution.
        target (dict[str, int]): Desired distribution of tokens in percent.
        n (int): Number of new tokens to assign.
    Returns:
        dicht[str, int]: Update of current with n new tokens assigned following the target proportions.
    Raises:
        AssertionError: If the lengths of current and target do not match
        AssertionError: If the keys of current and target do not match.
        AssertionError: If n is not a positive integer.
        AssertionError: If any values in current are negative
        AssertionError: If any values in target are negative.
        AssertionError: If the sum of target does not equal 1. (margin of 1e-6)

    """
    assert len(current) == len(target), "Current and target must have the same length."
    assert current.keys() == target.keys(), "Current and target must have the same keys."
    assert n > 0, "n must be a positive integer."
    assert all(x >= 0 for x in current.values()), "Current counts must be non-negative."
    assert all(x >= 0 for x in target.values()), "Target percentages must be non-negative."
    assert abs(sum(target.values()) - 1) < 1e-6, "Target percentages must sum to 1."

    if not isinstance(n, int):
        n = int(n)

    current_np = np.array([current[key] for key in current.keys()], dtype=int)
    target_np = np.array([target[key] for key in current.keys()], dtype=float)

    target_new = (target_np * (current_np.sum() + n)) - current_np.sum()
    added_indices = np.zeros(len(current_np), dtype=int)

    for _ in range(n):
        idx = np.argmax(target_new)
        added_indices[idx] += 1
        target_new[idx] -= 1

    return dict(zip(current.keys(), added_indices + current_np))


def vocab_allocation():
    pass


def tokenizer_from_vocab_and_merges(
    tokenizer_algorithm: str,
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    save_path: str = None,
) -> tokenizers.Tokenizer:
    """
    Creates a tokenizer from a vocabulary and merges.
    Args:
        tokenizer_algorithm (str): The type of tokenizer to create. Options are ["bpe"].
        vocab (dict): A dictionary mapping tokens to their IDs.
        merges (list): A list of (token1, token2) tuples representing merge rules.
        save_path (str): Path to save the tokenizer. If None, the tokenizer is not saved.
    Returns:
        tokenizer (Tokenizer): A Hugging Face Tokenizer object.
    Raises:
        ValueError: If the tokenizer type is not supported.
    """

    match tokenizer_algorithm.lower():
        case "bpe":
            tokenizer = tokenizers.Tokenizer(
                tokenizers.models.BPE(
                    vocab=vocab,
                    merges=merges,
                    unk_token="[UNK]",
                )
            )
        case "unigram":
            unk_id = vocab[('[UNK]', 0.0)]
            vocab = list(vocab.keys())
            tokenizer = tokenizers.Tokenizer(
                tokenizers.models.Unigram(
                    vocab=vocab,
                    unk_id=unk_id
                )
            )
        case "wordpiece":
            tokenizer = tokenizers.Tokenizer(
                tokenizers.models.WordPiece(
                    vocab=vocab,
                    unk_token="[UNK]"
                )
            )
        case _:
            raise ValueError(f"Tokenizer type {tokenizer_algorithm} not supported.")

    tokenizer.add_special_tokens(["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

    tokenizer.normalizer = tokenizers.normalizers.Sequence(
        [
            tokenizers.normalizers.NFD(),  # Unicode Normalizer
            tokenizers.normalizers.Lowercase(),
            tokenizers.normalizers.StripAccents(),
        ]
    )

    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [tokenizers.pre_tokenizers.Metaspace()]
    )

    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    tokenizer.decoder = tokenizers.decoders.Metaspace(replacement="▁")
    if save_path:
        tokenizer.save(save_path)
    return tokenizer


def train_and_merge_tokenizers(
    train_df: pd.DataFrame,
    tokenizer_algorithm: str,
    vocab_size: int,
    allocation: str,
    allocation_weights: list[float] = None,
    grouping: list[list[str]] = None,
    strict: bool = True,
    save_path: str = None,
) -> tokenizers.Tokenizer:

    if not grouping:  # default to treating each UPOS tag as a seperate group
        grouping = [
            ["ADJ"],
            ["ADP"],
            ["ADV"],
            ["AUX"],
            ["CCONJ"],
            ["DET"],
            ["INTJ"],
            ["NOUN"],
            ["NUM"],
            ["PART"],
            ["PRON"],
            ["PROPN"],
            ["PUNCT"],
            ["SCONJ"],
            ["SYM"],
            ["VERB"],
            ["X"],
        ]

    if allocation.lower() == "weighted_proportional":
        if allocation_weights is None:
            raise ValueError("Allocation weights must be provided for weighted proportional allocation.")
        if len(allocation_weights) != len(grouping):
            raise ValueError("Allocation weights must match the number of groups.")
        if any(weight < 0 for weight in allocation_weights):
            raise ValueError("Allocation weights must be non-negative.")

    tokenizers, merges, vocab, target_allocation, group_vocab_size = {}, {}, {}, {}, {}
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    # Ensure the Form column can be converted to string
    train_df.dropna(subset=["FORM"], inplace=True)
    train_df["FORM"] = train_df["FORM"].astype(str)

    match allocation.lower():
        case "proportional":
            allocation = train_df["UPOS"].value_counts(normalize=True).sort_index()
            for group in grouping:
                group_name = ", ".join(group)
                target_allocation[group_name] = allocation.loc[allocation.index.intersection(group)].sum()
        case "weighted_proportional":
            allocation = train_df["UPOS"].value_counts(normalize=True).sort_index()
            for weight, group in zip(allocation_weights, grouping):
                group_name = ", ".join(group)
                target_allocation[group_name] = allocation.loc[allocation.index.intersection(group)].sum() * weight
            target_allocation = {k: v / sum(target_allocation.values()) for k, v in target_allocation.items()}
            assert sum(target_allocation.values()) - 1 < 1e-6, "Target allocation must sum to 1."
        case _:
            raise ValueError(f"Allocation type {allocation} not supported.")

    for group in grouping:
        group_name = ", ".join(group)
        text = train_df[train_df["UPOS"].isin(group)]["FORM"].values.tolist()
        if not text:
            text = [""]
        tokenizers[group_name] = train_tokenizer(text, vocab_size, tokenizer_algorithm)
        group_vocab_size[group_name] = tokenizers[group_name].get_vocab_size()
        vocab[group_name], merges[group_name] = extract_vocab_and_merges(
            tokenizers[group_name]
        )
        target_allocation[group_name] = allocation.loc[
            allocation.index.intersection(group)
        ].sum()

    if strict:  # Only ensure space for the five special tokens
        vocab_allocation = {", ".join(group): 5 for group in grouping}

    vocab_set = set()
    exhausted = False
    while len(vocab_set) < vocab_size and not exhausted:
        vocab_allocation = assign_proportionally(vocab_allocation, target_allocation, vocab_size - len(vocab_set))
        for group in grouping:
            group_name = ", ".join(group)

            # If the tokenizer trained on a group has no new tokens to allocate, we try to allocate the remaining tokens to other groups
            # If all groups are exhausted, we stop the allocation process
            if vocab_allocation[group_name] > group_vocab_size[group_name]:
                target_allocation[group_name] = 0
                total = sum(target_allocation.values())
                if total == 0:
                    print("Warning: No tokens left to allocate. All groups have been exhausted.")
                    exhausted = True
                    break
                target_allocation = {key: value / total for key, value in target_allocation.items()}

            if tokenizer_algorithm.lower() == "unigram": # Unigram return (token, score) pairs 
                vocab_set.update([tuple(pair) for pair in vocab[group_name]][:vocab_allocation[group_name]])
            else:
                vocab_set.update(list(vocab[group_name])[:vocab_allocation[group_name]])

    res_vocab = {token: idx for idx, token in enumerate(vocab_set)}
    res_merges = []

    if tokenizer_algorithm.lower() in ["bpe"]:
        merges_set = set()
        for group in grouping:
            group_name = ", ".join(group)
            for merge in merges[group_name]:
                if all(token in vocab_set for token in [merge[0], merge[1], merge[0] + merge[1]]):
                    merges_set.add(merge)

        res_merges = list(merges_set)

    tokenizer = tokenizer_from_vocab_and_merges(
        tokenizer_algorithm,
        res_vocab,
        res_merges,
        save_path=save_path
    )
    
    return tokenizer