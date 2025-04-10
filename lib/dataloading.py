import pandas as pd


def load_conllu(file_path):
    """
    Loads a CoNLL-U formatted file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CoNLL-U file.

    Returns:
        pd.DataFrame: DataFrame containing the CoNLL-U data with "ID" as the index.
    """
    columns = [
        "ID",
        "FORM",
        "LEMMA",
        "UPOS",
        "XPOS",
        "FEATS",
        "HEAD",
        "DEPREL",
        "DEPS",
        "MISC",
    ]
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            fields = line.strip().split("\t")
            if len(fields) == len(columns):
                data.append(fields)

    df = pd.DataFrame(data, columns=columns)
    df.set_index("ID", inplace=True)

    return df


def clear_non_UPOS_tags(df):
    """
    Clears non-UPOS tags from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a "UPOS" column.
    
    Returns:
        pd.DataFrame: DataFrame with only valid UPOS tags in the "UPOS" column.

    Raises:
        ValueError: If the DataFrame does not contain a "UPOS" column.
    """
    if "UPOS" not in df.columns:
        raise ValueError("DataFrame must contain a 'UPOS' column.")

    upos_tags = [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]
    dropped_tags = df[~df["UPOS"].isin(upos_tags)]
    if not dropped_tags.empty:
        print(
            f"Dropped {len(dropped_tags)} rows with non-UPOS tags \n"
            f"Tags dropped: {dropped_tags["UPOS"].unique()}"
        )
    return df[df["UPOS"].isin(upos_tags)]
