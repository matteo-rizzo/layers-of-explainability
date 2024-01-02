import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.features_extraction.functional.text_features_utils import TextFeatureExtractor

""" Utilities to generate corpus of offensive/common words in the Italian language """


def remove_accent(text):
    """
    Remove accent from text
    :param text: text to remove accent from
    :return: text without accent
    """
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8", "ignore")


def lemmatize_set(words: Iterable[str]) -> list[str]:
    fex = TextFeatureExtractor()
    words = {w.strip() for w in words}
    non_accent = {remove_accent(w).strip() for w in words if remove_accent(w).strip()}
    words |= non_accent

    add_words = set()
    for w in words:
        sw = list()
        doc = fex._TextFeatureExtractor__spacy_model(w)
        for t in doc:
            sw.append(t.lemma_)
        lm = " ".join(sw)
        add_words.add(lm)

    return list(words | add_words)


def badwords_list() -> None:
    """
    Source:
        - https://github.com/valeriobasile/hurtlex
        - https://github.com/napolux/paroleitaliane

    """

    path_1: Path = Path("dataset") / "AMI2020" / "lexicon" / "lista_badwords.txt"
    path_2: Path = Path("dataset") / "AMI2020" / "lexicon" / "hurtlex_IT.tsv"
    path_o: Path = Path("dataset") / "AMI2020" / "lexicon" / "bad_words.txt"

    with open(path_1, mode="rt", encoding="utf-8") as fg:
        words = set(fg.readlines())

    words_df = pd.read_csv(path_2, sep="\t")  # (words_df["level"] == "conservative")

    # Concat all words
    words |= set(words_df[(words_df["category"].isin(["ps", "pr", "asf", "asm", "om"]))]["lemma"].tolist())

    all_words = lemmatize_set(words)

    with open(path_o, mode="wt", encoding="utf-8") as wf:
        for w in all_words:
            wf.write(f"{w}\n")


def common_words_list() -> None:
    """
    Source: https://github.com/napolux/paroleitaliane

    """
    path_1: Path = Path("dataset") / "AMI2020" / "lexicon" / "1000_parole_italiane_comuni.txt"
    path_o: Path = Path("dataset") / "AMI2020" / "lexicon" / "common_words.txt"

    with open(path_1, mode="rt", encoding="utf-8") as fg:
        words = set(fg.readlines())

    all_words = lemmatize_set(words)

    with open(path_o, mode="wt", encoding="utf-8") as wf:
        for w in all_words:
            wf.write(f"{w}\n")


if __name__ == "__main__":
    badwords_list()
    common_words_list()
