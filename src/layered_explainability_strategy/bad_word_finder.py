import pandas as pd

from src.feature_extraction.text_features import TextFeatureExtractor


def examine_single_doc(tokenized_doc: list[str], bad_words: list[str]) -> float:
    length: int = len(tokenized_doc)  # Number of words
    if length == 0:
        return 0.0
    num_bad_words: int = sum([w in bad_words for w in tokenized_doc])  # Number of bad words
    return num_bad_words / length


def examine_docs(docs: list[str]):
    bad_words_df = pd.read_csv("dataset/asset/profanity_en.csv")

    bw = set(bad_words_df["text"].dropna().tolist())
    cf1 = set(bad_words_df["canonical_form_1"].dropna().tolist())
    cf2 = set(bad_words_df["canonical_form_2"].dropna().tolist())
    cf3 = set(bad_words_df["canonical_form_3"].dropna().tolist())

    bad_words = list(bw.union(cf1.union(cf2.union(cf3))))
    tokenizer = TextFeatureExtractor(language="en").preprocessing_tokenizer
    tokenized_docs = [tokenizer(doc) for doc in docs]
    frequencies = [examine_single_doc(t_doc, bad_words) for t_doc in tokenized_docs]
    return frequencies


if __name__ == "__main__":
    print(examine_docs(["banana", "lol no vabbe", "I", "fucking reeee", "lmao what a fucking bitch"]))
    # print(examine_single_doc(["I", "love", "bananas"], ["darn", "homie", "bananas"]))
