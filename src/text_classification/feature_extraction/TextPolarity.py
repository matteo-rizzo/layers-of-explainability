from collections import defaultdict
from typing import Any

from textblob.tokenizers import SentenceTokenizer
from tqdm import tqdm

from src.text_classification.feature_extraction.Feature import Feature
from textblob import TextBlob


class TextPolarity(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 512):
        self.pipe = TextBlob
        # self.pipe = pipeline("text-classification", model="fabriceyhc/bert-base-uncased-amazon_polarity", device="cuda" if use_gpu else "cpu", top_k=None, batch_size=batch_size)

    def extract(self, texts: list[str]) -> Any:
        feature_df: dict[str, list[float]] = defaultdict(list)
        for t in tqdm(texts, desc="polarity analysis"):
            processed_text = self.pipe(t, tokenizer=SentenceTokenizer())
            feature_df["polarity"] = processed_text.sentiment_assessments.polarity
            feature_df["subjectivity"] = processed_text.sentiment_assessments.subjectivity
            feature_df["word_count"] = len(processed_text.words)
            feature_df["sentence_len"] = sum([len(s.words) for s in processed_text.sentences]) / len(processed_text.sentences)
        return feature_df
