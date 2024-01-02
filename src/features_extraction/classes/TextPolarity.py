from collections import defaultdict

from textblob import TextBlob
from textblob.tokenizers import SentenceTokenizer
from tqdm import tqdm

from src.features_extraction.classes.Feature import Feature


class TextPolarity(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 512, *args, **kwargs):
        self.pipe = TextBlob

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        feature_df: dict[str, list[float]] = defaultdict(list)
        # print([a for a, _ in enumerate(texts) if not _]) # Detect null texts
        for t in tqdm(texts, desc="polarity analysis"):
            if t:
                processed_text = self.pipe(t, tokenizer=SentenceTokenizer())
                polarity = processed_text.sentiment_assessments.polarity
                subjectivity = processed_text.sentiment_assessments.subjectivity
                word_count = len(processed_text.words)
                sentence_len = sum([len(s.words) for s in processed_text.sentences]) / len(processed_text.sentences)
            else:
                polarity = .0
                subjectivity = .0
                word_count = 0
                sentence_len = .0
            feature_df[f"{self.__class__.__name__}_polarity"].append(polarity)
            feature_df[f"{self.__class__.__name__}_subjectivity"].append(subjectivity)
            feature_df[f"{self.__class__.__name__}_word_count"].append(word_count)
            feature_df[f"{self.__class__.__name__}_sentence_len"].append(sentence_len)
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        return {
            f"{cls.__name__}_polarity": "polarity score (-1=negative, 0=neutral, 1=positive)",
            f"{cls.__name__}_subjectivity": "subjectivity score (0=very objective, 1=very subjective)",
            f"{cls.__name__}_word_count": "number of words in the text",
            f"{cls.__name__}_sentence_len": "average sentence length (in words)"
        }
