from collections import defaultdict

from src.features_extraction.classes.Feature import Feature
from src.text_classification.external.nlpkit import TextStatsExtractor


class TextStatistics(Feature):
    def __init__(self, *args, **kwargs):
        self.extractor = TextStatsExtractor()

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        ext_data = self.extractor.fit_transform(texts)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in ext_data for aspect, score in
         label_dict.items()]
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        return {
            f"{cls.__name__}_chars": "character count",
            f"{cls.__name__}_words": "number of words",
            f"{cls.__name__}_sents": "number of sentences in the text",
            f"{cls.__name__}_avg_sent_length": "average sentence length (in words)",
            f"{cls.__name__}_avg_syllables_per_word": "average number of syllables per word",
            f"{cls.__name__}_avg_letters_per_word": "average number of letters per word",
            f"{cls.__name__}_flesch": "Flesch–Kincaid readability score (0=extremely difficult, 1=extremely easy)",
            f"{cls.__name__}_automated_readability": "Automated Readability Index score (0=easy, 1=difficult (college student))",
            f"{cls.__name__}_dale_chall": "Dale–Chall readability score (0=easy, 1=difficult (college student))",
            f"{cls.__name__}_lix": "Lix readability score (0=easy, 1=difficult)",
            f"{cls.__name__}_coleman_liau": "Coleman–Liau readability score (0=easy, 1=difficult)",
            f"{cls.__name__}_difficult_words": "number of difficult words (not in the Dale-Chall list of easy words)"
        }
