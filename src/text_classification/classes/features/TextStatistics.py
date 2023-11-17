from collections import defaultdict

from src.text_classification.classes.features.Feature import Feature
from src.text_classification.external.nlpkit import TextStatsExtractor


class TextStatistics(Feature):
    def __init__(self, *args, **kwargs):
        self.extractor = TextStatsExtractor()

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        ext_data = self.extractor.fit_transform(texts)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in ext_data for aspect, score in label_dict.items()]
        return feature_df