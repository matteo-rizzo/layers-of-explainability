from collections import defaultdict

from nltk import CoreNLPParser

from src.features_extraction.classes.Feature import Feature
from src.text_classification.external.nlpkit import CFGExtractor


class CFGFeatures(Feature):
    def __init__(self, *args, **kwargs):
        self._sf_parser = CoreNLPParser(url="http://localhost:9000/")
        self.extractor = CFGExtractor(self._sf_parser)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        ext_data = self.extractor.fit_transform(texts[:10])
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in ext_data for aspect, score in
         label_dict.items()]
        return feature_df
