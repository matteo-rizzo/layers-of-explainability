from collections import defaultdict
from pathlib import Path

from src.text_classification.classes.features.Feature import Feature
from src.text_classification.external.nlpkit import LIWCExtractor
from src.text_classification.external.nlpkit.liwc import Liwc


class LIWCFeatures(Feature):
    LIWC_FILEPATH = Path("src") / "text_classification" / "external" / "nlpkit" / "liwc" / "resources" / "liwc.dic"

    def __init__(self, *args, **kwargs):
        liwc = Liwc(LIWCFeatures.LIWC_FILEPATH)
        self.extractor = LIWCExtractor(liwc)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        # [4233, 4776]
        ext_data: list[dict] = self.extractor.fit_transform(texts)
        # Fill missing categories when not all aspects are returned
        all_categories = {k: .0 for k in self.extractor.liwc.categories.values()}
        for d in ext_data:
            d.update(all_categories)
        # ext_data = [d.update(all_categories) for d in ext_data]
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in ext_data for aspect, score in label_dict.items()]
        return feature_df
