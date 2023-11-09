from collections import defaultdict

from empath import Empath

from src.text_classification.classes.features.Feature import Feature


class EmpathFeatures(Feature):
    def __init__(self, *args, **kwargs):
        self.lexicon = Empath()

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        processed_texts: list[dict[str, float]] = list()
        default_score = self.lexicon.analyze("Hi", normalize=True)
        default_score = {k: .0 for k in default_score}
        for t in texts:
            # [4233, 4776]
            data = self.lexicon.analyze(t, normalize=True)
            processed_texts.append(data if data is not None else default_score)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in processed_texts for aspect, score in label_dict.items()]
        return feature_df
