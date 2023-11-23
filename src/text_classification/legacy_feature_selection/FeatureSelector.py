import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, mutual_info_classif


class FeatureSelector:

    def __init__(self):
        pass

    @staticmethod
    def variance_threshold(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        selector = VarianceThreshold(threshold)
        selection = selector.fit_transform(data)
        print(selector.get_support())
        return pd.DataFrame(selection, columns=selector.get_support(indices=True))

    @staticmethod
    def k_best(data: pd.DataFrame, labels: pd.DataFrame, k: int = 20) -> pd.DataFrame:
        score_fun = [mutual_info_classif, f_classif]
        selector = SelectKBest(score_fun[1], k=k)
        selection = selector.fit_transform(data, labels)
        print(selector.get_support())
        return pd.DataFrame(selection, columns=selector.get_support(indices=True))
