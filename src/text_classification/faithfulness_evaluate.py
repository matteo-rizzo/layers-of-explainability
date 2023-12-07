from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
import shap

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.easy_classifier import EXCLUDE_LIST
from src.text_classification.utils import load_encode_dataset, _replace_with_column_average, _complementary_indices

DATASET: AbcDataset = CallMeSexistDataset()
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_CMS_FINAL_RFE.pkl"


def compute_faith_metrics(test_data: pd.DataFrame, model, shap_values_abs, base_probs: np.ndarray, feature_neutral: np.ndarray, top_k: int) -> dict[str, tuple[float, float]]:
    shap_top_k = np.argpartition(shap_values_abs, -top_k, axis=1)[:, -top_k:]  # (samples, top_k)
    shap_not_top_k = _complementary_indices(shap_values_abs, shap_top_k)

    test_data_suff = _replace_with_column_average(test_data.copy().to_numpy(), shap_top_k, feature_neutral)
    test_data_comp = _replace_with_column_average(test_data.copy().to_numpy(), shap_not_top_k, feature_neutral)

    probs_suff = model.predict_proba(test_data_suff)[:, 1]
    probs_comp = model.predict_proba(test_data_comp)[:, 1]

    suff = base_probs - probs_suff
    comp = base_probs - probs_comp
    return comp, suff


def evaluation_faith(test_data: pd.DataFrame, model, feature_neutral: np.ndarray, q: list[int] = None) -> dict:
    if q is None:
        q = [1, 5, 10, 20]

    probs = model.predict_proba(test_data)[:, 1]

    explainer = shap.TreeExplainer(model)

    shap_values: np.ndarray = explainer.shap_values(test_data)

    shap_values_abs = np.abs(shap_values)
    # shap_values_signs = np.where(shap_values < 0, -1, 1)
    # shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
    # shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change, columns=test_data.columns, index=test_data.index)

    metrics = defaultdict(list)
    for k in q:
        comp, suff = compute_faith_metrics(test_data, model, shap_values_abs, probs, feature_neutral, k)
        metrics["comp"].append(comp)
        metrics["suff"].append(suff)

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


def main():
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, exclude_features=EXCLUDE_LIST)
    data_train.pop("y")
    y_true_test = data_test.pop("y")
    # data_test = data_test.iloc[:20, :]

    # Feature neutral value
    # FIXME: consider using a sampling distribution
    feature_neutral = np.ones((data_test.shape[1],), dtype=float) * -1  # np.mean(data_train.to_numpy(), axis=0)

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)

    metrics = evaluation_faith(data_test, clf, feature_neutral, q=[1, 5, 10])
    pprint(metrics)


if __name__ == "__main__":
    main()
