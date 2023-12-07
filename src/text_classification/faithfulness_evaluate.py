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


def get_prediction_probabilities(model, test_data: pd.DataFrame, predictions: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return probabilities of predictions and prediction classes (int)

    @param model: model to use for prediction
    @param test_data: data to test the model
    @param predictions: if predictions are supplied, it will return the probabilities of these predictions,
        otherwise it will return the predictions with the highest probability
    @return: probabilities of predicted class and the predicted class (or input predictions)
    """
    probs = model.predict_proba(test_data)
    if predictions is None:
        predictions = np.argmax(probs, axis=1)
    # predictions = model.predict(test_data)
    # assert np.array_equal(predictions, probs_pred), "NO"
    probs_pred = np.array([probs[i, p] for i, p in enumerate(predictions)])
    return probs_pred, predictions


def compute_faith_metrics(test_data: pd.DataFrame, model, shap_values_abs, base_probs: np.ndarray,
                          original_predictions: np.ndarray,
                          feature_neutral: np.ndarray, top_k: int) -> tuple[float, float]:
    shap_top_k = np.argpartition(shap_values_abs, -top_k, axis=1)[:, -top_k:]  # (samples, top_k)
    shap_not_top_k = _complementary_indices(shap_values_abs, shap_top_k)

    test_data_suff = _replace_with_column_average(test_data.copy().to_numpy(), shap_top_k, feature_neutral)
    test_data_comp = _replace_with_column_average(test_data.copy().to_numpy(), shap_not_top_k, feature_neutral)

    probs_suff, _ = get_prediction_probabilities(model, test_data_suff, original_predictions)
    probs_comp, _ = get_prediction_probabilities(model, test_data_comp, original_predictions)

    # ABS val ?
    suff = np.abs(base_probs - probs_suff)  # keep only top k
    comp = np.abs(base_probs - probs_comp)  # remove top-k
    return comp, suff


def evaluation_faith(test_data: pd.DataFrame, model, feature_neutral: np.ndarray, q: list[int] = None) -> dict:
    if q is None:
        q = [1, 5, 10, 20]

    probs, predictions = get_prediction_probabilities(model, test_data)

    explainer = shap.TreeExplainer(model)

    shap_values: np.ndarray = explainer.shap_values(test_data)

    shap_values_abs = np.abs(shap_values)
    # shap_values_signs = np.where(shap_values < 0, -1, 1)
    # shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
    # shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change, columns=test_data.columns, index=test_data.index)

    metrics = defaultdict(list)
    for k in q:
        comp, suff = compute_faith_metrics(test_data, model, shap_values_abs, probs, predictions, feature_neutral, k)
        metrics["comp"].append(float(np.mean(comp)))
        metrics["suff"].append(float(np.mean(suff)))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


def main():
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, exclude_features=EXCLUDE_LIST)
    data_train.pop("y")
    y_true_test = data_test.pop("y")
    # data_test = data_test.iloc[:20, :]

    # Feature neutral value
    # noise = np.ones((data_test.shape[1],), dtype=float) * 0 # np.mean(data_train.to_numpy(), axis=0)
    noise = np.median(data_train.to_numpy(), axis=0)
    # noise = np.random.normal(loc=0.5, scale=0.1, size=data_test.shape)
    # Scale the noise to [0, 1]
    # noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)

    metrics = evaluation_faith(data_test, clf, noise, q=[1, 5, 10])
    pprint(metrics)


if __name__ == "__main__":
    main()
