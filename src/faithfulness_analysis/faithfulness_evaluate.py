from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
import shap

from src.datasets.classes.Dataset import AbcDataset
from src.datasets.classes.IMDBDataset import IMDBDataset
from src.text_classification.utils import load_encode_dataset, get_excluded_features_dataset

DATASET: AbcDataset = IMDBDataset()
suffix_str = 'CMS' if DATASET.__class__.__name__ == 'CallMeSexistDataset' else 'IMDB'
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / f"model_{suffix_str}_FINAL_RFE.pkl"


def get_prediction_probabilities(model, test_data: pd.DataFrame, predictions: np.ndarray = None) -> tuple[
    np.ndarray, np.ndarray]:
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
    probs_pred = np.array([probs[i, p] for i, p in enumerate(predictions)])
    return probs_pred, predictions


def compute_faith_metrics(test_data: pd.DataFrame, model, shap_values_abs, base_probs: np.ndarray,
                          original_predictions: np.ndarray,
                          feature_neutral: np.ndarray, k_perc: float) -> tuple[float, float]:
    test_data_comp = test_data.copy()
    test_data_suff = test_data.copy()
    for i in range(test_data.shape[0]):
        # Find contributing features in current examples (positive shap values)
        positive_features = np.where(shap_values_abs[i, :] > 0, 1, 0)
        positive_features_idx = np.asarray(positive_features == 1).nonzero()[0]
        # Take the percentage of positive features
        k = math.ceil(k_perc * len(positive_features_idx))
        # Find top-k contributing features (k < len(positive_feature_idx), so it can't remove negative ones)
        shap_top_k = np.argpartition(shap_values_abs[i, :], -k)[-k:]
        # Find complementary features (among the contributing ones)
        shap_not_top_k = np.setdiff1d(positive_features_idx, shap_top_k)

        # SUFF = keeps only the important tokens and calculates the change in output probability compared to the original predicted class
        test_data_suff.iloc[i, shap_not_top_k] = feature_neutral[shap_not_top_k]

        # COMP = change in the output probability of the predicted class after the important (top-k) tokens are removed
        test_data_comp.iloc[i, shap_top_k] = feature_neutral[shap_top_k]

    probs_suff, _ = get_prediction_probabilities(model, test_data_suff, original_predictions)
    probs_comp, _ = get_prediction_probabilities(model, test_data_comp, original_predictions)

    # ABS value? No, the paper accepts negative values
    # suff = np.abs(base_probs - probs_suff) # keep only top k
    suff = base_probs - probs_suff
    # comp = np.abs(base_probs - probs_comp)  # remove top-k
    comp = base_probs - probs_comp  # remove top-k
    return comp, suff


def evaluation_faith(test_data: pd.DataFrame, model, feature_neutral: np.ndarray, q_perc: list[int] = None) -> dict:
    if q_perc is None:
        q_perc = [1, 5, 10, 20]
    q_perc: list[float] = [q / 100 for q in q_perc]

    probs, predictions = get_prediction_probabilities(model, test_data)

    explainer = shap.TreeExplainer(model)

    shap_values: np.ndarray = explainer.shap_values(test_data)

    # We want to isolate contributions for the predicted class. If class is 0, contributed values have negative shap.
    signs = np.where(predictions > 0, 1.0, -1.0).reshape(-1, 1)  # (n_samples, 1)
    # Features with positive values will be the ones contributing to the predicted class
    shap_values_signed = shap_values * signs
    # Avoid removing features that contributed to the opposite of prediction
    # shap_values_signed[shap_values_signed < 0] = .0

    metrics = defaultdict(list)
    for k_perc in q_perc:
        comp, suff = compute_faith_metrics(test_data, model, shap_values_signed, probs, predictions, feature_neutral,
                                           k_perc)
        metrics["comp"].append(comp)
        metrics["suff"].append(suff)

    metrics = {k: np.stack(v, axis=-1) for k, v in metrics.items()}  # k: (samples, q)

    # Write runs to numpy files
    out_path = Path("dumps") / "faithfulness"
    out_path.mkdir(parents=True, exist_ok=True)
    for m, v in metrics.items():
        np.save(out_path / f"{m}_xg_{suffix_str}.npy", v)

    # Average of metrics, with average of per-sample STD over all k
    return {k: (float(v.mean()), float(v.std(axis=1).mean())) for k, v in metrics.items()}


def main():
    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)

    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True,
                                                exclude_features=get_excluded_features_dataset(DATASET, clf.__class__))
    data_train.pop("y")

    # Feature neutral value
    noise = np.median(data_train.to_numpy(), axis=0)

    metrics = evaluation_faith(data_test, clf, noise, q_perc=[1, 5, 10, 20, 50, 75])
    pprint(metrics)


if __name__ == "__main__":
    main()
