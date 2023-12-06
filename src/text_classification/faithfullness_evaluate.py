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
from src.text_classification.utils import load_encode_dataset

DATASET: AbcDataset = CallMeSexistDataset()
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_CMS_FINAL.pkl"


def replace_with_column_average(arr, col_indices):
    # Calculate the column averages
    col_avgs = np.mean(arr[:, col_indices], axis=0)

    # Replace the values in the specified columns with the column averages
    arr[:, col_indices] = col_avgs

    return arr


def faith_metrics(test_data: pd.DataFrame, model, shap_values_abs, base_probs: np.ndarray, top_k: int):
    shap_top_k = np.argpartition(shap_values_abs, -top_k, axis=1)[:, -top_k:]
    shap_not_top_k = set(range(test_data.shape[1])) - set(shap_top_k.tolist())

    test_data_suff = replace_with_column_average(test_data.copy().to_numpy(), shap_top_k)
    test_data_comp = replace_with_column_average(test_data.copy().to_numpy(), shap_top_k)

    probs_suff = model.predict_proba(test_data_suff)
    probs_comp = model.predict_proba(test_data_comp)

    suff = base_probs - probs_suff
    comp = base_probs - probs_comp
    return comp, suff


def evaluate_faith(test_data: pd.DataFrame, model) -> dict:
    probs = model.predict_proba(test_data)[:, 1]

    explainer = shap.TreeExplainer(model)

    shap_values: np.ndarray = explainer.shap_values(test_data)

    shap_values_abs = np.abs(shap_values)
    # shap_values_signs = np.where(shap_values < 0, -1, 1)
    # shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
    # shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change, columns=test_data.columns, index=test_data.index)

    metrics = defaultdict(list)
    for k in [1, 5, 10]:
        comp, suff = faith_metrics(test_data, model, shap_values_abs, probs, k)
        metrics["comp"].append(comp)
        metrics["suff"].append(suff)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def faith_eval():
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, exclude_features=EXCLUDE_LIST)
    data_train.pop("y")
    y_true_test = data_test.pop("y")

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)

    metrics = evaluate_faith(data_test, clf)
    pprint(metrics)
