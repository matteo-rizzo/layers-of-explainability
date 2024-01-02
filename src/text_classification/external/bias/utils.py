from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datasets.classes.Dataset import AbcDataset
from src.text_classification.external.bias.bias_madlibs import _read_word_list
from src.text_classification.external.bias.model_bias_analysis import add_subgroup_columns_from_text, \
    compute_bias_metrics_for_model, SUBGROUP_AUC, NEGATIVE_CROSS_AUC, \
    POSITIVE_CROSS_AUC, POSITIVE_AEG, NEGATIVE_AEG


def get_final_metric(bias_df, overall_auc_test, model_name):
    bias_score = float(np.average([
        bias_df[model_name + "_" + SUBGROUP_AUC],
        bias_df[model_name + "_" + NEGATIVE_CROSS_AUC],
        bias_df[model_name + "_" + POSITIVE_CROSS_AUC]
    ]))

    p_aeg = float(bias_df[model_name + "_" + POSITIVE_AEG].mean())
    n_aeg = float(bias_df[model_name + "_" + NEGATIVE_AEG].mean())

    print(f"Bias Score = {bias_score:.5f}")
    print(f"Negative AEG = {n_aeg:.5f}")
    print(f"Positive AEG = {p_aeg:.5f}")
    print(f"AUC Score = {overall_auc_test:.5f}")
    return float(np.mean([overall_auc_test, bias_score]))


def bias_computation(data: pd.DataFrame, y_pred_probs: list[float], bias_dataset: AbcDataset):
    # Generate DF with subgroups
    data["text"] = bias_dataset.get_train_data()
    base_dir = bias_dataset.BASE_DATASET / "words"

    jobs = _read_word_list(base_dir, 'occupations.txt')
    groups = _read_word_list(base_dir, 'group_people.txt')
    adjectives_people = _read_word_list(base_dir, 'adjectives_people.txt')
    subgroups: list[str] = jobs + groups + adjectives_people

    mapping_ = dict(zip(groups, adjectives_people))

    add_subgroup_columns_from_text(data, text_column="text", subgroups=subgroups, expect_spaces_around_words=False)

    for k, v in mapping_.items():
        data[v] |= data[k]

    # FOR NOW ONLY PEOPLE
    subgroups = adjectives_people + jobs

    data = data.loc[:, subgroups]

    # ADD PREDICTIONS
    model_name = "sexist_pred"
    data[model_name] = y_pred_probs
    data["y"] = bias_dataset.get_test_labels()

    bias_metrics = compute_bias_metrics_for_model(data, subgroups, model_name, "y")

    # Overall AUC
    overall_auc = roc_auc_score(data["y"], data[model_name])

    ami_bias_value = get_final_metric(bias_metrics, overall_auc, model_name=model_name)
    print(f"AMI bias = {ami_bias_value:.5f} (higher is better)")
