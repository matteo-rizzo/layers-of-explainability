from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from src.datasets.classes.CGReviewDataset import CGReviewDataset
from src.datasets.classes.Dataset import AbcDataset


def load_encode_dataset(dataset: AbcDataset, std_scale: bool = False, max_scale: bool = False,
                        features: list[str] | None = None, exclude_features: list[str] | None = None) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Load dataset composed of extracted features based on the 'DATASET' global params that must be set to an AbcDataset object.

    @return: two dataframes with training and test data; target label is set to class 'y' and is encoded with numbers in 0...n-1
    """
    assert not (
            std_scale and max_scale), "You can't use standard scaler and Max scaler together, only (at most) one operation should be done."

    data_path_train = Path(dataset.BASE_DATASET) / f"{dataset.__class__.__name__}_train_features.csv"
    data_path_test = Path(dataset.BASE_DATASET) / f"{dataset.__class__.__name__}_test_features.csv"

    data_train = pd.read_csv(data_path_train, usecols=features)
    data_test = pd.read_csv(data_path_test, usecols=features)

    if exclude_features is not None:
        data_train = data_train.drop(columns=exclude_features)
        data_test = data_test.drop(columns=exclude_features)

    if dataset.__class__ == CGReviewDataset:
        print("Removed CHATGpt detector features from the CGReviewDataset")
        data_train.pop("ChatGPTDetector_Human")
        data_train.pop("ChatGPTDetector_ChatGPT")
        data_test.pop("ChatGPTDetector_Human")
        data_test.pop("ChatGPTDetector_ChatGPT")

    train_labels = dataset.get_train_labels()
    test_labels = dataset.get_test_labels()

    # Binary encoding of labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.transform(test_labels)

    # Scaling/normalization
    if std_scale or max_scale:
        feature_names = data_train.columns.tolist()

        if std_scale:
            transformation = StandardScaler()
        else:
            transformation = MinMaxScaler(clip=True)  # MaxAbsScaler: range [-1, 1], 0s are kept

        data_train = pd.DataFrame(transformation.fit_transform(data_train))
        data_train.columns = feature_names

        data_test = pd.DataFrame(transformation.transform(data_test))
        data_test.columns = feature_names

    data_train["y"] = y_train
    data_test["y"] = y_test
    return data_train, data_test


def capitalize_first_letter(s: str) -> str:
    """ Make first letter uppercase, leavening the rest of the string unchanged"""
    s_chars = list(s)
    return s_chars[0].upper() + ''.join(s_chars[1:])


def bin_column(values: pd.Series, labels: list[str]) -> tuple[pd.Series, dict[str, float]]:
    """
    Utility to bin a single pandas series using, if possible, pd.qcut or alternatively pd.cut

    @param values: series to transform, will not be changed
    @param labels: list of interval labels in sorted order (low to high)
    @return: series with transformed values,
        and dictionary of bin edges used in the binning process ({bin label -> bin upper edge})
    """
    test_quantiles: list[float] = [values.quantile(x) for x in np.linspace(0, 1, len(labels) + 1).tolist()]
    has_duplicates: bool = len(np.unique(test_quantiles)) != len(test_quantiles)
    if has_duplicates:
        # Can't do partition by quantiles so just use linear partition
        t_values, bins = pd.cut(values, bins=len(labels), labels=labels, include_lowest=True, duplicates="raise",
                                ordered=True, right=True, retbins=True)
    else:
        t_values, bins = pd.qcut(values, q=len(labels), labels=labels, duplicates="raise", retbins=True)
    bin_labels = dict(zip(labels, bins[1:].tolist()))  # exclude the lowest bound which is always 0
    return t_values, bin_labels


def quantize_features(data: pd.DataFrame, quantiles: dict[str, dict[str, float]] | None = None,
                      interval_labels: list[str] = None) -> tuple[pd.DataFrame, dict]:
    """
    Quantization of feature based on frequency intervals

    @param interval_labels: labels of each quantile. Used to determine the number of bins. Ignored if quantiles != None
    @param quantiles: dictionary of {"column" -> {"label" -> bin upper bound}}, or None to compute quantiles automatically
    @param data: dataframe to transform, a copy will be returned
    @return: transformed dataframe, bins/quantiles for each column as dict (same as input)
    """
    data = data.copy()
    if interval_labels is None:
        interval_labels = ["low", "mid", "high"]

    if quantiles is None:
        quantiles = dict()
        for col in data.columns:
            data[col], quantiles[col] = bin_column(data[col], interval_labels)
    else:
        for col in data.columns:
            data[col] = pd.cut(data[col], bins=[0, *quantiles[col].values()], labels=list(quantiles[col].keys()),
                               include_lowest=True, duplicates="raise", right=True,
                               retbins=False)

    return data, quantiles


def _replace_with_column_average(arr, col_indices, feature_neutral: np.ndarray):
    # Calculate the column averages
    # col_avgs = feature_neutral  # np.zeros(arr.shape[1])  # np.mean(arr, axis=0)

    # Replace the values in the specified columns with the column averages
    for i in range(arr.shape[0]):
        arr[i, col_indices[i, :]] = feature_neutral[col_indices[i, :]]

    return arr


def _complementary_indices(arr, top_k_indices):
    # Get all indices
    all_indices = np.arange(arr.shape[1])

    # Initialize an empty array to store the complementary indices
    comp_indices = np.empty((arr.shape[0], arr.shape[1] - top_k_indices.shape[1]), dtype=int)

    # For each row, find the complementary indices
    for i in range(arr.shape[0]):
        comp_indices[i] = np.setdiff1d(all_indices, top_k_indices[i])

    return comp_indices


def get_excluded_features_dataset(dataset: AbcDataset, sk_classifier_type: type) -> list[str] | None:
    exclude_list = None
    exclude_path = Path(dataset.BASE_DATASET) / f"RFE_{dataset.__class__.__name__}_{sk_classifier_type.__name__}.json"
    if exclude_path.exists():
        with open(exclude_path, mode="r", encoding="utf-8") as fe:
            exclude_list: list[str] = json.load(fe)["removed"]
    return exclude_list
