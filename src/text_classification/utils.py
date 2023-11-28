from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset


def load_encode_dataset(dataset: AbcDataset, std_scale: bool = False, max_scale: bool = False, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset composed of extracted features based on the 'DATASET' global params that must be set to an AbcDataset object.

    @return: two dataframes with training and test data; target label is set to class 'y' and is encoded with numbers in 0...n-1
    """
    assert not (std_scale and max_scale), "You can't use standard scaler and Max scaler together, only (at most) one operation should be done."

    data_path_train = Path(dataset.BASE_DATASET) / f"{dataset.__class__.__name__}_train_features.csv"
    data_path_test = Path(dataset.BASE_DATASET) / f"{dataset.__class__.__name__}_test_features.csv"

    data_train = pd.read_csv(data_path_train, usecols=features)
    data_test = pd.read_csv(data_path_test, usecols=features)

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
