from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset


def load_encode_dataset(dataset: AbcDataset, scale: bool = False, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset composed of extracted features based on the 'DATASET' global params that must be set to an AbcDataset object.

    @return: two dataframes with training and test data; target label is set to class 'y' and is encoded with numbers in 0...n-1
    """
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
    if scale:
        feature_names = data_train.columns.tolist()

        scaler = StandardScaler()
        data_train = pd.DataFrame(scaler.fit_transform(data_train))
        data_train.columns = feature_names

        data_test = pd.DataFrame(scaler.transform(data_test))
        data_test.columns = feature_names

    data_train["y"] = y_train
    data_test["y"] = y_test
    return data_train, data_test
