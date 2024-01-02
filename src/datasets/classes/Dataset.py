from __future__ import annotations

import abc
from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class AbcDataset(abc.ABC):
    BASE_DATASET = "dataset"

    def __init__(self, target: str, validation: float = .0):
        self._target = target
        self._validation = validation
        self._split_data = None

    @property
    def target(self) -> str:
        return self._target

    def get_test_data(self) -> np.ndarray | list:
        return self._split_data["test"]["x"]

    def get_train_data(self) -> np.ndarray | list:
        return self._split_data["train"]["x"]

    def get_train_labels(self) -> np.ndarray | list:
        return self._split_data["train"]["y"]

    def get_test_labels(self) -> np.ndarray | list:
        return self._split_data["test"]["y"]

    def get_train_val_test_split(self) -> dict:
        return self._split_data

    @staticmethod
    @abc.abstractmethod
    def get_path_to_testset() -> str:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_path_to_trainset() -> str:
        pass

    @staticmethod
    def preprocessing(text_string: str) -> str:
        return text_string.strip()

    @abc.abstractmethod
    def _train_val_test(self) -> Dict:
        pass

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, sk_classifier_name: str = None) -> dict:
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", pos_label=1)
        acc = accuracy_score(y_true, y_pred)
        if sk_classifier_name:
            print(f"{sk_classifier_name} accuracy: {acc:.3f}")
            print(f"{sk_classifier_name} precision: {precision:.3f}")
            print(f"{sk_classifier_name} recall: {recall:.3f}")
            print(f"{sk_classifier_name} F1-score: {f1_score:.3f}")

        return {"f1": f1_score, "accuracy": acc, "precision": precision, "recall": recall}
