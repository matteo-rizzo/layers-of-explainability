from __future__ import annotations

from typing import Callable, Any

import pandas as pd
from sklearn.base import ClassifierMixin

from src.text_classification.base_models.classes.training.BaseUtility import BaseUtility


class TrainingModelUtility(BaseUtility):
    def __init__(self, configuration_parameters: dict, base_classifier_type: type,
                 base_classifier_kwargs: dict | None = None):
        super().__init__(configuration_parameters, base_classifier_type, base_classifier_kwargs)
        self.trained_classifier: ClassifierMixin | None = None

    def train_classifier(self, training_data: pd.DataFrame) -> ClassifierMixin:
        """
        Train a scikit-learn classifier on training data and return the fitted object.
        """
        clf = self._base_classifier_type(**self._base_classifier_kwargs,
                                         **self.train_config[self._base_classifier_type.__name__])

        y_train = training_data.pop("y")

        training_data = self.preprocess_x_data(training_data)
        y_train = self.preprocess_y_data(y_train)

        clf.fit(training_data, y_train)

        self.trained_classifier = clf

        return clf

    def evaluate(self, testing_data: pd.DataFrame, compute_metrics_fn: Callable[[Any, Any, *Any], dict[str, float]],
                 print_metrics: bool = True, return_predictions: bool = False) -> dict[str, float] | tuple[
        dict[str, float], list[int]]:
        """
        Evaluate dataset metrics of a scikit-learn classifier on testing data.
        """

        if self.trained_classifier is None:
            raise ValueError("Cannot evaluate before training a classifier")

        y_test = testing_data.pop("y")

        testing_data = self.preprocess_x_data(testing_data)
        y_pred = self.trained_classifier.predict(testing_data).tolist()

        metrics = compute_metrics_fn(y_true=y_test, y_pred=y_pred)

        if print_metrics:
            print("Classification metrics on test data")
            for k, v in metrics.items():
                print(f"{self.trained_classifier.__class__.__name__} {k}: {v:.3f}")
        if return_predictions:
            return metrics, y_pred
        return metrics
