from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

import lightgbm as lgb
import pandas as pd
from sklearn.base import ClassifierMixin

from src.text_classification.base_models.classes.training.TrainingModelUtility import TrainingModelUtility


class LGBMTrainingModelUtility(TrainingModelUtility):

    def train_classifier(self, training_data: pd.DataFrame) -> ClassifierMixin:
        """
        Train a scikit-learn classifier on training data and return the fitted object.
        """

        y_train = training_data.pop("y")

        training_set = lgb.Dataset(training_data, label=y_train)
        train_bin = Path("dumps/lgbm/train.bin")
        train_bin.mkdir(parents=True, exist_ok=True)
        training_set.save_binary(train_bin)
        validation_data = training_set.create_valid(train_bin)

        clf = self._base_classifier_type(**self._base_classifier_kwargs,
                                         **self.train_config[self._base_classifier_type.__name__],
                                         eval_metric="accuracy",
                                         eval_set=[validation_data])

        # num_round = 10
        # clf = lgb.train(self._base_classifier_kwargs, training_set, num_round,
        #                 feval=[],
        #                 valid_sets=[validation_data],
        #                 callbacks=[lgb.early_stopping(5)])
        self.trained_classifier = clf

        return clf

    def evaluate(self, testing_data: pd.DataFrame, compute_metrics_fn: Callable[[Any, Any, *Any], dict[str, float]],
                 print_metrics: bool = True, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], list[int]]:
        """
        Evaluate dataset metrics of a scikit-learn classifier on testing data.
        """

        if self.trained_classifier is None:
            raise ValueError("Cannot evaluate before training a classifier")

        y_test = testing_data.pop("y")

        y_pred = self.trained_classifier.predict(testing_data, num_iteration=self.trained_classifier.best_iteration_).tolist()

        metrics = compute_metrics_fn(y_true=y_test, y_pred=y_pred)

        if print_metrics:
            print("Classification metrics on test data")
            for k, v in metrics.items():
                print(f"{self.trained_classifier.__class__.__name__} {k}: {v:.3f}")
        if return_predictions:
            return metrics, y_pred
        return metrics
