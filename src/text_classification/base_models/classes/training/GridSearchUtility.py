from __future__ import annotations

import copy
from pprint import pprint
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV, train_test_split

from src.text_classification.base_models.classes.training.BaseUtility import BaseUtility


class GridSearchUtility(BaseUtility):
    def __init__(self, configuration_parameters: dict, base_classifier_type: type,
                 base_classifier_kwargs: dict | None = None):
        super().__init__(configuration_parameters, base_classifier_type, base_classifier_kwargs)

    def grid_search_best_params(self, training_data: pd.DataFrame,
                                compute_metrics_fn: Callable[[Any, Any], dict[str, float]]) -> ClassifierMixin:
        """
        Perform grid search of the selected classifier (see 'SK_CLASSIFIER_TYPE').

        @param training_data: dataframe with training data. Target should be in 'y' column;
        @param compute_metrics_fn: callable that accepts predictions and targets and return metrics
        """
        # Load configuration
        num_rand_states: int = self.train_config["grid_search_params"]["num_seeds"]
        test_size: float = self.train_config["grid_search_params"]["test_size"]

        # Initiate training
        avg_metrics: dict[str, list] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        best_params = list()

        # Per seed training
        for rs in range(num_rand_states):
            # Prepare splits
            train_df, val_df = train_test_split(training_data, random_state=0, shuffle=True,
                                                stratify=training_data["y"].tolist(), test_size=test_size)

            y_train = train_df.pop("y")
            y_val = val_df.pop("y")

            # Setup and train classifier
            params = self.train_config["grid_search_params"][self._base_classifier_type.__name__]
            base_estimator_params = self.train_config[self._base_classifier_type.__name__]

            gs = GridSearchCV(self._base_classifier_type(**self._base_classifier_kwargs), param_grid=params, verbose=10,
                              refit=True, n_jobs=8, cv=5)
            # Set base parameters in the estimator inside GS
            gs.estimator.set_params(**base_estimator_params)

            # Preprocess using hooks
            train_df = self.preprocess_x_data(train_df)
            val_df = self.preprocess_x_data(val_df)
            y_train = self.preprocess_y_data(y_train)
            y_val = self.preprocess_y_data(y_val)

            gs.fit(train_df, y_train)
            y_pred = gs.predict(val_df).tolist()

            # Calculate metrics
            metrics = compute_metrics_fn(y_true=y_val, y_pred=y_pred)

            # Print results
            print(f"Random Seed {rs} - Validation Metrics:")
            for metric, value in metrics.items():
                print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

            avg_metrics["accuracy"].append(metrics["accuracy"])
            avg_metrics["precision"].append(metrics["precision"])
            avg_metrics["recall"].append(metrics["recall"])
            avg_metrics["f1"].append(metrics["f1"])

            params = {
                "gs_params": copy.deepcopy(gs.best_params_),
                "all_params": gs.get_params(True)
            }

            best_params.append(params)

        print("-----------------------------------------------------------")
        print(f"Average Validation Metrics Over {num_rand_states} Random Seeds:")
        for metric, value in avg_metrics.items():
            print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

        print("-----------------------------------------------------------")
        pprint(best_params)

        return gs.best_estimator_
