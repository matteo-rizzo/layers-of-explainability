from __future__ import annotations

from pprint import pprint

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.TMP_simple_model.pipeline import make_pipeline
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()
CLASSIFIER = XGBClassifier  # XGBClassifier  # RidgeClassifier()


def grid_search_best_params(sk_classifier_type: type):
    # Load configuration
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    num_rand_states: int = train_config["grid_search_params"]["num_seeds"]
    test_size: float = train_config["grid_search_params"]["test_size"]

    # Initiate training
    avg_metrics: dict[str, list] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    best_params = list()

    # Prepare splits
    data = DATASET._split_data
    train_x, train_y = data["train"]["x"], data["train"]["y"]

    pipeline = make_pipeline(sk_classifier_type())

    # Setup and train classifier
    params = train_config["grid_search_params"][sk_classifier_type.__name__]

    # Per seed training
    for seed in range(num_rand_states):
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=seed)

        # Define the grid search

        param_grid = {f'classifier__{k}': v for k, v in params.items()}
        param_grid |= {'vectorizer__ngram_range': [(1, 3), (1, 2)], "vectorizer__max_features": [10000, 8000]}
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=None, scoring="f1_macro", refit=True)

        # Fit the grid search
        print("GRIDDING")
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params.append(grid_search.best_params_)

        # Evaluate the model
        y_pred = grid_search.predict(X_val)
        avg_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        avg_metrics["precision"].append(precision_score(y_val, y_pred))
        avg_metrics["recall"].append(recall_score(y_val, y_pred))
        avg_metrics["f1"].append(f1_score(y_val, y_pred))

    print("-----------------------------------------------------------")
    print(f"Average Validation Metrics Over {num_rand_states} Random Seeds:")
    for metric, value in avg_metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

    print("-----------------------------------------------------------")
    pprint(best_params)

    return avg_metrics, best_params


if __name__ == "__main__":
    print("*** GRID SEARCH ")
    grid_search_best_params(CLASSIFIER)
