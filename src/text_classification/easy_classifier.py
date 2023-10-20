""" Script to GS and fit a classifier on review dataset, to use as feature extractor """

from pathlib import Path
from pprint import pprint
from typing import Type

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.explainable_strategy.pipeline import make_pipeline
from src.text_classification.main import compute_metrics
from src.utils.yaml_manager import load_yaml

sk_classifier_type = RandomForestClassifier


def naive_classifier(sk_classifier: ClassifierMixin, training_data: pd.DataFrame, params=dict()) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    pipe = make_pipeline(sk_classifier)

    print("------ Training")

    pipe.fit(training_data["text_"], training_data["label"])

    predicted = None
    # if predict:
    #     print("------ Testing")
    #
    #     # Predicting with a test dataset
    #     predicted = pipe.predict(training_data["test"]["x"])
    #
    # if not return_pipe:
    #     return predicted
    # else:
    return predicted, pipe


def grid_search_best_params(sk_classifier_type: Type[ClassifierMixin], data: pd.DataFrame, train_config):
    # Load configuration
    num_rand_states: int = train_config["grid_search_params"]["num_seeds"]
    test_size: float = train_config["grid_search_params"]["test_size"]

    # Initiate training
    avg_metrics: dict[str, list] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    best_params = list()

    # Per seed training
    for rs in range(num_rand_states):
        # Prepare splits
        train_df, val_df = train_test_split(data, random_state=0, shuffle=True, stratify=data["label"], test_size=test_size)

        # Setup and train classifier
        params = train_config["grid_search_params"][sk_classifier_type.__name__]

        gs = GridSearchCV(sk_classifier_type(), param_grid=params, verbose=10, refit=True, n_jobs=-1, cv=5)
        grid_clf = make_pipeline(gs)

        grid_clf.fit(train_df["text_"], train_df["label"])
        y_pred = grid_clf.predict(val_df["text_"]).tolist()

        # Calculate metrics
        metrics = compute_metrics(y_pred, val_df["label"])

        # Print results
        print(f"Random Seed {rs} - Validation Metrics:")
        for metric, value in metrics.items():
            print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

        avg_metrics["accuracy"].append(metrics["accuracy"])
        avg_metrics["precision"].append(metrics["precision"])
        avg_metrics["recall"].append(metrics["recall"])
        avg_metrics["f1"].append(metrics["f1"])

        best_params.append(gs.best_params_)

    print("-----------------------------------------------------------")
    print(f"Average Validation Metrics Over {num_rand_states} Random Seeds:")
    for metric, value in avg_metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

    print("-----------------------------------------------------------")
    pprint(best_params)


def main():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    data_path = Path("dataset") / "fake_reviews_dataset.csv"

    data = pd.read_csv(data_path)

    train_df, test_df = train_test_split(data, random_state=31, shuffle=True, stratify=data["label"], test_size=0.25)

    # grid_search_best_params(RandomForestClassifier, train_df, train_config)

    # Setup and train classifier
    _, clf = naive_classifier(RandomForestClassifier(train_config["RandomForestClassifier"]), train_df)

    y_pred = clf.predict(test_df["text_"]).tolist()

    joblib.dump(clf, "dumps/text_classification/clf_pipeline.pkl", compress=1)

    # Calculate metrics
    compute_metrics(y_pred, test_df["label"], sk_classifier_name="Final RF")


if __name__ == "__main__":
    main()
