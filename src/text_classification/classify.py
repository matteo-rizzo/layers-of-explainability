""" Script to GS and fit a classifier on review dataset, to use as feature extractor """
from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Type

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.explainable_strategy.pipeline import make_pipeline
from src.text_classification.classes.FeatureSelector import FeatureSelector
from src.text_classification.main import compute_metrics
from src.utils.yaml_manager import load_yaml

sk_classifier_type = RandomForestClassifier


def naive_classifier(sk_classifier: ClassifierMixin, training_data: pd.DataFrame, params=dict()) -> np.ndarray | tuple[
    np.ndarray, Pipeline]:
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
        train_df, val_df = train_test_split(data, random_state=0, shuffle=True, stratify=data["label"],
                                            test_size=test_size)

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


def final_ensemble(train_config: dict, feature_file: Path | str, tfidf_clf: RandomForestClassifier,
                   all_data: pd.DataFrame) -> None:
    # Load feature vectors
    f_df = pd.read_csv(feature_file)

    # Add clf predictions
    # probs = tfidf_clf.predict_proba(all_data["text_"].tolist())
    #
    # feature_clf = {
    #     "score_0": probs[:, 0].tolist(),
    #     "score_1": probs[:, 1].tolist()
    # }
    #
    # f_df.update(feature_clf)

    train_df, test_df, train_y, test_y = train_test_split(f_df, all_data["label"], random_state=31, shuffle=True,
                                                          stratify=all_data["label"], test_size=0.25)

    clf = DecisionTreeClassifier(**train_config["DecisionTreeClassifier"])

    clf.fit(train_df, y=train_y.tolist())

    y_pred = clf.predict(test_df).tolist()

    print("Metrics AFTER ensemble")
    compute_metrics(y_pred, test_y.tolist(), sk_classifier_name="Final ENSEMBLE")


def all_classifier(train_config, training_data: pd.DataFrame, testing_data: pd.DataFrame):
    base_clf = DecisionTreeClassifier(**train_config["DecisionTreeClassifier"])
    clf = AdaBoostClassifier(n_estimators=5000, random_state=11, estimator=base_clf)

    y_train = training_data.pop("y")
    y_test = testing_data.pop("y")

    clf.fit(training_data, y=y_train.tolist())

    y_pred = clf.predict(testing_data).tolist()

    print("Metrics AFTER AdaBoost")
    compute_metrics(y_pred, y_test.tolist(), sk_classifier_name="Final ENSEMBLE")


def main():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    data_path_train = Path("dataset") / "ami2018_misogyny_detection" / "AMI2018Dataset_train_features.csv"
    data_path_test = Path("dataset") / "ami2018_misogyny_detection" / "AMI2018Dataset_test_features.csv"

    scaler = StandardScaler()

    data_train = pd.read_csv(data_path_train)
    train_labels = pd.read_csv(data_path_train.parent / "en_training_anon.tsv", sep="\t")["misogynous"]

    data_test = pd.read_csv(data_path_test)
    test_labels = pd.read_csv(data_path_test.parent / "en_testing_labeled_anon.tsv", sep="\t")["misogynous"]

    features = data_train.columns.values.tolist()

    # sel_data = FeatureSelector.variance_threshold(data_train, threshold=0.8)
    sel_data = FeatureSelector.k_best(data_train, train_labels)
    selected_features = sel_data.columns.values.tolist()
    data_train = data_train.iloc[:, selected_features]
    data_test = data_test.iloc[:, selected_features]

    data_train = scaler.fit_transform(data_train)
    data_train = pd.DataFrame(data_train, columns=selected_features)

    data_test = scaler.transform(data_test)
    data_test = pd.DataFrame(data_test, columns=selected_features)

    data_train["y"] = train_labels
    data_test["y"] = test_labels

    # train_df, test_df = train_test_split(data, random_state=31, shuffle=True, stratify=data["label"], test_size=0.25)
    #
    # # grid_search_best_params(RandomForestClassifier, train_df, train_config)

    # Setup and train classifier
    # _, clf = naive_classifier(RandomForestClassifier(**train_config["RandomForestClassifier"], n_jobs=-1), train_df)
    #
    # y_pred = clf.predict(test_df["text_"]).tolist()
    #
    # joblib.dump(clf, "dumps/text_classification/clf_pipeline.pkl", compress=1)
    #
    # # Calculate metrics
    # print("Metrics before ensemble")
    # compute_metrics(y_pred, test_df["label"], sk_classifier_name="Final RF")

    all_classifier(train_config, data_train, data_test)

    # final_ensemble(train_config, "dataset/fake_reviews_features.csv", None, data)


if __name__ == "__main__":
    main()
