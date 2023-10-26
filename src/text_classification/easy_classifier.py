""" Script to GS and fit a classifier on review dataset, to use as feature extractor """
from __future__ import annotations

import pickle
from pathlib import Path
from pprint import pprint
from typing import Type

import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from torch import nn

from src.deep_learning_strategy.classes.AMI2018Dataset import AMI2018Dataset
from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.explainable_strategy.pipeline import make_pipeline
from src.text_classification.classes.MLP import MLP
from src.text_classification.main import compute_metrics
from src.utils.yaml_manager import load_yaml

sk_classifier_type = RandomForestClassifier
dataset: AbcDataset = CGReviewDataset()  # AMI2018Dataset()


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


def target_conversion(x: torch.Tensor):
    return np.round(torch.sigmoid(x).detach().numpy())


def neural_classifier(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    from skorch import NeuralNetBinaryClassifier
    from skorch.callbacks import EarlyStopping, Checkpoint, EpochScoring

    layers = [
        (512, 0.3, True, nn.ReLU()),
        (256, 0.4, True, nn.ReLU()),
        (128, 0.3, True, nn.ReLU()),
        (64, 0.4, True, nn.ReLU()),
        (32, 0.3, True, nn.ReLU()),
        (64, 0.4, True, nn.ReLU()),
        (128, 0.3, True, nn.ReLU()),
        (32, 0.4, True, nn.ReLU()),
        (16, 0.3, True, nn.ReLU()),
        (8, 0.2, True, nn.ReLU()),
        (1, 0.0, False, None)
    ]

    y_train = train_data.pop("y")
    y_test = test_data.pop("y")

    network_model = MLP(input_dim=len(train_data.columns), layers=layers)

    # Binary encoding of labels
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_test = encoder.transform(y_test)

    # Convert to 2D PyTorch tensors
    x_train = torch.tensor(train_data.values, dtype=torch.float32)
    y_train = torch.tensor(encoder.transform(y_train), dtype=torch.float32)

    x_test = torch.tensor(test_data.values, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.float32)

    classifier = NeuralNetBinaryClassifier(
        network_model,
        callbacks=[EarlyStopping(),
                   Checkpoint(f_params="params_{last_epoch[epoch]}.pt",
                              dirname=Path("dumps") / "nlp_models" / "checkpoints"),
                   # EpochScoring("f1", name="valid_f1", lower_is_better=False, target_extractor=target_conversion)
                   ],
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=1e-2,
        lr=1e-5,
        max_epochs=300,
        batch_size=8,
        verbose=True
    )

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test).tolist()

    # saving
    with open(Path("dumps") / "nlp_models" / "test_nn.pkl", "wb") as f:
        pickle.dump(classifier, f)

    print("Metrics of MLP")
    compute_metrics(y_pred, y_test.tolist(), sk_classifier_name="MLP Classifier")


def final_ensemble(feature_classifier: ClassifierMixin,
                   train_data: pd.DataFrame, test_data: pd.DataFrame,
                   tfidf_clf: RandomForestClassifier = None) -> None:
    # Add clf predictions
    # probs = tfidf_clf.predict_proba(all_data["text_"].tolist())
    #
    # feature_clf = {
    #     "score_0": probs[:, 0].tolist(),
    #     "score_1": probs[:, 1].tolist()
    # }
    #
    # f_df.update(feature_clf)

    # train_df, test_df, train_y, test_y = train_test_split(f_df, all_data["label"], random_state=31, shuffle=True, stratify=all_data["label"], test_size=0.25)

    y_train = train_data.pop("y")
    y_test = test_data.pop("y")

    feature_classifier.fit(train_data, y=y_train.tolist())

    y_pred = feature_classifier.predict(test_data).tolist()

    print("Metrics AFTER ensemble")
    compute_metrics(y_pred, y_test.tolist(), sk_classifier_name="Final ENSEMBLE")


def adaboost_classifier(train_config, training_data: pd.DataFrame, testing_data: pd.DataFrame):
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
    data_path_train = Path(dataset.BASE_DATASET) / f"{dataset.__class__.__name__}_train_features.csv"
    data_path_test = Path(dataset.BASE_DATASET) / f"{dataset.__class__.__name__}_test_features.csv"

    data_train = pd.read_csv(data_path_train)
    data_test = pd.read_csv(data_path_test)

    data_train["y"] = dataset.get_train_labels()
    data_test["y"] = dataset.get_test_labels()

    clf = sk_classifier_type(**train_config[sk_classifier_type.__name__])

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

    # adaboost_classifier(train_config, data_train, data_test)

    # final_ensemble(clf, data_train, data_test)

    neural_classifier(data_train, data_test)


if __name__ == "__main__":
    main()
