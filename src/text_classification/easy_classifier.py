""" Script to GS and fit a classifier on review dataset, to use as feature extractor """
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from torch import nn

from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.classes.MLP import MLP
from src.text_classification.main import compute_metrics
from src.utils.yaml_manager import load_yaml

SK_CLASSIFIER_TYPE: type = AdaBoostClassifier
SK_CLASSIFIER_PARAMS: dict = dict(estimator=LogisticRegression())
# RandomForestClassifier()
DATASET: AbcDataset = CGReviewDataset()  # AMI2018Dataset()
DO_GRID_SEARCH = True


def grid_search_best_params(training_data: pd.DataFrame, train_config):
    """
    Perform grid search of the selected classifier (see 'SK_CLASSIFIER_TYPE').

    @param training_data: dataframe with training data. Target should be in 'y' column;
    @param train_config: training configuration with parameters for GS and for classifier.
    """
    # Load configuration
    num_rand_states: int = train_config["grid_search_params"]["num_seeds"]
    test_size: float = train_config["grid_search_params"]["test_size"]

    # Initiate training
    avg_metrics: dict[str, list] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    best_params = list()

    # Per seed training
    for rs in range(num_rand_states):
        # Prepare splits
        train_df, val_df = train_test_split(training_data, random_state=0, shuffle=True, stratify=training_data["y"].tolist(), test_size=test_size)

        y_train = train_df.pop("y")
        y_val = val_df.pop("y")

        # Setup and train classifier
        params = train_config["grid_search_params"][SK_CLASSIFIER_TYPE.__name__]
        base_estimator_params = train_config[SK_CLASSIFIER_TYPE.__name__]

        gs = GridSearchCV(SK_CLASSIFIER_TYPE(**SK_CLASSIFIER_PARAMS), param_grid=params, verbose=10, refit=True, n_jobs=-1, cv=5)
        # Set base parameters in the estimator inside GS
        gs.estimator.set_params(**base_estimator_params)

        gs.fit(train_df, y_train)
        y_pred = gs.predict(val_df).tolist()

        # Calculate metrics
        metrics = compute_metrics(y_pred, y_val)

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

    return best_params


def update_params_composite_classifiers(train_config: dict) -> dict:
    """
    Some classifiers (ensemble, boosting, etc.) may need specific configuration depending on the type.
    For instance, AdaBoost takes an "estimator" argument to set the base estimator.
    This cannot be specified in YAML, and each estimator can have its hyperparameters and grid search params.
    Hence, this method updates the configuration of AdaBoost with all parameters of nested classifiers.
    """
    if SK_CLASSIFIER_TYPE in [AdaBoostClassifier]:  # potentially works for other "nested" cases
        base_est_name = SK_CLASSIFIER_PARAMS.setdefault("estimator", DecisionTreeClassifier()).__class__.__name__
        base_est_config = {f"estimator__{k}": v for k, v in train_config[base_est_name].items()}
        base_est_gs_config = {f"estimator__{k}": vs for k, vs in train_config["grid_search_params"][base_est_name].items()}
        train_config[f"{SK_CLASSIFIER_TYPE.__name__}"].update(base_est_config)
        train_config["grid_search_params"][f"{SK_CLASSIFIER_TYPE.__name__}"].update(base_est_gs_config)
    return train_config


def target_conversion(x: torch.Tensor):
    return np.round(torch.sigmoid(x).detach().numpy())


def neural_classifier(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    from skorch import NeuralNetBinaryClassifier
    from skorch.callbacks import EarlyStopping, Checkpoint

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


def load_encode_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset composed of extracted features based on the 'DATASET' global params that must be set to an AbcDataset object.

    @return: two dataframes with training and test data; target label is set to class 'y' and is encoded with numbers in 0...n-1
    """
    data_path_train = Path(DATASET.BASE_DATASET) / f"{DATASET.__class__.__name__}_train_features.csv"
    data_path_test = Path(DATASET.BASE_DATASET) / f"{DATASET.__class__.__name__}_test_features.csv"

    data_train = pd.read_csv(data_path_train)
    data_test = pd.read_csv(data_path_test)

    if DATASET.__class__ == CGReviewDataset:
        print("Removed CHATGpt detector features from the CGReviewDataset")
        data_train.pop("ChatGPTDetector_Human")
        data_train.pop("ChatGPTDetector_ChatGPT")
        data_test.pop("ChatGPTDetector_Human")
        data_test.pop("ChatGPTDetector_ChatGPT")

    train_labels = DATASET.get_train_labels()
    test_labels = DATASET.get_test_labels()

    # Binary encoding of labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.transform(test_labels)

    data_train["y"] = y_train
    data_test["y"] = y_test
    return data_train, data_test


def train_classifier(data_train: pd.DataFrame, train_config: dict) -> ClassifierMixin:
    """
    Train a scikit-learn classifier on training data and return the fitted object.
    """
    clf = SK_CLASSIFIER_TYPE(**SK_CLASSIFIER_PARAMS, **train_config[SK_CLASSIFIER_TYPE.__name__])

    y_train = data_train.pop("y")

    clf.fit(data_train, y=y_train.tolist())

    return clf


def evaluate_metrics(clf: ClassifierMixin, data_test: pd.DataFrame):
    """
    Evaluate dataset metrics of a scikit-learn classifier on testing data.
    """
    y_test = data_test.pop("y")

    y_pred = clf.predict(data_test).tolist()

    print("Classification metrics")
    DATASET.compute_metrics(y_pred, y_test, sk_classifier_name=clf.__class__.__name__)


def main():
    data_train, data_test = load_encode_dataset()

    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    update_params_composite_classifiers(train_config)

    if DO_GRID_SEARCH:
        clf = grid_search_best_params(data_train, train_config)
    else:
        clf = train_classifier(data_train, train_config)

    evaluate_metrics(clf, data_test)

    # neural_classifier(data_train, data_test)


if __name__ == "__main__":
    main()
