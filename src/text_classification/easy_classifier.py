"""
Script to GS and fit a classifier on review dataset, to use as feature extractor

INSTRUCTIONS:
1. Set settings variables
    - SK_CLASSIFIER_TYPE
    - SK_CLASSIFIER_PARAMS: dict of parameters that must be passed to the classifier instance
2. Set dataset in variable DATASET
3. Set DO_GRID_SEARCH variable

Classifier/grid search configuration is to be set in "src/text_classification/config/classifier.yml"
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
from torch import nn

from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.classes.torch_models.MLP import MLP
from src.text_classification.classes.training.GridSearchUtility import GridSearchUtility
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml


def update_params_composite_classifiers(train_config: dict, SK_CLASSIFIER_TYPE: type, SK_CLASSIFIER_PARAMS: dict) -> dict:
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


# def target_conversion(x: torch.Tensor):
#     return np.round(torch.sigmoid(x).detach().numpy())


def create_skorch_model_arguments(train_data: pd.DataFrame) -> dict:
    """ Create parameters to train Neural model with skorch """
    layers = [
        (512, 0.1, True, nn.ReLU()),
        (512, 0.2, True, nn.ReLU()),
        (256, 0.1, True, nn.ReLU()),
        (256, 0.1, True, nn.ReLU()),
        (128, 0.1, True, nn.ReLU()),
        (128, 0.2, True, nn.ReLU()),
        (64, 0.1, True, nn.ReLU()),
        (32, 0.1, True, nn.ReLU()),
        (16, 0.1, True, nn.ReLU()),
        (8, 0.1, True, nn.ReLU()),
        (1, 0.1, False, None)
    ]

    network_model = MLP(input_dim=len(train_data.columns) - 1, layers=layers)

    classifier = dict(
        module=network_model,
        callbacks=[EarlyStopping(),
                   Checkpoint(f_params="params_{last_epoch[epoch]}.pt",
                              dirname=Path("dumps") / "nlp_models" / "checkpoints")
                   # EpochScoring("f1", name="valid_f1", lower_is_better=False, target_extractor=target_conversion)
                   ],
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.AdamW,
        verbose=True,
        device="cuda"
    )

    return classifier


DATASET: AbcDataset = IMDBDataset()
DO_GRID_SEARCH = True


def main():
    data_train, data_test = load_encode_dataset(dataset=DATASET, scale=True)
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    # SETTINGS:
    # ------------- SK learn classifiers
    SK_CLASSIFIER_TYPE: type = LogisticRegression
    SK_CLASSIFIER_PARAMS: dict = dict()  # dict(estimator=LogisticRegression())

    # ------------- TORCH with SKORCH
    # SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
    # SK_CLASSIFIER_PARAMS: dict = create_skorch_model_arguments(data_train)

    update_params_composite_classifiers(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)

    if DO_GRID_SEARCH:
        gsu = GridSearchUtility(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)
        clf = gsu.grid_search_best_params(data_train, DATASET.compute_metrics)
    else:
        tmu = TrainingModelUtility(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)
        clf = tmu.train_classifier(data_train)
        tmu.evaluate(data_test, DATASET.compute_metrics)

    # TO BE DONE: Implement saving if needed
    save_dir = Path("dumps") / "nlp_models" / clf.__class__.__name__ / f"model_{time.time()}.pkl"
    save_dir.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, save_dir)


if __name__ == "__main__":
    main()
