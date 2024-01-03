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
from sklearn.tree import DecisionTreeClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
from torch import nn
from xgboost import XGBClassifier

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.Dataset import AbcDataset
from src.text_classification.base_models.classes.torch_models.MLP import MLP
from src.text_classification.base_models.classes.training.GridSearchUtility import GridSearchUtility
from src.text_classification.base_models.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import get_excluded_features_dataset, load_encode_dataset
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


DATASET: AbcDataset = CallMeSexistDataset()
DO_GRID_SEARCH = False


def main():
    # Define which feature to use, or None to use everything
    keep_features = None

    # SETTINGS:
    # ------------- SK learn classifiers
    SK_CLASSIFIER_TYPE: type = XGBClassifier

    exclude_list = get_excluded_features_dataset(DATASET, SK_CLASSIFIER_TYPE)

    if exclude_list:
        print(len(exclude_list))

    all_metrics = []
    for seed in range(1):
        data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, features=keep_features)
        train_config: dict = load_yaml("src/text_classification/base_models/config.yml")

        # SETTINGS:
        # ------------- SK learn classifiers
        SK_CLASSIFIER_TYPE: type = XGBClassifier
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
            metrics = tmu.evaluate(data_test, DATASET.compute_metrics)
            all_metrics.append(metrics)

        # TO BE DONE: Implement saving if needed
        save_dir = Path("dumps") / "nlp_models" / clf.__class__.__name__ / f"model_{time.time()}.pkl"
        save_dir.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(clf, save_dir)

    df = pd.DataFrame(all_metrics)
    print("CV metrics on test data")
    for column in df.columns:
        print(f"average {column}: {df[column].mean()}")
    # for k, v in df.items():
    #     print(f"{self.trained_classifier.__class__.__name__} {k}: {v:.3f}")


if __name__ == "__main__":
    main()
