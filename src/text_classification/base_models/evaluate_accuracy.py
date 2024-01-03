"""
Script to GS and fit a classifier on review dataset, to use as feature extractor

INSTRUCTIONS:
1. Set settings variables
    - SK_CLASSIFIER_TYPE
    - SK_CLASSIFIER_PARAMS: dict of parameters that must be passed to the classifier instance
2. Set dataset in variable DATASET
3. Set DO_GRID_SEARCH variable

Classifier/grid search configuration is to be set in "src/text_classification/config/config.yml"
"""

from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

import joblib
from xgboost import XGBClassifier

from src.text_classification.base_models.classes.training.TrainingModelUtility import TrainingModelUtility
from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.IMDBDataset import IMDBDataset
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
        base_est_gs_config = {f"estimator__{k}": vs for k, vs in
                              train_config["grid_search_params"][base_est_name].items()}
        train_config[f"{SK_CLASSIFIER_TYPE.__name__}"].update(base_est_config)
        train_config["grid_search_params"][f"{SK_CLASSIFIER_TYPE.__name__}"].update(base_est_gs_config)
    return train_config


# Path to a trained models on sexism dataset
MODEL_DIR = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_CMS_FINAL_RFE.pkl"
# MODEL_DIR = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_IMDB_FINAL_RFE.pkl"

SK_CLASSIFIER_TYPE: type = XGBClassifier
DATASET = CallMeSexistDataset()  # IMDBDataset()


def main():
    # train_config: dict = load_yaml("src/text_classification/base_models/config.yml")
    # bias_dataset = BiasDataset()

    # data_train, data_test = load_encode_dataset(dataset=bias_dataset, max_scale=True, features=None)
    train_config: dict = load_yaml("src/text_classification/base_models/config.yml")
    bias_dataset = DATASET
    clf = joblib.load(MODEL_DIR)

    features = list(set(clf.feature_names_in_.tolist()))
    data_train, data_test = load_encode_dataset(dataset=bias_dataset, max_scale=True, features=features)
    # data_base, _ = load_encode_dataset(dataset=DATASET, max_scale=True, features=None)

    # Load model
    # missing_features = list(set(clf.feature_names_in_.tolist()) - set(data_train.columns.tolist()))
    # for f in missing_features:
    #     data_train[f] = np.random.random((data_train.shape[0],))
    #
    # data_train = data_train[clf.feature_names_in_.tolist() + ["y"]]

    # Test evaluation
    tmu = TrainingModelUtility(train_config, SK_CLASSIFIER_TYPE, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_test, bias_dataset.compute_metrics)
    # Get probabilities
    # y_pred_probs = tmu.trained_classifier.predict_proba(data_train)[:, 1]

    # bias_computation(data_train, y_pred_probs, bias_dataset=bias_dataset)


if __name__ == "__main__":
    main()
