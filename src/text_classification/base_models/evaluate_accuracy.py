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

from pathlib import Path

import joblib
from xgboost import XGBClassifier

from src.text_classification.base_models.classes.training.TrainingModelUtility import TrainingModelUtility
from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.IMDBDataset import IMDBDataset
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

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
