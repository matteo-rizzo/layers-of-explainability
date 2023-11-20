"""
Test features

INSTRUCTIONS:
1. Set settings variables
    - SK_CLASSIFIER_TYPE
    - SK_CLASSIFIER_PARAMS: dict of parameters that must be passed to the classifier instance
2. Set dataset in variable DATASET

Classifier/grid search configuration is to be set in "src/text_classification/config/classifier.yml"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src.deep_learning_strategy.classes.AMI2018Dataset import AMI2018Dataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.classes.experiments.FeatureAblator import FeatureAblator
from src.text_classification.classes.experiments.FeatureImportance import FeatureImportance
from src.text_classification.easy_classifier import update_params_composite_classifiers
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = IMDBDataset()
MODE: str = "importance"  # "ablation"
MODEL_DIR = Path("dumps") / "nlp_models" / "LogisticRegression" / "model_1700476850.319165.pkl"
FEATURE: str = "GenderBiasDetector_LABEL_1"


def main():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    if MODE == "ablation":
        # SETTINGS:
        # ------------- SK learn classifiers
        SK_CLASSIFIER_TYPE: type = LogisticRegression
        SK_CLASSIFIER_PARAMS: dict = dict()  # dict(estimator=LogisticRegression())

        # ------------- TORCH with SKORCH
        # SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
        # SK_CLASSIFIER_PARAMS: dict = create_skorch_model_arguments(data_train)
        update_params_composite_classifiers(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)

        abl = FeatureAblator(dataset=DATASET, train_config=train_config, classifier_type=SK_CLASSIFIER_TYPE, classifier_kwargs=SK_CLASSIFIER_PARAMS, out_path="dumps/ablation")
        with open(Path("dumps") / "ablation" / f"features_{DATASET.__class__.__name__}_{SK_CLASSIFIER_TYPE.__name__}.json", mode="r") as fo:
            feature_excluded = json.load(fo)[FEATURE] + [FEATURE]

        abl.run_ablation(k_folds=5, use_all_data=True, exclude_feature_set=feature_excluded)
    else:
        clf = joblib.load(MODEL_DIR)
        abl = FeatureImportance(dataset=DATASET, out_path="dumps/ablation")
        abl.run_importance_test(clf, metric="accuracy", decorrelate=True)


if __name__ == "__main__":
    main()
