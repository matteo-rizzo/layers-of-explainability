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

from sklearn.linear_model import LogisticRegression

from src.deep_learning_strategy.classes.AMI2018Dataset import AMI2018Dataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.classes.experiments.FeatureAblator import FeatureAblator
from src.text_classification.easy_classifier import update_params_composite_classifiers
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()


def main():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    # SETTINGS:
    # ------------- SK learn classifiers
    SK_CLASSIFIER_TYPE: type = LogisticRegression
    SK_CLASSIFIER_PARAMS: dict = dict()  # dict(estimator=LogisticRegression())

    # ------------- TORCH with SKORCH
    # SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
    # SK_CLASSIFIER_PARAMS: dict = create_skorch_model_arguments(data_train)

    update_params_composite_classifiers(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)

    abl = FeatureAblator(dataset=DATASET, train_config=train_config, classifier_type=SK_CLASSIFIER_TYPE, classifier_kwargs=SK_CLASSIFIER_PARAMS, out_path="dumps/ablation")
    abl.run_ablation()


if __name__ == "__main__":
    main()
