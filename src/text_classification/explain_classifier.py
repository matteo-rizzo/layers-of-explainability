from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetBinaryClassifier

from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.CategoricalShapExplainer import CategoricalShapExplainer
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.easy_classifier import load_encode_dataset
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()
SK_CLASSIFIER_TYPE: type = RandomForestClassifier
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_1700493001.8366137.pkl"


# SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
# LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "NeuralNetBinaryClassifier" / "model_1698674139.4118443.pkl"


def main():
    # Load data
    data_train, data_test = load_encode_dataset()
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    data_train.pop("y")

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)

    # Test evaluation
    tmu = TrainingModelUtility(train_config, SK_CLASSIFIER_TYPE, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_test, DATASET.compute_metrics)

    explainer = CategoricalShapExplainer(clf)
    explainer.run(data_train, data_test.iloc[0:100, :], output_names=["not", "yes"])


if __name__ == "__main__":
    main()
