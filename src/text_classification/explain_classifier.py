from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skorch import NeuralNetBinaryClassifier

from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.CategoricalShapExplainer import CategoricalShapExplainer
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = IMDBDataset()
SK_CLASSIFIER_TYPE: type = LogisticRegression
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "LogisticRegression" / "model_1700476850.319165.pkl"


# SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
# LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "NeuralNetBinaryClassifier" / "model_1698674139.4118443.pkl"


def main():
    # Load data
    data_train, data_test = load_encode_dataset(dataset=DATASET, scale=True, features=None)
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    data_train.pop("y")

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)

    # Test evaluation
    tmu = TrainingModelUtility(train_config, SK_CLASSIFIER_TYPE, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_test, DATASET.compute_metrics)

    explainer = CategoricalShapExplainer(clf)
    explainer.run(data_train, data_test.iloc[0:500, :], output_names=["real", "fake"])


if __name__ == "__main__":
    main()
