from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.explainable_strategy.SHAP.LocalShapExplainer import LocalShapExplainer
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_1700735225.8083348.pkl"


# LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "NeuralNetBinaryClassifier" / "model_1698674139.4118443.pkl"


def main():
    # Load data
    data_train, data_test = load_encode_dataset(dataset=DATASET, scale=True, features=None)
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    data_train.pop("y")
    y_true_test = data_test["y"].tolist()

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)
    sk_classifier_type = clf.__class__

    # Test evaluation
    tmu = TrainingModelUtility(train_config, sk_classifier_type, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_test, DATASET.compute_metrics)

    explainer = LocalShapExplainer(clf)

    feat_names = defaultdict(str)
    feat_names.update({v: v for v in data_test.columns.tolist()})

    explainer.run_tree(data_test.iloc[0:10, :], DATASET.get_train_data(), y_true_test, feat_names, label_names={0: "not sexist", 1: "sexist"})

    # Add local exp


if __name__ == "__main__":
    main()
