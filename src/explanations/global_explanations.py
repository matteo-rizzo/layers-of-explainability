from __future__ import annotations

from pathlib import Path

import joblib

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.Dataset import AbcDataset
from src.shap.core.classes.CategoricalShapExplainer import CategoricalShapExplainer
from src.text_classification.base_models.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_1701079040.357031.pkl"


def main():
    # Load data
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, features=None)
    train_config: dict = load_yaml("src/text_classification/config/config.yml")
    data_train.pop("y")

    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)
    sk_classifier_type = clf.__class__

    # Test evaluation
    tmu = TrainingModelUtility(train_config, sk_classifier_type, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_test, DATASET.compute_metrics)

    explainer = CategoricalShapExplainer(clf)
    explainer.run_tree(data_train, data_test.iloc[0:100, :], output_names=["0", "1"])


if __name__ == "__main__":
    main()
