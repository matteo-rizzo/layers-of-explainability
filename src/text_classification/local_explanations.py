from __future__ import annotations

from collections import defaultdict
from importlib import import_module
from pathlib import Path

import joblib

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.explainable_strategy.SHAP.LocalShapExplainer import LocalShapExplainer
from src.text_classification.classes.features.Feature import Feature
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_1701189623.104557.pkl"


def read_feature_descriptions(column_names: list[str]) -> dict[str, str]:
    """
    Retrieve feature description from column names.
    Assume module and classes for feature extraction are named the same.
    This will infer module names from feature names, load them and get descriptions.
    """
    modules = list({e.split("_")[0] for e in column_names})
    feature_dict = defaultdict(lambda: "Other features")
    for mod_path in modules:
        module = import_module(f"src.text_classification.classes.features.{mod_path}")
        feature_class: type[Feature] = getattr(module, mod_path)
        descriptions = feature_class.label_description()
        if descriptions is not None:
            feature_dict.update(descriptions)
    return feature_dict


def main():
    # Load data
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True, features=None)
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

    # feat_names = defaultdict(lambda: "Other features")
    # feat_names.update({v: v for v in data_test.columns.tolist()})
    feat_names = read_feature_descriptions(data_test.columns.tolist())

    explainer.run_tree(data_test.iloc[0:10, :], DATASET.get_train_data(), y_true_test, feat_names, label_names={0: "not sexist", 1: "sexist"}, effect_threshold=.02)


if __name__ == "__main__":
    main()
