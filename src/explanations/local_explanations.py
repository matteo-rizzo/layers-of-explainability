from __future__ import annotations

from collections import defaultdict
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.Dataset import AbcDataset
from src.features_extraction.classes.Feature import Feature
from src.shap.core.classes.LocalShapExplainer import LocalShapExplainer
from src.text_classification.base_models.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset, quantize_features, get_excluded_features_dataset
from src.utils.yaml_manager import load_yaml

DATASET: AbcDataset = CallMeSexistDataset()
LOAD_MODEL_DUMP = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_CMS_FINAL_RFE.pkl"


def read_feature_descriptions(column_names: list[str]) -> dict[str, str]:
    """
    Retrieve feature description from column names.
    Assume module and classes for feature extraction are named the same.
    This will infer module names from feature names, load them, and get descriptions.
    """
    modules = list({e.split("_")[0] for e in column_names})
    feature_dict = defaultdict(lambda: "Other features")
    for mod_path in modules:
        module = import_module(f"src.features_extraction.classes.{mod_path}")
        feature_class: type[Feature] = getattr(module, mod_path)
        descriptions = feature_class.label_description()
        if descriptions is not None:
            feature_dict.update(descriptions)
    return feature_dict


def main():
    # Load model
    clf = joblib.load(LOAD_MODEL_DUMP)
    sk_classifier_type = clf.__class__

    # Load data
    data_train, data_test = load_encode_dataset(dataset=DATASET, max_scale=True,
                                                exclude_features=get_excluded_features_dataset(DATASET, clf.__class__))
    train_config: dict = load_yaml("src/text_classification/base_models/config.yml")
    data_train.pop("y")
    y_true_test = data_test["y"].tolist()

    # Test evaluation
    tmu = TrainingModelUtility(train_config, sk_classifier_type, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_test, DATASET.compute_metrics)

    explainer = LocalShapExplainer(clf)
    feat_names = read_feature_descriptions(data_test.columns.tolist())

    # * Quantize the training set
    data_train_1, q_bins = quantize_features(data_train)
    # Testing/sanity checks
    data_train_2, _ = quantize_features(data_train, quantiles=q_bins)
    assert list(q_bins.keys()) == list(_.keys()), "Quantization problem: Keys are not equal"
    assert list(q_bins.values()) == list(_.values()), "Quantization problem: Values are not equal"
    assert (data_train_1 == data_train_2).all().all(), "Quantization problem: Results are not equal"
    # * Quantize the test set
    data_test_q, _ = quantize_features(data_test, quantiles=q_bins)

    ids_to_explain = np.random.randint(0, data_test.shape[0], size=30).tolist()
    print(f"Chosen ids: {ids_to_explain}")

    explainer.run_explain(data_test, ids_to_explain, DATASET.get_test_data(), y_true_test, feat_names,
                          label_names={0: "not sexist", 1: "sexist"}, top_k=10,
                          quantized_features=data_test_q)


if __name__ == "__main__":
    main()
