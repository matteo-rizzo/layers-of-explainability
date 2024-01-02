"""
Test features

INSTRUCTIONS:
1. Set settings variables
    - SK_CLASSIFIER_TYPE
    - SK_CLASSIFIER_PARAMS: dict of parameters that must be passed to the classifier instance
2. Set dataset in variable DATASET

Classifier/grid search configuration is to be set in "src/text_classification/config/config.yml"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.Dataset import AbcDataset
from src.features_analysis.classes.FeatureAblator import FeatureAblator
from src.features_analysis.classes.FeatureImportance import FeatureImportance
from src.text_classification.base_models.evaluate_accuracy import update_params_composite_classifiers
from src.utils.yaml_manager import load_yaml

# HOWTO:
# - first run importance analysis for each dataset and classifier of interest. This will generate clusters of correlated features and do importance plot.
# - secondly, run ablation removing 1 or more important clusters, looking at the importance plot. It will generate a ODS sheet with metric drops.

# Dataset to use
DATASET: AbcDataset = CallMeSexistDataset()

# 'ablation' will try to retrain a model removing 1 feature at a time,
# or if 'FEATURE_CLUSTERS_REMOVE' is a list of features it will try to remove only this set of features
# 'importance' will require a fitted model in 'MODEL_DIR' and will try feature permutation to understand the feature importance in model
MODE: str = "importance"  # "ablation", "importance"

# For feature importance, path to a trained models
MODEL_DIR = Path("dumps") / "nlp_models" / "XGBClassifier" / "model_CMS_FINAL.pkl"

# For feature ablation, the clusters of features to be removed
# (all feature in same clusters, as well as the cluster name will be removed)
FEATURE_CLUSTERS_REMOVE: list[str] | None = None  # ["GenderBiasDetector_LABEL_1", "TopicLM_sports"]


def main():
    train_config: dict = load_yaml("src/text_classification/config/config.yml")

    if MODE == "ablation":
        # SETTINGS:
        # ------------- SK learn classifiers
        SK_CLASSIFIER_TYPE: type = joblib.load(MODEL_DIR).__class__
        SK_CLASSIFIER_PARAMS: dict = dict()  # dict(estimator=LogisticRegression())

        # ------------- TORCH with SKORCH
        # SK_CLASSIFIER_TYPE: type = NeuralNetBinaryClassifier
        # SK_CLASSIFIER_PARAMS: dict = create_skorch_model_arguments(data_train)
        update_params_composite_classifiers(train_config, SK_CLASSIFIER_TYPE, SK_CLASSIFIER_PARAMS)

        abl = FeatureAblator(dataset=DATASET, train_config=train_config, classifier_type=SK_CLASSIFIER_TYPE,
                             classifier_kwargs=SK_CLASSIFIER_PARAMS, out_path="dumps/ablation")
        if FEATURE_CLUSTERS_REMOVE:
            with open(
                    Path(
                        "dumps") / "ablation" / f"features_{DATASET.__class__.__name__}_{SK_CLASSIFIER_TYPE.__name__}.json",
                    mode="r") as fo:
                feature_clusters: dict[str, list[str]] = json.load(fo)
            feature_excluded = [f for fc in FEATURE_CLUSTERS_REMOVE for f in
                                feature_clusters[fc]] + FEATURE_CLUSTERS_REMOVE
        else:
            feature_excluded = None

        abl.run_ablation(k_folds=5, use_all_data=False, exclude_feature_set=feature_excluded)
    else:
        clf = joblib.load(MODEL_DIR)
        abl = FeatureImportance(dataset=DATASET, out_path="dumps/ablation")
        abl.run_importance_test(clf, metric="accuracy", decorrelate=True, use_validation=True, use_all_data=False)


if __name__ == "__main__":
    main()
