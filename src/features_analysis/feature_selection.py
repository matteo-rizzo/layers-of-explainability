import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.Dataset import AbcDataset
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

SK_CLASSIFIER_TYPE: type = XGBClassifier
DATASET: AbcDataset = CallMeSexistDataset()


# 330 features: 0.902 baseline and after
# 270 features: 0.902 baseline, 0.903 after
# try to remove 170 features
# 214 features 0.904 NO BEST
# 220 features, best
# 210 features, worse recall, better acc
# Try 250, or keep 270 -> keep 220 as it is justifiable

def get_model(conf):
    return SK_CLASSIFIER_TYPE(**conf[SK_CLASSIFIER_TYPE.__name__])


def cv_evaluate(model, x, y, cv, scoring, jobs):
    cv_scores = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=jobs, error_score="raise", verbose=0)
    # scores will now be a dict with keys 'test_accuracy' and 'test_f1'
    cv_metrics = {k: float(np.mean(v)) for k, v in cv_scores.items()}
    return cv_metrics


def selection_rfe(min_features_to_select: int = None, n_jobs=-1, rfe_metric: str = "f1_macro"):
    """
    Perform Recursive Feature Elimination using classifier defined in SK_CLASSIFIER_TYPE, and trying to keep several numbers of features

    @param min_features_to_select: RFE will stop when reaching this number of features
    @param rfe_metric: sklearn scoring callable or string (only single metric is allowed)
    @param n_jobs: max parallel threads
    """
    train_config: dict = load_yaml("src/text_classification/config/config.yml")

    data_train, _ = load_encode_dataset(dataset=DATASET, max_scale=True, features=None, exclude_features=None)

    y_train = data_train.pop("y")

    rkfcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=11)

    # Create a dictionary of the metrics you want to calculate
    scoring = {"accuracy": make_scorer(accuracy_score),
               "f1": make_scorer(f1_score, average="macro")}
    cv_metrics = cv_evaluate(get_model(train_config), data_train, y_train, rkfcv, scoring, n_jobs)
    run_log = {"metrics_before": cv_metrics, "removed": list()}

    out_path = Path(
        "dumps") / "feature_selection" / f"RFE_{DATASET.__class__.__name__}_{SK_CLASSIFIER_TYPE.__name__}_{time.time()}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rfe = RFECV(estimator=get_model(train_config), min_features_to_select=min_features_to_select, scoring=rfe_metric,
                cv=rkfcv, step=1, n_jobs=n_jobs)
    x_transformed = rfe.fit_transform(data_train, y_train)

    plt.figure(figsize=(32, 9))
    plt.xlabel("Number of features selected")
    plt.ylabel(f"Mean test '{rfe_metric}'")
    plt.errorbar(
        range(min_features_to_select, len(rfe.cv_results_["mean_test_score"]) + min_features_to_select),
        rfe.cv_results_["mean_test_score"],
        yerr=rfe.cv_results_["std_test_score"],
        fmt="-o"
    )
    plt.title(f"RFE with min={min_features_to_select} features")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=400)

    # summarize all features
    selected = [rfe.feature_names_in_[i] for i in range(data_train.shape[1]) if rfe.support_[i]]

    assert x_transformed.shape[1] == len(selected), f"Wrong num of features: {x_transformed.shape[1]}, {len(selected)}"
    assert rfe.n_features_ == len(selected), f"Unexpected n of features: {rfe.n_features_}, {len(selected)}"

    cv_metrics = cv_evaluate(get_model(train_config), data_train.loc[:, selected], y_train, rkfcv, scoring, n_jobs)
    removed_columns = list(set(data_train.columns) - set(selected))
    run_log["metrics_after"] = cv_metrics
    run_log["removed"] = removed_columns

    with open(out_path.with_suffix(".json"), mode="w", encoding="utf-8") as fo:
        json.dump(run_log, fo, indent=2)


def get_feature_by_importance():
    """ Function to read results of feature importance run and compute a number of features to remove. Technically, using directly RFE is better. """
    feat_dict = json.load(open("dumps/ablation/features_CallMeSexistDataset_XGBClassifier.json"))
    importance_df = pd.read_csv(
        "dumps/ablation/importance_reduced_set_CallMeSexistDataset_XGBClassifier_validation.csv", index_col="index")

    feat_to_remove: list[str] = importance_df[~(importance_df["25%"] >= 0)].index.to_list()
    all_feat_to_remove = [sf for f in feat_to_remove for sf in feat_dict[f]] + feat_to_remove
    print(feat_to_remove)
    print(all_feat_to_remove)
    print(len(all_feat_to_remove))


if __name__ == "__main__":
    # get_feature_by_importance()
    selection_rfe(min_features_to_select=10)
