import json
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from tqdm import tqdm
from xgboost import XGBClassifier

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

SK_CLASSIFIER_TYPE: type = XGBClassifier


# 330 features: 0.902 baseline and after
# 270 features: 0.902 baseline, 0.903 after
# try to remove 170 features
# 214 features 0.904 NO BEST
# 220 features, best
# 210 features, worse recall, better acc
# Try 250, or keep 270 -> keep 220 as it is justifiable

def get_model(conf):
    return SK_CLASSIFIER_TYPE(**conf[SK_CLASSIFIER_TYPE.__name__])


def selection_rfe(feature_min_step: int = 50, n_jobs=-1):
    """
    Perform Recursive Feature Elimination using classifier defined in SK_CLASSIFIER_TYPE, and trying to keep several numbers of features

    @param feature_min_step: RFE will be run several times, each time removing this number of additional features
    @param n_jobs: max parallel threads
    """
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    DATASET: AbcDataset = CallMeSexistDataset()
    data_train, _ = load_encode_dataset(dataset=DATASET, max_scale=True, features=None, exclude_features=None)

    y_train = data_train.pop("y")
    # data_train = data_train.iloc[:, :10]

    rkfcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=11)

    all_features = data_train.shape[1]
    min_features = all_features // 2

    # Create a dictionary of the metrics you want to calculate
    scoring = {"accuracy": make_scorer(accuracy_score),
               "f1": make_scorer(f1_score, average="macro")}
    cv_scores = cross_validate(get_model(train_config), data_train, y_train, scoring=scoring, cv=rkfcv, n_jobs=n_jobs, error_score="raise", verbose=0)
    # scores will now be a dict with keys 'test_accuracy' and 'test_f1'
    cv_metrics = {k: float(np.mean(v)) for k, v in cv_scores.items()}
    print("Baseline scores on CV:")
    pprint(cv_metrics)

    run_log = {all_features: {"metrics": cv_metrics, "removed": list()}}

    n_feature_test: list[int] = [all_features - (feature_min_step * i) for i in range(1, (min_features // feature_min_step) + 1)]
    n_feature_test.append(n_feature_test[-1] - (min_features % feature_min_step))
    print(n_feature_test)
    for n in tqdm(n_feature_test):
        # create pipeline
        rfe = RFECV(estimator=get_model(train_config), min_features_to_select=n, cv=rkfcv, step=1, n_jobs=n_jobs)
        # pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
        x_transformed = rfe.fit_transform(data_train, y_train)

        # res = rfe.cv_results_
        # pprint(res)

        # summarize all features
        selected = [rfe.feature_names_in_[i] for i in range(data_train.shape[1]) if rfe.support_[i]]
        print(selected)

        assert x_transformed.shape[1] == len(selected), f"Wrong num of features: {x_transformed.shape[1]}, {len(selected)}"

        cv_scores = cross_validate(get_model(train_config), data_train.loc[:, selected], y_train, scoring=scoring, cv=rkfcv, n_jobs=n_jobs, error_score="raise", verbose=0)
        cv_metrics = {k: float(np.mean(v)) for k, v in cv_scores.items()}
        print(f"Scores on CV with only {len(selected)} features:")
        pprint(cv_metrics)

        removed_columns = list(set(data_train.columns) - set(selected))
        run_log[n] = {"metrics": cv_metrics, "removed": removed_columns}

        # print("*** Removed")
        # print(removed_columns)

    out_path = Path("dumps") / "feature_selection" / f"RFE_{DATASET.__class__.__name__}_{SK_CLASSIFIER_TYPE.__name__}_{time.time()}.json"
    with open(out_path, mode="w", encoding="utf-8") as fo:
        json.dump(run_log, fo, indent=2)


def get_feature_by_importance():
    """ Function to read results of feature importance run and compute a number of features to remove. Technically, using directly RFE is better. """
    # df = pd.read_excel("dumps/ablation/ablation_features_CallMeSexistDataset_XGBClassifier.ods", sheet_name="drop_ratio")
    feat_dict = json.load(open("dumps/ablation/features_CallMeSexistDataset_XGBClassifier.json"))
    importance_df = pd.read_csv("dumps/ablation/importance_reduced_set_CallMeSexistDataset_XGBClassifier_validation.csv", index_col="index")

    feat_to_remove: list[str] = importance_df[~(importance_df["25%"] >= 0)].index.to_list()
    # TOO MUCH feat_to_remove: list[str] = importance_df[importance_df["50%"] <= 0].index.to_list()
    # feat_to_remove: list[str] = importance_df[importance_df["50%"] < 0].index.to_list()
    # feat_to_remove: list[str] = importance_df[importance_df["50%"] < 0].index.to_list()
    # feat_to_remove_2: list[str] = importance_df[(importance_df["50%"] == 0) & (importance_df["25%"] < 0) & (importance_df["75%"] <= 0)].index.to_list()
    # feat_to_remove += feat_to_remove_2
    all_feat_to_remove = [sf for f in feat_to_remove for sf in feat_dict[f]] + feat_to_remove
    print(feat_to_remove)
    print(all_feat_to_remove)
    print(len(all_feat_to_remove))

    # low_important_features = ["LIWCFeatures_verb", "TextEmotion_fear", "LIWCFeatures_cogmech", "TextPolarity_subjectivity", "TextGrammarErrors_error_PUNCTUATION",
    #                           "EmpathFeatures_suffering", "EmpathFeatures_aggression", "LIWCFeatures_money", "LIWCFeatures_health", "EmpathFeatures_vacation",
    #                           "EmpathFeatures_journalism", "EmpathFeatures_cheerfulness", "EmpathFeatures_fear", "EmpathFeatures_fabric", "EmpathFeatures_school"]
    # all_unimportant_features = [sf for f in low_important_features for sf in feat_dict[f]] + low_important_features
    # low_metrics_features = set(df[df["accuracy"] <= 0].iloc[:, 0].tolist()) & set(df[df["f1"] <= 0].iloc[:, 0].tolist())
    # joint_features = set(all_unimportant_features) & low_metrics_features


if __name__ == "__main__":
    # get_feature_by_importance()
    selection_rfe(feature_min_step=30)