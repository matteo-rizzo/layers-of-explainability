import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
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
# 220 features, same as 330, worse recall

def test_rfe():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")

    DATASET: AbcDataset = CallMeSexistDataset()
    data_train, _ = load_encode_dataset(dataset=DATASET, max_scale=True, features=None, exclude_features=None)

    y_train = data_train.pop("y")
    # data_train = data_train.iloc[:, :10]

    # create pipeline
    rfe = RFECV(estimator=SK_CLASSIFIER_TYPE(**train_config[SK_CLASSIFIER_TYPE.__name__]), min_features_to_select=250, step=2)
    model = SK_CLASSIFIER_TYPE(**train_config[SK_CLASSIFIER_TYPE.__name__])
    # pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1011)

    x_transformed = rfe.fit_transform(data_train, y_train)
    # print(x_transformed.columns.tolist())

    # summarize all features
    selected = list()
    print(type(rfe.support_[0]))
    for i in range(data_train.shape[1]):
        print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
        if rfe.support_[i]:
            selected.append(rfe.feature_names_in_[i])
    print(selected)

    n_scores = cross_val_score(model, data_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise", verbose=0)
    print(f"Baseline scores on CV: {float(np.mean(n_scores)):.3}")

    n_scores = cross_val_score(model, data_train.loc[:, selected], y_train, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise", verbose=0)
    print(f"Scores on CV with only {len(selected)} features: {float(np.mean(n_scores)):.3}")

    print("*** Removed")
    print(list(set(data_train.columns) - set(selected)))


def get_feature_by_importance():
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
    test_rfe()
