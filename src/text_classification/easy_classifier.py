""" Script to GS and fit a classifier on review dataset, to use as feature extractor """
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.text_classification.main import compute_metrics
from src.utils.yaml_manager import load_yaml

sk_classifier_type = RandomForestClassifier


def main():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    data_path_train = Path("dataset") / "ami2018_misogyny_detection" / "AMI2018Dataset_train_features.csv"
    data_path_test = Path("dataset") / "ami2018_misogyny_detection" / "AMI2018Dataset_test_features.csv"

    scaler = StandardScaler()

    data_train = pd.read_csv(data_path_train)
    train_labels = pd.read_csv(data_path_train.parent / "en_training_anon.tsv", sep="\t")["misogynous"]

    data_test = pd.read_csv(data_path_test)
    test_labels = pd.read_csv(data_path_test.parent / "en_testing_labeled_anon.tsv", sep="\t")["misogynous"]

    # features = data_train.columns.values.tolist()

    # sel_data = FeatureSelector.variance_threshold(data_train, threshold=0.8)
    # sel_data = FeatureSelector.k_best(data_train, train_labels)
    # selected_features = sel_data.columns.values.tolist()
    # data_train = data_train.iloc[:, selected_features]
    # data_test = data_test.iloc[:, selected_features]

    features = data_train.columns.values.tolist()

    data_train = scaler.fit_transform(data_train)
    # data_train = pd.DataFrame(data_train, columns=selected_features)
    data_train = pd.DataFrame(data_train, columns=features)

    data_test = scaler.transform(data_test)
    # data_test = pd.DataFrame(data_test, columns=selected_features)
    data_test = pd.DataFrame(data_test, columns=features)

    data_train["y"] = train_labels
    data_test["y"] = test_labels

    base_clf = DecisionTreeClassifier(**train_config["DecisionTreeClassifier"])
    clf = AdaBoostClassifier(n_estimators=5000, random_state=11, estimator=base_clf)

    y_train = data_train.pop("y").tolist()
    y_test = data_test.pop("y").tolist()

    clf.fit(data_train, y=y_train)

    y_pred = clf.predict(data_test).tolist()

    print("Metrics AFTER AdaBoost")
    compute_metrics(y_pred, y_test, sk_classifier_name="Final ENSEMBLE")

    min_features_to_select = 1  # Minimum number of features to consider
    cv = KFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(data_train, y=y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()


if __name__ == "__main__":
    main()
