from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.deep_learning_strategy.classes.BiasDataset import BiasDataset
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.dataset_utils.bias_madlibs import _read_word_list
from src.text_classification.dataset_utils.model_bias_analysis import add_subgroup_columns_from_text, compute_bias_metrics_for_model, SUBGROUP_AUC, NEGATIVE_CROSS_AUC, \
    POSITIVE_CROSS_AUC, POSITIVE_AEG, NEGATIVE_AEG
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

# Path to a trained models on sexism dataset
MODEL_DIR = Path("dumps") / "nlp_models" / "LogisticRegression" / "model_1700476850.319165.pkl"

SK_CLASSIFIER_TYPE: type = LogisticRegression


def get_final_metric(bias_df, overall_auc_test, model_name):
    bias_score = float(np.average([
        bias_df[model_name + "_" + SUBGROUP_AUC],
        bias_df[model_name + "_" + NEGATIVE_CROSS_AUC],
        bias_df[model_name + "_" + POSITIVE_CROSS_AUC]
    ]))

    p_aeg = float(bias_df[model_name + "_" + POSITIVE_AEG].mean())
    n_aeg = float(bias_df[model_name + "_" + NEGATIVE_AEG].mean())

    print(f"Bias Score = {bias_score:.5f}")
    print(f"Negative AEG = {n_aeg:.5f}")
    print(f"Positive AEG = {p_aeg:.5f}")
    print(f"AUC Score = {overall_auc_test:.5f}")
    return float(np.mean([overall_auc_test, bias_score]))


def main():
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    bias_dataset = BiasDataset()

    data_train, data_test = load_encode_dataset(dataset=bias_dataset, scale=True, features=None)
    # data_base, _ = load_encode_dataset(dataset=DATASET, scale=True, features=None)

    # Load model
    clf = joblib.load(MODEL_DIR)
    missing_features = list(set(clf.feature_names_in_.tolist()) - set(data_train.columns.tolist()))
    for f in missing_features:
        data_train[f] = np.random.random((data_train.shape[0],))

    data_train = data_train[clf.feature_names_in_.tolist() + ["y"]]

    # Test evaluation
    tmu = TrainingModelUtility(train_config, SK_CLASSIFIER_TYPE, dict())
    tmu.trained_classifier = clf
    tmu.evaluate(data_train, bias_dataset.compute_metrics)
    # Get probabilities
    y_pred_probs = tmu.trained_classifier.predict_proba(data_train)[:, 1]

    # Generate DF with subgroups

    data_train["text"] = bias_dataset.get_train_data()
    base_dir = bias_dataset.BASE_DATASET / "words"

    jobs = _read_word_list(base_dir, 'occupations.txt')
    groups = _read_word_list(base_dir, 'group_people.txt')
    adjectives_people = _read_word_list(base_dir, 'adjectives_people.txt')
    subgroups: list[str] = jobs + groups + adjectives_people

    mapping_ = dict(zip(groups, adjectives_people))

    add_subgroup_columns_from_text(data_train, text_column="text", subgroups=subgroups, expect_spaces_around_words=False)

    for k, v in mapping_.items():
        data_train[v] |= data_train[k]

    # FOR NOW ONLY PEOPLE
    subgroups = adjectives_people

    data_train = data_train.loc[:, subgroups]

    # ADD PREDICTIONS
    model_name = "sexist_pred"
    data_train[model_name] = y_pred_probs
    data_train["y"] = data_test["y"]

    # TODO: write script to evaluate subgroups

    bias_metrics = compute_bias_metrics_for_model(data_train, subgroups, model_name, "y")

    # Overall AUC
    overall_auc = roc_auc_score(data_train["y"], data_train[model_name])

    ami_bias_value = get_final_metric(bias_metrics, overall_auc, model_name=model_name)
    print(f"AMI bias = {ami_bias_value:.5f} (higher is better)")


if __name__ == "__main__":
    main()
