from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.deep_learning_strategy.classes.BiasDataset import BiasDataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
from src.utils.yaml_manager import load_yaml

# Path to a trained models on sexism dataset
MODEL_DIR = Path("dumps") / "nlp_models" / "LogisticRegression" / "model_1700476850.319165.pkl"

SK_CLASSIFIER_TYPE: type = LogisticRegression


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

    # TODO: write script to evaluate subgroups

if __name__ == "__main__":
    main()
