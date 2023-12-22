from pathlib import Path

import pandas as pd
from sklearn.linear_model import RidgeClassifier

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.TMP_simple_model.pipeline import naive_classifier
from src.utils.yaml_manager import load_yaml

classifier_type = RidgeClassifier

DATASET: AbcDataset = CallMeSexistDataset()
DO_GRID_SEARCH = False

if __name__ == "__main__":
    train_config: dict = load_yaml("src/text_classification/config/classifier.yml")
    clf_params = train_config[classifier_type.__name__]

    print("*** Predicting misogyny ")

    data = DATASET._split_data

    m_pred, pipe_m = naive_classifier(classifier_type(**clf_params), data, return_pipe=True)
    m_f1 = DATASET.compute_metrics(data["test"]["y"], m_pred, classifier_type.__name__)["f1"]
