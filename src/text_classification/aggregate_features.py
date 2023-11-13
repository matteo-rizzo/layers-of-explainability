from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.deep_learning_strategy.classes.AMI2018Dataset import AMI2018Dataset
from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.classes.features.ChatGPTDetector import ChatGPTDetector
from src.text_classification.classes.features.EmotionLex import EmotionLex
from src.text_classification.classes.features.EmpathFeatures import EmpathFeatures
from src.text_classification.classes.features.EvidenceType import EvidenceType
from src.text_classification.classes.features.GibberishDetector import GibberishDetector
from src.text_classification.classes.features.LIWCFeatures import LIWCFeatures
from src.text_classification.classes.features.POSFeatures import POSFeatures
from src.text_classification.classes.features.ParrotAdequacy import ParrotAdequacy
from src.text_classification.classes.features.TextEmotion import TextEmotion
from src.text_classification.classes.features.TextGrammarErrors import TextGrammarErrors
from src.text_classification.classes.features.TextPolarity import TextPolarity
from src.text_classification.classes.features.TextStatistics import TextStatistics
from src.text_classification.classes.features.TopicLM import TopicLM
from src.text_classification.classes.features.Wellformedness import Wellformedness


def compute_features(dataset_: AbcDataset, do_label: bool = False):
    data = dataset_.get_train_val_test_split()

    train_texts: list[str] = data["train"]["x"]
    test_texts: list[str] = data["test"]["x"]

    len_train: int = len(train_texts)
    train_texts.extend(test_texts)

    feature_transforms = [EvidenceType, TopicLM, Wellformedness, ChatGPTDetector,
                          ParrotAdequacy, GibberishDetector, TextEmotion, TextPolarity, TextGrammarErrors]
    feature_transforms += [TextStatistics, LIWCFeatures, EmpathFeatures, EmotionLex]
    feature_transforms = [f(use_gpu=True) for f in feature_transforms]

    all_features: dict[str, list[float]] = dict()
    for f in feature_transforms:
        res = f.extract(train_texts)
        all_features.update(res)

    train_features = {k: f[:len_train] for k, f in all_features.items()}
    test_features = {k: f[len_train:] for k, f in all_features.items()}

    if do_label:
        df = pd.DataFrame.from_dict(train_features)
        df["label"] = data["train"]["y"]
        df.to_csv(Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_train_features.csv", index=False)

        df = pd.DataFrame.from_dict(test_features)
        df["label"] = data["test"]["y"]
        df.to_csv(Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_test_features.csv", index=False)
    else:
        pd.DataFrame.from_dict(train_features).to_csv(Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_train_features.csv", index=False)
        pd.DataFrame.from_dict(test_features).to_csv(Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_test_features.csv", index=False)


if __name__ == "__main__":
    # data_path = Path("dataset") / "fake_reviews_dataset.csv"
    dataset = CallMeSexistDataset()
    # dataset = CGReviewDataset(test_size=0.25)
    compute_features(dataset, do_label=True)
