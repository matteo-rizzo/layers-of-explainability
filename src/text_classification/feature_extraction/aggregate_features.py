from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from src.deep_learning_strategy.classes.AMI2018Dataset import AMI2020Dataset, AMI2018Dataset
from src.text_classification.feature_extraction.ChatGPTDetector import ChatGPTDetector
from src.text_classification.feature_extraction.EvidenceType import EvidenceType
from src.text_classification.feature_extraction.GibberishDetector import GibberishDetector
from src.text_classification.feature_extraction.ParrotAdequacy import ParrotAdequacy
from src.text_classification.feature_extraction.TextEmotion import TextEmotion
from src.text_classification.feature_extraction.TextGrammarErrors import TextGrammarErrors
from src.text_classification.feature_extraction.TextPolarity import TextPolarity
from src.text_classification.feature_extraction.TopicLM import TopicLM
from src.text_classification.feature_extraction.Wellformedness import Wellformedness


def compute_features(dataset_: AMI2020Dataset):
    data = dataset_.get_train_val_test_split()

    train_texts: list[str] = data["train"]["x"]
    test_texts: list[str] = data["test"]["x"]

    len_train: int = len(train_texts)
    train_texts.extend(test_texts)

    feature_transforms = [TextEmotion, TextPolarity, TextGrammarErrors, EvidenceType, TopicLM, Wellformedness, ChatGPTDetector, ParrotAdequacy, GibberishDetector]
    feature_transforms = [f(use_gpu=False) for f in feature_transforms]

    all_features = dict()
    for f in feature_transforms:
        res = f.extract(train_texts)
        all_features.update(res)

    train_features = {k: f[:len_train] for k, f in all_features.items()}
    test_features = {k: f[len_train:] for k, f in all_features.items()}

    pd.DataFrame.from_dict(train_features).to_csv(dataset_.BASE_AMI_DATASET / "fake_reviews_features.csv", index=False)
    pd.DataFrame.from_dict(test_features).to_csv(dataset_.BASE_AMI_DATASET / "fake_reviews_features.csv", index=False)


if __name__ == "__main__":
    # data_path = Path("dataset") / "fake_reviews_dataset.csv"
    dataset = AMI2018Dataset()
    compute_features(dataset)
