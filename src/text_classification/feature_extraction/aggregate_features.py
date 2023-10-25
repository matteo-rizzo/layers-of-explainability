from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.text_classification.feature_extraction.ChatGPTDetector import ChatGPTDetector
from src.text_classification.feature_extraction.EvidenceType import EvidenceType
from src.text_classification.feature_extraction.GibberishDetector import GibberishDetector
from src.text_classification.feature_extraction.ParrotAdequacy import ParrotAdequacy
from src.text_classification.feature_extraction.TextEmotion import TextEmotion
from src.text_classification.feature_extraction.TextGrammarErrors import TextGrammarErrors
from src.text_classification.feature_extraction.TextPolarity import TextPolarity
from src.text_classification.feature_extraction.TopicLM import TopicLM
from src.text_classification.feature_extraction.Wellformedness import Wellformedness


def compute_features(dataset_path: str | Path):
    data = pd.read_csv(dataset_path)

    texts = data["text_"].tolist()

    feature_transforms = [TextEmotion, TextPolarity, TextGrammarErrors, EvidenceType, TopicLM, Wellformedness, ChatGPTDetector, ParrotAdequacy, GibberishDetector]
    feature_transforms = [f() for f in feature_transforms]

    all_features = dict()
    for f in feature_transforms:
        res = f.extract(texts)
        all_features.update(res)

    pd.DataFrame.from_dict(all_features).to_csv(data_path.parent / "fake_reviews_features.csv", index=False)


if __name__ == "__main__":
    # data_path = Path("dataset") / "fake_reviews_dataset.csv"
    data_path = Path("dataset") / "ami2018_misogyny_detection" / "en_training_anon.tsv"
    compute_features(data_path)
