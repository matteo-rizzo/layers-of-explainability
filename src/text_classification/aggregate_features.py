from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.deep_learning_strategy.classes.AMI2018Dataset import AMI2018Dataset
from src.deep_learning_strategy.classes.CGReviewDataset import CGReviewDataset
from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.IMDBDataset import IMDBDataset
from src.text_classification.classes.features import (ChatGPTDetector, EmotionLex, EmpathFeatures,
                                                      EvidenceType, GibberishDetector, LIWCFeatures, ParrotAdequacy,
                                                      TextEmotion, TextGrammarErrors, TextPolarity, TextStatistics,
                                                      TopicLM, Wellformedness, GenderBiasDetector, IronyDetector,
                                                      OffensivenessDetector, SarcasmDetector)
from src.text_classification.classes.features.POSFeatures import POSFeatures


def compute_features(dataset_: AbcDataset, do_label: bool = False):
    data = dataset_.get_train_val_test_split()

    train_texts: list[str] = data["train"]["x"]
    test_texts: list[str] = data["test"]["x"]

    len_train: int = len(train_texts)
    train_texts.extend(test_texts)

    feature_transforms = [GenderBiasDetector, IronyDetector, OffensivenessDetector, SarcasmDetector]
    feature_transforms += [EvidenceType, TopicLM, Wellformedness, ChatGPTDetector,
                          ParrotAdequacy, GibberishDetector, TextEmotion, TextPolarity, TextGrammarErrors]
    feature_transforms += [TextStatistics, LIWCFeatures, EmpathFeatures, EmotionLex]
    feature_transforms = [f(use_gpu=True) for f in feature_transforms]

    all_features: dict[str, list[float]] = dict()
    for f in tqdm(feature_transforms, desc="Extracting features...."):
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
        pd.DataFrame.from_dict(train_features).to_csv(
            Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_train_features.csv", index=False)
        pd.DataFrame.from_dict(test_features).to_csv(
            Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_test_features.csv", index=False)


if __name__ == "__main__":
    dataset = IMDBDataset()
    compute_features(dataset, do_label=False)