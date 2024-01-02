from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch.cuda
from tqdm import tqdm

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset
from src.datasets.classes.Dataset import AbcDataset
from src.features_extraction.classes import GenderBiasDetector, IronyDetector, OffensivenessDetector, SarcasmDetector, \
    ChatGPTDetector, Wellformedness, TopicLM, EvidenceType, ParrotAdequacy, GibberishDetector, TextEmotion, \
    TextPolarity, TextGrammarErrors, EmotionLex, EmpathFeatures, LIWCFeatures, TextStatistics


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
        torch.cuda.empty_cache()

    train_features = {k: f[:len_train] for k, f in all_features.items()}
    test_features = {k: f[len_train:] for k, f in all_features.items()}

    df_train = pd.DataFrame.from_dict(train_features)
    df_test = pd.DataFrame.from_dict(test_features)

    # Remove columns with constant values from the train set
    df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]
    # Test set has the same columns as the train set
    df_test = df_test.loc[:, df_train.columns]

    if do_label:
        df_train["label"] = data["train"]["y"]
        df_test["label"] = data["test"]["y"]

    df_train.to_csv(Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_train_features.csv", index=False)
    df_test.to_csv(Path(dataset_.BASE_DATASET) / f"{dataset_.__class__.__name__}_test_features.csv", index=False)


if __name__ == "__main__":
    dataset = CallMeSexistDataset()
    compute_features(dataset, do_label=False)
