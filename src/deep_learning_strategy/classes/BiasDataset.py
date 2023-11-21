from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.settings import RANDOM_SEED
from src.feature_extraction.text_features import separate_html_entities


class BiasDataset(AbcDataset):
    BASE_DATASET = Path("dataset") / "bias"

    def __init__(self, target: str = "label", validation: float = None):
        super().__init__(target, validation)
        self._split_data = self._train_val_test()

    @staticmethod
    def get_path_to_testset() -> str:
        pass

    @staticmethod
    def get_path_to_trainset() -> str:
        pass

    def _train_val_test(self) -> Dict:
        # Fetch data
        all_data_path = os.path.join(self.BASE_DATASET, "samples.csv")
        train_df = pd.read_csv(all_data_path)

        # TODO: 0 = bad, but should be inverted
        train_df["y"] = train_df["Label"].replace({"BAD": 0, "NOT_BAD": 1})
        train_df.drop(columns="Label", inplace=True)
        train_y = train_df.pop("y").tolist()
        train_x = train_df["Text"].tolist()

        return {
            "train": {"x": train_x, "y": train_y},
            "test": {"x": train_x, "y": train_y}
        }
