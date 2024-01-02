from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd

from src.datasets.classes.Dataset import AbcDataset


class BiasDataset(AbcDataset):
    BASE_DATASET = Path("dataset") / "bias"

    def __init__(self, target: str = "Label", validation: float = None):
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

        train_df["y"] = train_df[self._target].replace({"BAD": 1, "NOT_BAD": 0})
        train_df.drop(columns=self._target, inplace=True)
        train_y = train_df.pop("y").tolist()
        train_x = train_df["Text"].tolist()

        return {
            "train": {"x": train_x, "y": train_y},
            "test": {"x": train_x, "y": train_y}
        }
