from __future__ import annotations

import json
import os
import re
from typing import Dict, Tuple, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.classes.Dataset import AbcDataset
from src.features_extraction.functional.text_features_utils import separate_html_entities
from src.utils.setup import RANDOM_SEED


class CallMeSexistDataset(AbcDataset):
    BASE_DATASET = os.path.join("dataset", "call_me_sexist_sexism_detection")
    EMOJI2CODE: dict = json.load(open("dataset/asset/emoji_to_code.json", mode="r", encoding="utf-8"))

    def __init__(self, target: str = "sexist", validation: float = .0, test_size: float = 0.25):
        super().__init__(target, validation)
        self._test_size = test_size
        self._split_data = self._train_val_test()

    @staticmethod
    def get_path_to_testset() -> str:
        return f"{CallMeSexistDataset.BASE_DATASET}/dataset.csv"

    @staticmethod
    def get_path_to_trainset() -> str:
        return CallMeSexistDataset.get_path_to_testset()

    @staticmethod
    def preprocessing(text_string: str) -> str:
        text_string_p = re.sub("MENTION\d+", "MENTION", text_string).strip()

        text_string_p = re.sub(r"(http|ftp|https)://([\w_-]+(?:\.[\w_-]+)+)([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])",
                               "LINK", text_string_p).strip()

        # Separate EMOJIS from adjacent words if necessary
        text_string_p = separate_html_entities(text_string_p)

        # Remove double spaces
        text_string_p = re.sub(" +", " ", text_string_p).strip()

        return text_string_p

    def _preprocess(self, corpora: List) -> Union[Tuple, None]:
        if not corpora:
            return None
        prepr_ds = tuple(
            [[self.preprocessing(text).strip() for text in corpus] if corpus is not None else None for corpus in
             corpora])
        return prepr_ds

    def _train_val_test(self) -> Dict:
        # Fetch data
        all_data_path = self.get_path_to_testset()
        train_df = pd.read_csv(all_data_path)

        # Split train and test
        train_df, test_df = train_test_split(train_df, test_size=self._test_size, random_state=RANDOM_SEED,
                                             shuffle=True, stratify=train_df[self._target])
        train_x, train_y = train_df["text"].tolist(), train_df[self._target].astype(int).tolist()
        test_x, test_y = test_df["text"].tolist(), test_df[self._target].astype(int).tolist()

        # Preprocessing train and test set
        train_x, test_x = self._preprocess([train_x, test_x])

        # Add a validation set if required
        validation = dict()
        if self._validation > 0:
            train_x, val_x, train_y, val_y, train_ids, val_ids = (
                train_test_split(train_x, train_y, test_size=self._validation, random_state=RANDOM_SEED, shuffle=True,
                                 stratify=train_y))
            validation = {"val": {"x": val_x, "y": val_y, "ids": val_ids}}

        return {
            "train": {"x": train_x, "y": train_y},
            "test": {"x": test_x, "y": test_y},
            **validation
        }


if __name__ == "__main__":
    ds = CallMeSexistDataset()
    print("")
