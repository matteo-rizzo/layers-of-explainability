from __future__ import annotations

import os
import re
from typing import Dict, Tuple, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.classes.Dataset import AbcDataset
from src.features_extraction.functional.text_features_utils import separate_html_entities
from src.utils.setup import RANDOM_SEED


class EACL21MisognyDataset(AbcDataset):
    BASE_DATASET = os.path.join("dataset", "misogyny_eacl2021", "data")

    def __init__(self, target: str = "level_1", validation: float = .0, test_size: float = 0.25):
        super().__init__(target, validation)
        self._test_size = test_size
        self._split_data = self._train_val_test()

    @staticmethod
    def get_path_to_testset() -> str:
        return f"{EACL21MisognyDataset.BASE_DATASET}/final_labels.csv"

    @staticmethod
    def get_path_to_trainset() -> str:
        return EACL21MisognyDataset.get_path_to_testset()

    @staticmethod
    def preprocessing(text_string: str) -> str:
        # Separate EMOJIS from adjacent words if necessary
        text_string = separate_html_entities(text_string)

        # Remove all substrings with < "anything but spaces" >
        text_string = re.sub("<\S+>", "", text_string, flags=re.RegexFlag.IGNORECASE).strip()

        # Remove double spaces
        return re.sub(" +", " ", text_string).strip()

    def _preprocess(self, corpora: List) -> Union[Tuple, None]:
        if not corpora:
            return None
        prepr_ds = tuple(
            [[self.preprocessing(text).strip() for text in corpus] if corpus is not None else None for corpus in
             corpora])
        # Remove texts with no content (e.g., only mentions). For new we deal with missing text in feature extraction modules
        # prepr_ds = tuple([[text for text in texts if text] if texts is not None else None for texts in prepr_ds])
        return prepr_ds

    def _train_val_test(self) -> Dict:
        # Fetch data
        all_data_path = self.get_path_to_testset()
        train_df = pd.read_csv(all_data_path)
        # Drop empty
        train_df = train_df.dropna(subset=['body'])
        # Split train and test
        train_df, test_df = train_test_split(train_df, test_size=self._test_size, random_state=RANDOM_SEED,
                                             shuffle=True, stratify=train_df[self._target])
        train_x, train_y = train_df["body"].tolist(), train_df[self._target].tolist()
        test_x, test_y = test_df["body"].tolist(), test_df[self._target].tolist()

        # Preprocessing train and test set
        train_x, test_x = self._preprocess([train_x, test_x])
        train_y, test_y = ([0 if label == 'Nonmisogynistic' else 1 for label in train_y],
                           [0 if label == 'Nonmisogynistic' else 1 for label in test_y])
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
    ds = EACL21MisognyDataset()
    print("")
