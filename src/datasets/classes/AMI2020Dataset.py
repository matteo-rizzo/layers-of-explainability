from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.classes.Dataset import AbcDataset
from src.features_extraction.functional.text_features_utils import separate_html_entities
from src.utils.setup import RANDOM_SEED


class AMI2020Dataset(AbcDataset):
    BASE_DATASET = os.path.join("dataset", "ami2020_misogyny_detection", "data")

    def __init__(self, augment_training=False, target="misogynous", validation: float = .0):
        super().__init__(target, validation)
        self._augment_training = augment_training
        self._split_data = self._train_val_test()

    def get_train_val_test_split(self) -> Dict:
        return self._split_data

    def get_synthetic_test_data(self) -> np.ndarray:
        return np.asarray(self._split_data["test_synt"]["y"])

    def get_test_ids(self) -> np.ndarray | list:
        return self._split_data["test"]["ids"]

    def get_synthetic_test_ids(self) -> np.ndarray | list:
        return self._split_data["test_synt"]["ids"]

    @staticmethod
    def get_path_to_testset() -> str:
        return str(Path(AMI2020Dataset.BASE_DATASET) / "test_raw_groundtruth.tsv")

    @staticmethod
    def get_path_to_trainset() -> str:
        return str(Path(AMI2020Dataset.BASE_DATASET) / "training_raw_groundtruth.tsv")

    @staticmethod
    def preprocessing(text_string: str) -> str:
        # Separate EMOJIS from adjacent words if necessary
        text_string = separate_html_entities(text_string)

        # Remove all substrings with < "anything but spaces" >
        text_string = re.sub("<\S+>", "", text_string, flags=re.RegexFlag.IGNORECASE).strip()

        # Remove double spaces
        return re.sub(" +", " ", text_string).strip()

    def __fetch_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path_to_train_raw_gt = os.path.join(AMI2020Dataset.BASE_DATASET, "training_raw_groundtruth.tsv")
        train_df = pd.read_csv(path_to_train_raw_gt, sep="\t", usecols=["id", "text", self._target])

        path_to_test_raw_gt = os.path.join(AMI2020Dataset.BASE_DATASET, "test_raw_groundtruth.tsv")
        test_df = pd.read_csv(path_to_test_raw_gt, sep="\t", usecols=["id", "text", self._target])

        return train_df, test_df

    def __fetch_synt_data(self) -> Tuple[pd.DataFrame, List, List, List]:
        path_to_train_synt_gt = os.path.join(AMI2020Dataset.BASE_DATASET, "training_synt_groundtruth.tsv")
        synt_train = pd.read_csv(path_to_train_synt_gt, sep="\t", usecols=["id", "text", "misogynous"])
        synt_train["id"] = "s_" + synt_train["id"].astype(str)

        path_to_test_synt_gt = os.path.join(AMI2020Dataset.BASE_DATASET, "test_synt_groundtruth.tsv")
        synt_test = pd.read_csv(path_to_test_synt_gt, sep="\t", usecols=["id", "text", "misogynous"])

        return synt_train, synt_test["text"].tolist(), synt_test[self._target].tolist(), synt_test["id"].tolist()

    def _preprocess(self, corpora: List) -> Union[Tuple, None]:
        if not corpora:
            return None
        prepr_ds = tuple(
            [[self.preprocessing(text).strip() for text in corpus] if corpus is not None else None for corpus in
             corpora])
        # Remove texts with no content (e.g., only mentions). For new we deal with missing text in feature extraction modules
        # prepr_ds = tuple([[text for text in texts if text] if texts is not None else None for texts in prepr_ds])
        return prepr_ds

    def _make_val_data(self, train_x: List, train_y: List, train_ids: List) -> Tuple:
        return train_test_split(train_x, train_y, train_ids, test_size=self._validation,
                                random_state=RANDOM_SEED, shuffle=True, stratify=train_y)

    def _sub_split_data(self, data: pd.DataFrame) -> Tuple:
        return data["text"].tolist(), data[self._target].tolist(), data["id"].tolist()

    def _train_val_test(self) -> Dict:

        train_df, test_df = self.__fetch_train_test()

        # Synthetic data augmentation
        synt_train, synt_test_x, synt_test_y, synt_test_ids = None, None, None, None
        if self._augment_training and self._target == "misogynous":
            synt_train, synt_test_x, synt_test_y, synt_test_ids = self.__fetch_synt_data()
            train_df = pd.concat([train_df, synt_train])

        train_x, train_y, train_ids = self._sub_split_data(train_df)
        test_x, test_y, test_ids = self._sub_split_data(test_df)

        # Preprocessing
        train_x, test_x, synt_test_x = self._preprocess([train_x, test_x, synt_test_x])

        validation = {}
        if self._validation > 0:
            train_x, val_x, train_y, val_y, train_ids, val_ids = self._make_val_data(train_x, train_y, train_ids)
            validation = {"val": {"x": val_x, "y": val_y, "ids": val_ids}}

        synt_test = {}
        if self._augment_training:
            synt_test = {"test_synt": {"x": synt_test_x, "y": synt_test_y, "ids": synt_test_ids}}

        return {
            "test_set_path": os.path.join(AMI2020Dataset.BASE_DATASET),
            "train": {"x": train_x, "y": train_y, "ids": train_ids},
            "test": {"x": test_x, "y": test_y, "ids": test_ids},
            **validation,
            **synt_test
        }
