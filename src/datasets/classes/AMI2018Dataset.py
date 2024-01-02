import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from src.datasets.classes.AMI2020Dataset import AMI2020Dataset


class AMI2018Dataset(AMI2020Dataset):
    BASE_AMI_DATASET = os.path.join("dataset", "ami2018_misogyny_detection")

    def __init__(self, augment_training=False, target="M", validation: float = .0):
        super().__init__(augment_training, target, validation)
        self._target = "misogynous" if target == "M" else None
        assert target == "M", f"We don't currently support targets other than M, got target={target}"
        self._augment_training = augment_training
        self._validation = validation
        self._split_data = self._train_val_test()

    def get_synthetic_test_data(self) -> np.ndarray:
        raise NotImplementedError("AMI2018 has no synthetic data")

    def get_synthetic_test_ids(self) -> np.ndarray | list:
        raise NotImplementedError("AMI2018 has no synthetic data")

    @staticmethod
    def get_path_to_testset() -> str:
        return AMI2018Dataset.BASE_AMI_DATASET

    def __fetch_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path_to_train_raw_gt = os.path.join(AMI2018Dataset.BASE_AMI_DATASET, "en_training_anon.tsv")
        train_df = pd.read_csv(path_to_train_raw_gt, sep="\t", usecols=["id", "text", self._target])

        path_to_test_raw_gt = os.path.join(AMI2018Dataset.BASE_AMI_DATASET, "en_testing_labeled_anon.tsv")
        test_df = pd.read_csv(path_to_test_raw_gt, sep="\t", usecols=["id", "text", self._target])

        return train_df, test_df

    def __fetch_synt_data(self) -> Tuple[pd.DataFrame, List, List, List]:
        raise NotImplementedError("AMI2018 has no synthetic data")

    def _train_val_test(self) -> Dict:
        train_df, test_df = self.__fetch_train_test()

        train_x, train_y, train_ids = self._sub_split_data(train_df)
        test_x, test_y, test_ids = self._sub_split_data(test_df)

        # Preprocessing
        train_x, test_x = self._preprocess([train_x, test_x])

        validation = {}
        if self._validation > 0:
            train_x, val_x, train_y, val_y, train_ids, val_ids = self._make_val_data(train_x, train_y, train_ids)
            validation = {"val": {"x": val_x, "y": val_y, "ids": val_ids}}

        return {
            "test_set_path": os.path.join(AMI2018Dataset.BASE_AMI_DATASET),
            "train": {"x": train_x, "y": train_y, "ids": train_ids},
            "test": {"x": test_x, "y": test_y, "ids": test_ids},
            **validation
        }
