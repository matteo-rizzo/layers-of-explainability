from __future__ import annotations

import numpy as np
from datasets import Features, Value, ClassLabel, NamedSplit, Dataset

from src.datasets.classes.CallMeSexistDataset import CallMeSexistDataset


class HuggingFaceCallMeSexistDataset(CallMeSexistDataset):

    def __init__(self, target: str = "sexist"):
        super().__init__(target)

        self.hf_data = {k: {"text": v["x"], "label": v["y"]}
                        for k, v in self._split_data.items() if isinstance(v, dict)}

        features = Features({
            "text": Value("string"),
            "label": ClassLabel(num_classes=2, names=["no", "yes"], names_file=None, id=None)}
        )

        self.train_data = Dataset.from_dict(self.hf_data["train"], split=NamedSplit("train"), features=features)
        self.test_data = Dataset.from_dict(self.hf_data["test"], split=NamedSplit("test"), features=features)

    def get_test_labels(self) -> np.ndarray[int]:
        return np.asarray(super().get_test_labels())

    def get_train_data(self) -> Dataset:
        return self.train_data
