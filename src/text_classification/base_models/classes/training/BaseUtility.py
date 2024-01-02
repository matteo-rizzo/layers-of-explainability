from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from skorch import NeuralNet


class BaseUtility:
    def __init__(self, configuration_parameters: dict, base_classifier_type: type,
                 base_classifier_kwargs: dict | None = None):
        self.train_config = configuration_parameters
        self._base_classifier_kwargs: dict = base_classifier_kwargs if base_classifier_kwargs is not None else dict()
        self._base_classifier_type: type = base_classifier_type

    def preprocess_x_data(self, x: pd.DataFrame) -> Any:
        """ Stub called before feeding features to the model. By default, convert to tensor if using skorch. """
        if issubclass(self._base_classifier_type, NeuralNet):
            x = torch.tensor(x.values, dtype=torch.float32)
        return x

    def preprocess_y_data(self, y: Any) -> Any:
        """ Stub called before feeding target labels to the model. By default, convert to tensor if using skorch. """
        if issubclass(self._base_classifier_type, NeuralNet):
            y = torch.tensor(y.values, dtype=torch.float32)
        return y
