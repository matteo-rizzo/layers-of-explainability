import os
from typing import List

import numpy as np
from transformers import pipeline

from src.utils.yaml_manager import load_yaml


class HuggingFacePipeline:

    def __init__(self, model_name: str, batch_size: int = 4, top_k=1):
        config = load_yaml(os.path.join("src", "text_classification", "deep_learning", "config.yml"))
        self.pipeline = pipeline(task="text-classification", model=model_name, batch_size=batch_size, top_k=top_k,
                                 device=0 if config["use_gpu"] else "cpu", function_to_apply="sigmoid")

    def test(self, data: np.ndarray, target_label: str) -> List:
        predictions = self.pipeline(data)
        # return [1 if prediction[0]["label"] == target_label else 0 for prediction in predictions] # softmax version
        a = [prediction[0]["score"] for prediction in predictions]
        return (np.array(a) >= .5).astype(int).tolist()

    def get(self) -> pipeline:
        return self.pipeline
