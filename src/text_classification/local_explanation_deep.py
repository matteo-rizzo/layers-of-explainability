"""
Get explanations for HuggingFace LM, using Shap and Transhap (see plots/ folder)

LM dump must be set in src/deep_learning_strategy/config.yml

"""

from __future__ import annotations

import os
from pprint import pprint

import numpy as np

from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.HuggingFaceCallMeSexistDataset import HuggingFaceCallMeSexistDataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.explainable_strategy.SHAP.TranShapExplainer import TranShapExplainer
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "deep_learning_strategy", "config.yml"))
BATCH_SIZE = config["training"]["test_batch_size"]
TEST_MODEL_NAME = config["testing"]["model_name"]
TARGET_LABEL = config["testing"]["target_label"]
DATASET: AbcDataset = HuggingFaceCallMeSexistDataset()


def main():
    # Load data
    pipeline = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    predictions = pipeline.test(DATASET.get_test_data(), TARGET_LABEL)
    metrics = DATASET.compute_metrics(y_pred=np.array(predictions), y_true=DATASET.get_test_labels())
    pprint(metrics)

    # explainer = HuggingFaceShapExplainer(pipeline.pipeline, pipeline.pipeline.tokenizer, target_label=TARGET_LABEL)
    # explainer.run(DATASET.get_test_data()[:5], show=True)

    explainer = TranShapExplainer(pipeline.pipeline, pipeline.pipeline.tokenizer, target_label=TARGET_LABEL,
                                  device=0 if config["use_gpu"] else "cpu")
    explainer.run(DATASET.get_test_data(), explain_ids=[4, 5, 30, 40, 50, 100], show=False, out_label_name="sexist")


if __name__ == "__main__":
    main()
