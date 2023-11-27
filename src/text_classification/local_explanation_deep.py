from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np

from src.deep_learning_strategy.classes.CallMeSexistDataset import CallMeSexistDataset
from src.deep_learning_strategy.classes.HuggingFaceAMI2020Dataset import HuggingFaceAMI2020Dataset
from src.deep_learning_strategy.classes.HuggingFaceCallMeSexistDataset import HuggingFaceCallMeSexistDataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.explainable_strategy.SHAP.CategoricalShapExplainer import CategoricalShapExplainer
from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.explainable_strategy.SHAP.HuggingFaceShapExplainer import HuggingFaceShapExplainer
from src.explainable_strategy.SHAP.TranShapExplainer import TranShapExplainer
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset
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

    explainer = TranShapExplainer(pipeline.pipeline, pipeline.pipeline.tokenizer, target_label=TARGET_LABEL, device="cuda")
    explainer.run(DATASET.get_test_data(), explain_ids=[0, 1, 2], show=True)

    # shap_explain(dataset_processed, model=hf_pipeline, tokenizer=hf_pipeline.tokenizer, target_label=target_label)
    # transhap_explain(dataset_processed, explain_ids=[0, 99], model=hf_pipeline, tokenizer=hf_pipeline.tokenizer,
    #                  target_label=target_label,
    #                  device="cuda" if config["use_gpu"] else "cpu")


if __name__ == "__main__":
    main()
