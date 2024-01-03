"""
Get explanations for HuggingFace LM, using Shap and Transhap (see plots/ folder)

LM dump must be set in src/deep_learning_strategy/config.yml

"""

from __future__ import annotations

import os
from pprint import pprint

import numpy as np

from src.datasets.classes.Dataset import AbcDataset
from src.datasets.classes.HuggingFaceCallMeSexistDataset import HuggingFaceCallMeSexistDataset
from src.shap.core.classes.TranShapExplainer import TranShapExplainer
from src.text_classification.deep_learning.classes.HuggingFacePipeline import HuggingFacePipeline
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "text_classification", "deep_learning", "config.yml"))
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

    # VANILLA core
    # explainer = HuggingFaceShapExplainer(pipeline.pipeline, pipeline.pipeline.tokenizer, target_label=TARGET_LABEL)
    # explainer.run(DATASET.get_test_data()[:5], show=True)

    # TRANSHAP
    explainer = TranShapExplainer(pipeline.pipeline, pipeline.pipeline.tokenizer, target_label=TARGET_LABEL,
                                  device=0 if config["use_gpu"] else "cpu")

    test_set = DATASET.get_test_data()
    ids_to_explain = np.random.randint(0, len(test_set), size=30).tolist()

    # TRANSHAP + pretty visualization
    explainer.run_explain(test_set, ids_to_explain, label_names={0: "not sexist", 1: "sexist"}, top_k=10,
                          out_label_name="sexist", targets=DATASET.get_test_labels())


if __name__ == "__main__":
    main()
