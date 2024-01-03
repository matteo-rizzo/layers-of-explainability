import os.path
from pprint import pprint

import numpy as np

from src.datasets.classes.Dataset import AbcDataset
from src.datasets.classes.HuggingFaceCallMeSexistDataset import HuggingFaceCallMeSexistDataset
from src.datasets.classes.HuggingFaceIMDBDataset import HuggingFaceIMDBDataset
from src.text_classification.deep_learning.classes.HuggingFacePipeline import HuggingFacePipeline
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "text_classification", "deep_learning", "config.yml"))
BATCH_SIZE = config["training"]["test_batch_size"]
TEST_MODEL_NAME = config["testing"]["model_name"]
TARGET_LABEL = config["testing"]["target_label"]
DATASET: AbcDataset = HuggingFaceCallMeSexistDataset()  # HuggingFaceIMDBDataset()


def main():
    print(f"*** TESTING on {DATASET.__class__.__name__} ***")
    pipeline = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    predictions = pipeline.test(DATASET.get_test_data(), TARGET_LABEL)
    metrics = DATASET.compute_metrics(y_pred=np.array(predictions), y_true=DATASET.get_test_labels())
    pprint(metrics)

    # print(f"\n*** BIAS evaluation ***")
    # predictions = pipeline.test(HuggingFaceBiasDataset().get_test_data(), TARGET_LABEL)

    # bias_dataset = BiasDataset()
    # bias_computation(pd.DataFrame(bias_dataset.get_test_data()), predictions, bias_dataset=bias_dataset)


if __name__ == "__main__":
    main()
