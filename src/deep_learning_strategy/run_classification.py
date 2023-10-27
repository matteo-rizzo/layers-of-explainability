import os.path
from pprint import pprint

import pandas as pd
import torch.cuda

from src.deep_learning_strategy.classes.HuggingFaceAMI2020Dataset import HuggingFaceAMI2020Dataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.utils.ami_2020_scripts.dataset_handling import compute_metrics
from src.utils.ami_2020_scripts.evaluation_submission import read_gold, evaluate_task_b_singlefile
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "deep_learning_strategy", "config.yml"))
BATCH_SIZE = config["training"]["test_batch_size"]
TEST_MODEL_NAME = config["testing"]["task_m_model_name"]
TASK_A_MODEL_NAME = config["testing"]["task_a_model_name"]
TASK_B_MODEL_NAME = config["testing"]["task_b_model_name"]
TARGET_LABEL = config["testing"]["target_label"]
TASK = config["task"]
ADD_SYNTHETIC = config["add_synthetic"]


def task_a() -> float:
    print("--> Task A: Predicting aggressiveness")
    pipeline = HuggingFacePipeline(TASK_A_MODEL_NAME, BATCH_SIZE)
    dataset = HuggingFaceAMI2020Dataset(target="A")
    predictions = pipeline.test(dataset.get_test_data(), TARGET_LABEL)
    metrics = compute_metrics(y_pred=predictions, y_true=dataset.get_test_labels())
    pprint(metrics)

    score = metrics["f1"]
    print(f"\n*** Task A score (aggressiveness): {score:.5f} ***")

    return score


def task_b(predictions):
    print("--> Task B ")
    pipeline = HuggingFacePipeline(TASK_B_MODEL_NAME, BATCH_SIZE)
    dataset = HuggingFaceAMI2020Dataset(target="M")
    predictions_synthetic = pipeline.test(dataset.get_synthetic_test_data(), TARGET_LABEL)

    df_pred = pd.Series(predictions, index=pd.Index(dataset.get_test_ids(), dtype=str))
    df_pred_synt = pd.Series(predictions_synthetic, index=pd.Index(dataset.get_synthetic_test_ids(), dtype=str))

    # Preparing data for evaluation
    df_pred = df_pred.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
    df_pred_synt = df_pred_synt.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})

    path_to_testset = dataset.get_path_to_testset()
    raw_groundtruth, synt_groundtruth, identity_terms = read_gold(
        os.path.join(path_to_testset, "test_raw_groundtruth.tsv"),
        os.path.join(path_to_testset, "test_synt_groundtruth.tsv"),
        os.path.join(path_to_testset, "test_identity_terms.txt"),
        task="b")

    score = evaluate_task_b_singlefile(df_pred, df_pred_synt, raw_groundtruth, synt_groundtruth, identity_terms)
    print(f"\n*** Task B score (misogyny and bias): {score:.5f} ***")


def main():
    print("*** PREDICTING MISOGYNY ***")
    pipeline = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    dataset = HuggingFaceAMI2020Dataset(augment_training=ADD_SYNTHETIC or TASK == "B")
    predictions = pipeline.test(dataset.get_test_data(), TARGET_LABEL)
    metrics = compute_metrics(y_pred=predictions, y_true=dataset.get_test_labels())
    pprint(metrics)

    m_f1 = metrics["f1"]
    torch.cuda.empty_cache()

    match "task":
        case "A":
            a_f1 = task_a()
            a_score = (m_f1 + a_f1) / 2
            print(f"\n*** Task A score (misogyny + aggressiveness): {a_score:.5f} ***")
        case "B":
            task_b(predictions)


if __name__ == "__main__":
    main()
