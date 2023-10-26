import argparse
import os.path
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
import torch.cuda

from src.deep_learning_strategy.classes.AMI2020Dataset import AMI2020Dataset
from src.deep_learning_strategy.classes.HuggingFaceAMI2020Dataset import HuggingFaceAMI2020Dataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.utils.ami_2020_scripts.evaluation_submission import read_gold, evaluate_task_b_singlefile
from src.utils.setup import PATH_TO_CONFIG, RANDOM_SEED, make_deterministic, print_namespace
from src.utils.yaml_manager import load_yaml

BATCH_SIZE = 4
TARGET_LABEL = "hateful"
TASK = "A"


def task_a(model_name: str, bs: int, target_label: str, m_f1: float):
    print("--> Task A: Predicting aggressiveness")
    pipeline = HuggingFacePipeline(model_name, bs)
    dataset = HuggingFaceAMI2020Dataset(target="A")
    predictions = pipeline.test(dataset.get_test_data(), target_label)
    metrics = AMI2020Dataset.compute_metrics(y_pred=np.ndarray(predictions), y_true=dataset.get_test_groundtruth())
    pprint(metrics)

    a_f1 = metrics["f1"]
    print(f"\n*** Task A f1 score (aggressiveness): {a_f1:.5f} ***")
    print(f"\n*** Task A score (misogyny + aggressiveness): {(m_f1 + a_f1) / 2:.5f} ***")


def task_b(model_name: str, bs: int, target_label: str, predictions: List):
    print("--> Task B ")
    pipeline = HuggingFacePipeline(model_name, bs)
    dataset = HuggingFaceAMI2020Dataset(target="M")
    predictions_synthetic = pipeline.test(dataset.get_synthetic_test_data(), target_label)

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
    print(f"\n*** Task B f1 score (misogyny and bias): {score:.5f} ***")


def main(ns: argparse.Namespace):
    bs = ns.batch_size
    target_label = ns.target_label
    task = ns.task

    config = load_yaml(PATH_TO_CONFIG)
    test_model_name = config["testing"]["task_m_model_name"]
    task_a_model_name = config["testing"]["task_a_model_name"]
    task_b_model_name = config["testing"]["task_b_model_name"]
    add_synthetic = config["add_synthetic"]

    print("*** PREDICTING MISOGYNY ***")
    pipeline = HuggingFacePipeline(test_model_name, bs)
    dataset = HuggingFaceAMI2020Dataset(augment_training=add_synthetic or task == "B")
    predictions = pipeline.test(dataset.get_test_data(), target_label)
    metrics = AMI2020Dataset.compute_metrics(y_pred=np.ndarray(predictions), y_true=dataset.get_test_groundtruth())
    pprint(metrics)

    m_f1 = metrics["f1"]
    torch.cuda.empty_cache()

    match task:
        case "A":
            task_a(task_a_model_name, bs, target_label, m_f1)
        case "B":
            task_b(task_b_model_name, bs, target_label, predictions)
        case _:
            raise ValueError("Task {} is not defined. Supported tasks are {} and {}".format(task, "A", "B"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--target_label", type=str, default=TARGET_LABEL)
    parser.add_argument("--task", type=str, default=TASK)
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
