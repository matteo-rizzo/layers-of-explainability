from pathlib import Path
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from src.utils.ami_2020_scripts.evaluation_submission import read_gold, evaluate_task_b_singlefile

BASE_AMI_DATASET = Path("dataset/ami2020_misogyny_detection/data")

def loop_preprocess(fn: Callable[[str], str], texts: list[str]) -> list[str]:
    texts_preprocessed = list()
    for txt in texts:
        texts_preprocessed.append(fn(txt))
    return texts_preprocessed


def train_val_test(target: str = "M", validation: float = .0,
                   random_state: int = 0, add_synthetic_train: bool = False,
                   preprocessing_function: Callable[[str], str] = None) -> dict[str, dict[str, list]]:
    target = "misogynous" if target == "M" else "aggressiveness"

    # base_dataset
    train_df = pd.read_csv(BASE_AMI_DATASET / "training_raw_groundtruth.tsv", sep="\t", usecols=["id", "text", target])
    test_df = pd.read_csv(BASE_AMI_DATASET / "test_raw_groundtruth.tsv", sep="\t", usecols=["id", "text", target])

    synt_test_x, synt_test_y, synt_test_ids = None, None, None  # Synthetic test set, only returned if add_synthetic_train = True
    if add_synthetic_train and target == "misogynous":
        train_df_synt = pd.read_csv(BASE_AMI_DATASET / "training_synt_groundtruth.tsv", sep="\t", usecols=["id", "text", "misogynous"])
        train_df_synt["id"] = "s_" + train_df_synt["id"].astype(str)
        train_df = pd.concat([train_df, train_df_synt])

        test_df_synt = pd.read_csv(BASE_AMI_DATASET / "test_synt_groundtruth.tsv", sep="\t", usecols=["id", "text", "misogynous"])
        synt_test_x, synt_test_y, synt_test_ids = test_df_synt["text"].tolist(), test_df_synt[target].tolist(), \
            test_df_synt["id"].tolist()

    train_x, train_y, train_ids = train_df["text"].tolist(), train_df[target].tolist(), train_df["id"].tolist()
    test_x, test_y, test_ids = test_df["text"].tolist(), test_df[target].tolist(), test_df["id"].tolist()

    if preprocessing_function is not None:
        train_x = loop_preprocess(preprocessing_function, train_x)
        test_x = loop_preprocess(preprocessing_function, test_x)
        synt_test_x = loop_preprocess(preprocessing_function, synt_test_x) if synt_test_x else None

    add_val = dict()

    if validation > 0:
        train_x, val_x, train_y, val_y, train_ids, val_ids = train_test_split(train_x, train_y, train_ids,
                                                                              test_size=validation,
                                                                              random_state=random_state,
                                                                              shuffle=True, stratify=train_y)
        add_val = {
            "val": {
                "x": val_x,
                "y": val_y,
                "ids": val_ids
            }
        }

    if add_synthetic_train:
        add_val.update({
            "test_synt": {
                "x": synt_test_x,
                "y": synt_test_y,
                "ids": synt_test_ids
            }
        })

    return {
        "test_set_path": BASE_AMI_DATASET / "testset",  # used for task B evaluation
        "train": {
            "x": train_x,
            "y": train_y,
            "ids": train_ids
        },
        "test": {
            "x": test_x,
            "y": test_y,
            "ids": test_ids
        },
        **add_val
    }


def compute_metrics(y_pred, y_true, sk_classifier_name: str = None) -> dict[str, float]:
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                                             pos_label=1)
    acc = metrics.accuracy_score(y_true, y_pred)
    if sk_classifier_name:
        print(f"{sk_classifier_name} accuracy: {acc:.3f}")
        print(f"{sk_classifier_name} precision: {precision:.3f}")
        print(f"{sk_classifier_name} recall: {recall:.3f}")
        print(f"{sk_classifier_name} F1-score: {f1_score:.3f}")

    return {"f1": f1_score, "accuracy": acc, "precision": precision, "recall": recall}


def batch_list(iterable: Iterable, batch_size: int = 10) -> Iterable:
    """
    Yields batches from an iterable container

    :param iterable: elements to be batched
    :param batch_size: (max) number of elements in a single batch
    :return: generator of batches
    """
    data_len = len(iterable)
    for ndx in range(0, data_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, data_len)]


def task_b_eval(data_dict: dict, df_pred: pd.DataFrame, df_pred_synt: pd.DataFrame):
    """ Evaluate AMI TASK B """
    test_set_base_path = data_dict["test_set_path"]
    raw_data_gold, synt_data_gold, identityterms = read_gold(test_set_base_path / "test_raw_groundtruth.tsv",
                                                             test_set_base_path / "test_synt_groundtruth.tsv",
                                                             test_set_base_path / "test_identity_terms.txt",
                                                             "b")
    evaluate_task_b_singlefile(df_pred, df_pred_synt, raw_data_gold, synt_data_gold, identityterms)


def wrong_predictions(y_pred: np.ndarray, y_true: np.ndarray, threshold: float) -> pd.DataFrame:
    """ Get FP and FN """
    m_pred_hard = np.where(y_pred > threshold, 1, 0)

    wrong_indices = np.nonzero(y_true - m_pred_hard)[0]

    error: np.ndarray = y_pred[wrong_indices] - threshold
    error_type = ["fn" if e < 0 else "fp" for e in error.tolist()]  # np.where(error < 0, "fn", "fp")

    error_df = pd.DataFrame({"error": np.abs(error), "indices": wrong_indices, "type": error_type})
    error_df = error_df.sort_values(by="error", ascending=False).reset_index(drop=True)
    error_df.columns = ["error", "indices", "type"]
    return error_df
