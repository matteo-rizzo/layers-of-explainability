import numpy as np
import pandas as pd

from src.utils.ami_2020_scripts.evaluation_submission import read_gold, evaluate_task_b_singlefile


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
