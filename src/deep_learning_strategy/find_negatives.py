from pathlib import Path

import numpy as np
import pandas as pd
from src.ami2020.dataset import train_val_test, wrong_predictions, compute_metrics
from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.deep_learning_strategy.pipeline import HugghingFacePipeline, deep_preprocessing

if __name__ == "__main__":
    out = Path("dumps") / "nlp_models" / "error_reports" / "DL"

    config: dict = load_yaml("src/nlp/params/deep_learning_strategy.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]
    add_synthetic: bool = True  # config["add_synthetic"]

    print("*** Predicting misogyny ")
    pipe_m = HugghingFacePipeline(config["testing"]["task_m_model_name"], device=0 if use_gpu else "cpu", batch_size=bs,
                                  top_k=None)
    dataset_m = train_val_test(target="M", add_synthetic_train=add_synthetic, preprocessing_function=deep_preprocessing)
    x_data = dataset_m["test"]["x"] + dataset_m["test_synt"]["x"]
    y_data = dataset_m["test"]["y"] + dataset_m["test_synt"]["y"]

    raw_results = pipe_m(x_data)
    # Rework results, make a list of dicts with {label: score}
    r_dict: list[dict[str, float]] = [dict([tuple(a.values()) for a in row]) for row in raw_results]

    other_label: str = [k for k in r_dict[0].keys() if k != target_label][0]
    results = [1 if e[target_label] > e[other_label] else 0 for e in r_dict]
    print("Metrics on RAW and SYNTHETIC datasets combined")
    compute_metrics(y_pred=results, y_true=y_data, sk_classifier_name=pipe_m.model.__class__.__name__)

    # Predict scores with the model on test data
    m_scores = [e[target_label] for e in r_dict]
    assert [1 if e > .5 else 0 for e in m_scores] == results, "Results and scores do not match"

    # Tokenize dataset, then extract non-zero entries from vectorizer to get the effective features (words) that are considered
    # TODO

    # Find out which are wrong predictions
    error_df: pd.DataFrame = wrong_predictions(y_pred=np.asarray(m_scores, dtype=float),
                                               y_true=np.asarray(y_data, dtype=int), threshold=.5)

    # Concatenate to wrong samples its input text
    input_df = pd.DataFrame({"original_text": x_data})  # "features": x_cleaned})
    input_df = input_df.iloc[error_df["indices"], :].reset_index(drop=True)
    error_df = pd.concat([error_df, input_df], axis=1)  # concat columns (same number of rows)

    # Separate errors in FP and FN and write reports to file
    error_df_fp = error_df[error_df["type"] == "fp"]
    error_df_fn = error_df[error_df["type"] == "fn"]

    out.mkdir(parents=True, exist_ok=True)
    error_df_fp.to_csv(out / "errors_fp.csv", index=False)
    error_df_fn.to_csv(out / "errors_fn.csv", index=False)
