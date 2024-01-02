from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import shap
from nltk import TweetTokenizer
from src.deep_learning_strategy import HuggingFacePipeline

from src.datasets.classes.Dataset import AbcDataset
from src.datasets.classes.HuggingFaceIMDBDataset import HuggingFaceIMDBDataset
from src.shap.transhap.classes.SHAP_for_text import SHAPexplainer
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "deep_learning_strategy", "config.yml"))
BATCH_SIZE = config["training"]["test_batch_size"]
TEST_MODEL_NAME = config["testing"]["model_name"]
TARGET_LABEL = config["testing"]["target_label"]
MODEL_MAX_LEN = config["training"]["model_max_length"]
DATASET: AbcDataset = HuggingFaceIMDBDataset()
BASE_FOLDER: str = "dumps"
suffix_str = 'CMS' if DATASET.__class__.__name__ == 'CallMeSexistDataset' else 'IMDB'


def get_prediction_probabilities(model, texts, original_predictions: np.ndarray = None) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Return probabilities of predictions and prediction classes (int)

    @param texts: strings to use as model input
    @param model: model to use for prediction
    @param original_predictions: if predictions are supplied, it will return the probabilities of these predictions,
        otherwise it will return the predictions with the highest probability
    @return: probabilities of predicted class and the predicted class (or input predictions)
    """
    probs_1 = np.array([prediction[0]["score"] for prediction in
                        model(texts, max_length=MODEL_MAX_LEN, truncation=True, padding=True)])
    if original_predictions is None:
        original_predictions = (probs_1 >= .5).astype(int)
    probs_pred = np.array(
        [(p_1 if o > 0 else 1 - p_1) for p_1, o in zip(probs_1.tolist(), original_predictions.tolist())])
    return probs_pred, original_predictions


def get_importance(model, texts, indices: list[int] | None, device: str):
    labels = ["not sexist", "sexist"]

    word_tokenizer = TweetTokenizer()
    bag_of_words = list(sorted(set([xx for x in texts for xx in word_tokenizer.tokenize(x)])))

    words_dict = {0: None}
    words_dict_reverse = {None: 0}
    for h, hh in enumerate(bag_of_words):
        words_dict[h + 1] = hh
        words_dict_reverse[hh] = h + 1

    predictor = SHAPexplainer(model, model.tokenizer, words_dict, words_dict_reverse, device=device, use_logits=True,
                              max_tokens=MODEL_MAX_LEN)
    train_dt = np.array([predictor.split_string(x) for x in np.array(texts)])
    idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt, truncate=True)

    explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50),
                                     output_names=labels)

    texts_: list[list[str]] = [predictor.split_string(x) for x in texts]
    idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len, truncate=True)

    if indices is not None:
        idx_texts_to_explain = idx_texts[indices, :]
    else:
        idx_texts_to_explain = idx_texts

    # HERE try to use more nsamples (the highest possible number)
    # 'nsamples' should be >= number of tokens/features
    shap_values = explainer.shap_values(X=idx_texts_to_explain, nsamples=256,
                                        l1_reg="aic")  # nsamples="auto" should be better

    shap_values_to_explain: np.ndarray = shap_values[0]
    shap_word_indices = pd.DataFrame(idx_texts_to_explain).map(lambda x: words_dict[x])

    return shap_values_to_explain, shap_word_indices


def evaluation_faith(pipeline, test_data, device: str, q_perc: list[int] = None, n_explanations: int = 10) -> dict[
    str, tuple[float, float]]:
    """
    Compute faithfulness metrics (COMP and SUFF)

    @param device: torch device str
    @param pipeline: HF pipeline
    @param test_data: list of raw text documents
    @param q_perc: k% values to use for computing faithfulness
    @param n_explanations: amount of explanation to use (default: 10, -1 = all, but requires time)
    @return: metric and its mean and std value in a dictionary
    """
    if q_perc is None:
        q_perc = [1, 5, 10, 20]
    q_perc: list[float] = [q / 100 for q in q_perc]

    if n_explanations > 0:
        np.random.seed(3)
        idx_to_use_rnd = np.random.randint(0, len(test_data), size=n_explanations).tolist()
        test_data_reduced = [test_data[i] for i in idx_to_use_rnd]
    else:
        idx_to_use_rnd = None
        test_data_reduced = test_data

    word_importance, words = get_importance(pipeline, test_data, indices=idx_to_use_rnd, device=device)

    # Separate words with whitespace, to correctly mask them later
    tok = TweetTokenizer()
    tok_text = tok.tokenize_sents(test_data_reduced)
    test_data_reduced = [" ".join(s) for s in tok_text]

    base_probs, predictions = get_prediction_probabilities(pipeline, test_data_reduced)

    # Consider the direction of contribution w.r.t. predicted class
    signs = np.where(predictions > 0, 1.0, -1.0).reshape(-1, 1)  # (n_samples, 1)
    # Features with positive values will be the ones contributing to the predicted class
    shap_values_signed = word_importance * signs

    # Avoid selecting None words as top-k (debatable)
    shap_values_signed[words.isnull()] = -1000.0

    metrics = defaultdict(list)
    for k_perc in q_perc:
        suff_texts = list()
        comp_texts = list()
        for i, original_text in enumerate(test_data_reduced):
            # Find contributing words in current examples (positive shap values)
            positive_words = np.where(shap_values_signed[i, :] > 0, 1, 0)
            positive_words_idx = np.asarray(positive_words == 1).nonzero()[0]
            # means.append(len(positive_words_idx))

            # Compute number of words to remove
            k = math.ceil(k_perc * len(positive_words_idx))

            # Select word indices (k < len(positive_words_idx), so it can't remove negative ones)
            shap_top_k = np.argpartition(shap_values_signed[i, :], -k)[-k:]  # (top_k,)
            shap_not_top_k = np.setdiff1d(positive_words_idx, shap_top_k)

            # Select top-k words to be removed
            words_top_k: list[str] = [w for w in words.iloc[i, shap_top_k].tolist() if w is not None]
            words_not_top_k: list[str] = [w for w in words.iloc[i, shap_not_top_k].tolist() if w is not None]

            # Replace words in the list with UNK token
            text_comp = original_text
            for word in words_top_k:
                # text_comp = text_comp.replace(word, pipeline.tokenizer.unk_token)
                # This avoids replacement of sub-words, and only matches whole tokens
                text_comp = re.sub(r"(\s|^)" + re.escape(word) + r"(\s|$)", rf"\1{pipeline.tokenizer.unk_token}\2",
                                   text_comp)

            text_suff = original_text
            for word in words_not_top_k:
                text_suff = re.sub(r"(\s|^)" + re.escape(word) + r"(\s|$)", rf"\1{pipeline.tokenizer.unk_token}\2",
                                   text_suff)

            suff_texts.append(text_suff)
            comp_texts.append(text_comp)

        # print(np.average(means))
        # Now use the modified text in the pipeline
        probs_suff, _ = get_prediction_probabilities(pipeline, suff_texts, predictions)
        scores_suff = base_probs - probs_suff

        probs_comp, _ = get_prediction_probabilities(pipeline, comp_texts, predictions)
        scores_comp = base_probs - probs_comp

        metrics["comp"].append(scores_comp)
        metrics["suff"].append(scores_suff)

    metrics = {k: np.stack(v, axis=-1) for k, v in metrics.items()}  # k: (samples, q)

    # Write runs to numpy files
    out_path = Path(BASE_FOLDER) / "faithfulness"
    out_path.mkdir(parents=True, exist_ok=True)
    for m, v in metrics.items():
        np.save(out_path / f"{m}_lm_{suffix_str}.npy", v)

    return {k: (float(v.mean()), float(v.std(axis=1).mean())) for k, v in metrics.items()}


def main():
    # Load data
    pipeline = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    test_data = DATASET.get_test_data()

    device = 0 if config["use_gpu"] else "cpu"
    metrics = evaluation_faith(pipeline.pipeline, test_data, q_perc=[1, 5, 10, 20, 50, 75], n_explanations=-1,
                               device=device)
    pprint(metrics)

    out_path = Path(
        BASE_FOLDER) / "faithfulness" / f"faith_{DATASET.__class__.__name__}_{TEST_MODEL_NAME}_{time.time()}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".json"), mode="w", encoding="utf-8") as fo:
        json.dump(metrics, fo, indent=2)


if __name__ == "__main__":
    main()
