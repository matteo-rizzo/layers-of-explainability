"""
Get explanations for HuggingFace LM, using Shap and Transhap (see plots/ folder)

LM dump must be set in src/deep_learning_strategy/config.yml

"""

from __future__ import annotations

import json
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

from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.deep_learning_strategy.classes.HuggingFaceCallMeSexistDataset import HuggingFaceCallMeSexistDataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.explainable_strategy.transhap.explainers.SHAP_for_text import SHAPexplainer
from src.text_classification.utils import _complementary_indices
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "deep_learning_strategy", "config.yml"))
BATCH_SIZE = config["training"]["test_batch_size"]
TEST_MODEL_NAME = config["testing"]["model_name"]
TARGET_LABEL = config["testing"]["target_label"]
DATASET: AbcDataset = HuggingFaceCallMeSexistDataset()


def get_prediction_probabilities(model, texts, original_predictions: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return probabilities of predictions and prediction classes (int)

    @param texts: strings to use as model input
    @param model: model to use for prediction
    @param original_predictions: if predictions are supplied, it will return the probabilities of these predictions,
        otherwise it will return the predictions with the highest probability
    @return: probabilities of predicted class and the predicted class (or input predictions)
    """
    probs_1 = np.array([prediction[0]["score"] for prediction in model(texts)])
    if original_predictions is None:
        original_predictions = (probs_1 >= .5).astype(int)
    probs_pred = np.array([(p_1 if o > 0 else 1 - p_1) for p_1, o in zip(probs_1.tolist(), original_predictions.tolist())])
    return probs_pred, original_predictions


def get_importance(model, texts, indices: list[int] | None, device: str):
    labels = [f"not sexist", "sexist"]

    word_tokenizer = TweetTokenizer()
    bag_of_words = list(sorted(set([xx for x in texts for xx in word_tokenizer.tokenize(x)])))

    words_dict = {0: None}
    words_dict_reverse = {None: 0}
    for h, hh in enumerate(bag_of_words):
        words_dict[h + 1] = hh
        words_dict_reverse[hh] = h + 1

    predictor = SHAPexplainer(model, model.tokenizer, words_dict, words_dict_reverse, device=device, use_logits=True)
    train_dt = np.array([predictor.split_string(x) for x in np.array(texts)])
    idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

    explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50), output_names=labels)

    texts_: list[list[str]] = [predictor.split_string(x) for x in texts]
    idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

    if indices is not None:
        idx_texts_to_explain = idx_texts[indices, :]
    else:
        idx_texts_to_explain = idx_texts

    shap_values = explainer.shap_values(X=idx_texts_to_explain, nsamples=96, l1_reg="aic")  # nsamples="auto" should be better

    # shap_values_to_explain = np.array([shap_values[class_idx][i] for i, class_idx in enumerate(predicted_classes)])
    shap_values_to_explain: np.ndarray = shap_values[0]

    shap_importance = np.abs(shap_values_to_explain)
    shap_word_indices = pd.DataFrame(idx_texts_to_explain).map(lambda x: words_dict[x])

    return shap_importance, shap_word_indices


def evaluation_faith(pipeline, test_data, device: str, q: list[int] = None, n_explanations: int = 10) -> dict[str, tuple[float, float]]:
    """
    Compute faithfulness metrics (COMP and SUFF)

    @param device: torch device str
    @param pipeline: HF pipeline
    @param test_data: list of raw text documents
    @param q: top_k values to use for computing faithfulness
    @param n_explanations: number of explanation to use (default: 10, requires time to use all of them)
    @return: metric and its mean and std value in a dictionary
    """
    if q is None:
        q = [1, 5, 10, 20]

    if n_explanations > 0:
        idx_to_use_rnd = np.random.randint(0, len(test_data), size=n_explanations).tolist()
        test_data_reduced = [test_data[i] for i in idx_to_use_rnd]
    else:
        idx_to_use_rnd = None
        test_data_reduced = test_data

    word_importance, words = get_importance(pipeline, test_data, indices=idx_to_use_rnd, device=device)

    # Separate words with whitespace, to correctly mask them later
    # PROBLEM: punctuation remains separated by whitespaces
    tok = TweetTokenizer()
    test_data_reduced = [" ".join(s) for s in tok.tokenize_sents(test_data_reduced)]

    base_probs, predictions = get_prediction_probabilities(pipeline, test_data_reduced)

    # Avoid selecting None words as top-k (debatable)
    word_importance[words.isnull()] = -1.0

    metrics = defaultdict(list)
    for k in q:
        shap_top_k = np.argpartition(word_importance, -k, axis=1)[:, -k:]  # (samples, top_k)
        shap_not_top_k = _complementary_indices(word_importance, shap_top_k)

        suff_texts = list()
        comp_texts = list()
        for i, original_text in enumerate(test_data_reduced):
            words_top_k: list[str] = [w for w in words.iloc[i, shap_top_k[i, :]].tolist() if w is not None]
            words_not_top_k: list[str] = [w for w in words.iloc[i, shap_not_top_k[i, :]].tolist() if w is not None]

            # Replace words in the list with UNK token
            text_comp = original_text
            for word in words_top_k:
                # text_comp = text_comp.replace(word, pipeline.tokenizer.unk_token)
                # This avoids replacement of sub-words, and only matches whole tokens
                text_comp = re.sub(r"(\s|^)" + re.escape(word) + r"(\s|$)", rf"\1{pipeline.tokenizer.unk_token}\2", text_comp)

            text_suff = original_text
            for word in words_not_top_k:
                # text_suff = text_suff.replace(word, pipeline.tokenizer.unk_token)
                text_suff = re.sub(r"(\s|^)" + re.escape(word) + r"(\s|$)", rf"\1{pipeline.tokenizer.unk_token}\2", text_suff)

            suff_texts.append(text_suff)
            comp_texts.append(text_comp)

        # Now use the modified text in the pipeline
        # PROBLEM: using abs, which is not mentioned in paper
        probs_suff = get_prediction_probabilities(pipeline, suff_texts, predictions)
        scores_suff = np.abs(base_probs - probs_suff)

        probs_comp = get_prediction_probabilities(pipeline, comp_texts, predictions)
        scores_comp = np.abs(base_probs - probs_comp)

        metrics["comp"].append(float(np.mean(scores_comp)))
        metrics["suff"].append(float(np.mean(scores_suff)))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


def main():
    # Load data
    pipeline = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    test_data = DATASET.get_test_data()

    device = 0 if config["use_gpu"] else "cpu"
    metrics = evaluation_faith(pipeline.pipeline, test_data, q=[1, 5, 10], n_explanations=5, device=device)
    pprint(metrics)

    out_path = Path("dumps") / "faithfulness" / f"faith_{DATASET.__class__.__name__}_{TEST_MODEL_NAME}_{time.time()}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".json"), mode="w", encoding="utf-8") as fo:
        json.dump(metrics, fo, indent=2)


if __name__ == "__main__":
    main()
