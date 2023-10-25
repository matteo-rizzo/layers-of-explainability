import os
import time
from typing import Dict, Set, List, Tuple

import numpy as np
import shap
from matplotlib import pyplot as plt
from nltk import TweetTokenizer

from src.explainable_strategy.transhap.explainers import visualize_explanations
from src.explainable_strategy.transhap.explainers.SHAP_for_text import SHAPexplainer


class TranShapExplainer:

    def __init__(self, model, tokenizer, target_label: str, device: str = None):
        self.__model = model
        self.__tokenizer = tokenizer
        self.__device = device
        self.__target_dir = os.path.join("src", "deep_learning_strategy", "plots", "transhap_{}".format(time.time()))
        os.makedirs(self.__target_dir, exist_ok=True)

    @staticmethod
    def __make_bag_of_words(corpus: List[str]) -> Set:
        return set([xx for x in corpus for xx in TweetTokenizer().tokenize(x)])

    def __make_words_dicts(self, corpus) -> tuple[Dict, Dict]:
        words_dict, words_dict_reverse = {0: None}, {None: 0}
        bag_of_words = self.__make_bag_of_words(corpus)
        for h, hh in enumerate(bag_of_words):
            words_dict[h + 1], words_dict_reverse[hh] = hh, h + 1
        return words_dict, words_dict_reverse

    def __make_predictor(self, corpus: List[str]) -> SHAPexplainer:
        words_dict, words_dict_reverse = self.__make_words_dicts(corpus)
        return SHAPexplainer(self.__model, self.__tokenizer, words_dict, words_dict_reverse, device=self.__device)

    def __make_explainer(self, corpus: List[str], predictor: SHAPexplainer) -> Tuple:
        train_dt = np.array([predictor.split_string(x) for x in np.array(corpus)])
        idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)
        explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50))
        corpus = [predictor.split_string(x) for x in corpus]
        idx_corpus, _ = predictor.dt_to_idx(corpus, max_seq_len=max_seq_len)
        return explainer, idx_corpus

    def __plot_explanations_by_id(self, token_corpus: List[str], idx_corpus: np.ndarray, shap_v: np.ndarray,
                                  num_ids: int, predictor, explainer, show: bool):

        for idx, j in enumerate(range(num_ids)):
            length = len(token_corpus[j])
            d = {i: sum(x > 0 for x in shap_v[i][j, :length]) for i, x in enumerate(shap_v)}
            m = max(d, key=d.get)
            print(" ".join(token_corpus[j]))
            shap.force_plot(explainer.expected_value[m], shap_v[m][j, :length], token_corpus[j], matplotlib=True)

            text_j = token_corpus[j]
            f = predictor.predict(idx_corpus[j].reshape(1, -1))
            predicted_f = np.argmax(f[0])

            visualize_explanations.joint_visualization(self.__target_dir, text_j, shap_v[predicted_f][j, :len(text_j)],
                                                       ["Non-hateful", "Hateful"][int(predicted_f)],
                                                       f[0][predicted_f], idx)
            if show:
                plt.show()

            plt.savefig(os.path.join(self.__target_dir, "transhap_{idx}_{i}.png"), dpi=400)
            plt.clf()

    def run(self, corpus: List[str], explain_ids: List[int], show: bool = True):
        predictor = self.__make_predictor(corpus)
        explainer, idx_corpus = self.__make_explainer(corpus, predictor)

        token_corpus = [corpus[a] for a in explain_ids]
        idx_corpus = np.asarray([idx_corpus[a] for a in explain_ids])
        shap_v = explainer.shap_values(X=idx_corpus, nsamples=64, l1_reg="aic")

        self.__plot_explanations_by_id(token_corpus, idx_corpus, shap_v, len(explain_ids), predictor, explainer, show)
