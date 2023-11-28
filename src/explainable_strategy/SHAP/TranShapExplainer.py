import os
import time
from pathlib import Path

import numpy as np
import scipy
import shap
from matplotlib import pyplot as plt
from nltk import TweetTokenizer

from src.explainable_strategy.transhap.explainers import visualize_explanations
from src.explainable_strategy.transhap.explainers.SHAP_for_text import SHAPexplainer


class LogitModel:
    # Not used
    def __init__(self, model):
        self.inner_model = model
        self.label2id = self.inner_model.model.config.label2id
        self.id2label = self.inner_model.model.config.id2label
        self.output_shape = (max(self.label2id.values()) + 1,)
        self.model = self.inner_model.model

    def __call__(self, strings, *args, **kwargs):
        # assert not isinstance(strings, str), "shap.models.TransformersPipeline expects a list of strings not a single string!"
        output = np.zeros([len(strings)] + list(self.output_shape))
        pipeline_dicts = self.inner_model(*args, **kwargs)
        for i, val in enumerate(pipeline_dicts):
            if not isinstance(val, list):
                val = [val]
            for obj in val:
                output[i, self.label2id[obj["label"]]] = scipy.special.logit(obj["score"])
        return output


class TranShapExplainer:

    def __init__(self, model, tokenizer, target_label: str, device: str = None):
        self.__model = model  # TransformersPipeline(model, rescale_to_logits=True)
        # self.__model.model = self.__model.inner_model
        self.__tokenizer = tokenizer
        self.__device = device
        self.__target_label = target_label
        self.__target_dir = Path(os.path.join("plots", "TranShap", "transhap_{}".format(time.time())))
        os.makedirs(self.__target_dir, exist_ok=True)

    def run(self, texts: list[str], explain_ids: list[int], show: bool = True, out_label_name: str = "target"):
        out_ = self.__model([texts[i] for i in explain_ids], top_k=1)
        labels = [f"not {out_label_name}", out_label_name]

        word_tokenizer = TweetTokenizer()
        bag_of_words = list(sorted(set([xx for x in texts for xx in word_tokenizer.tokenize(x)])))

        words_dict = {0: None}
        words_dict_reverse = {None: 0}
        for h, hh in enumerate(bag_of_words):
            words_dict[h + 1] = hh
            words_dict_reverse[hh] = h + 1

        predictor = SHAPexplainer(self.__model, self.__tokenizer, words_dict, words_dict_reverse, device=self.__device, use_logits=True)
        train_dt = np.array([predictor.split_string(x) for x in np.array(texts)])
        idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

        explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50), output_names=labels)

        texts_: list[list[str]] = [predictor.split_string(x) for x in texts]
        idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

        idx_texts_to_use: list[list[int]] = np.asarray([idx_texts[a] for a in explain_ids])
        tokenized_texts_: list[list[str]] = [texts_[a] for a in explain_ids]
        shap_values = explainer.shap_values(X=idx_texts_to_use, nsamples=96, l1_reg="aic")  # nsamples="auto" should be better

        # Explain each ID
        for idx, j in enumerate(range(len(explain_ids))):
            pred = int(out_[idx][0]["label"] == self.__target_label)
            pred_score = out_[idx][0]["score"]

            len_ = len(tokenized_texts_[j])
            # Dictionary position -> number of valid indices in sample j
            d = {i: sum(x > 0 for x in shap_values[i][j, :len_]) for i, x in enumerate(shap_values)}
            # Find key (position) with maximum number of valid indices
            m = max(d, key=d.get)
            print(" ".join(tokenized_texts_[j]))
            out = shap.force_plot(explainer.expected_value[m], shap_values[m][j, :len_], tokenized_texts_[j], matplotlib=False, show=False, out_names=labels[pred])
            shap.save_html(str(self.__target_dir / f"shap_force_{explain_ids[idx]}.htm"), out, full_html=True)

            # Barplot
            text_j = tokenized_texts_[j]
            # Not sure what this is, just use model output instead
            # idx_to_use = idx_texts_to_use[j].reshape(1, -1)
            # f = predictor.predict(idx_to_use)
            # probs = scipy.special.softmax(f[0])
            # # probs = f[0]
            # pred_f = np.argmax(probs)

            visualize_explanations.joint_visualization(self.__target_dir, text_j, shap_values[pred][j, :len(text_j)], labels[pred],
                                                       pred_score, explain_ids[idx], fig_name="transhap", pred_is_logit=False)

            if show:
                plt.show()
