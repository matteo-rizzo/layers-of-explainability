import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import shap
from matplotlib import pyplot as plt
from nltk import TweetTokenizer

from src.shap.core.utils import build_explanation
from src.shap.transhap import visualize_explanations
from src.shap.transhap.classes.SHAP_for_text import SHAPexplainer


class LogitModel:
    # Not used
    def __init__(self, model):
        self.inner_model = model
        self.label2id = self.inner_model.model.config.label2id
        self.id2label = self.inner_model.model.config.id2label
        self.output_shape = (max(self.label2id.values()) + 1,)
        self.model = self.inner_model.model

    def __call__(self, strings, *args, **kwargs):
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
        self.__model = model
        self.__tokenizer = tokenizer
        self.__device = device
        self.__target_label = target_label
        self.__target_dir = Path(os.path.join("plots", "TranShap", "transhap_{}".format(time.time())))
        os.makedirs(self.__target_dir, exist_ok=True)

    def run_explain(self, texts: list[str], explain_ids: list[int], targets: list[int], label_names: dict,
                    effect_threshold: float = None, top_k: int = None, out_label_name: str = "target"):
        # Extract examples for explanation
        texts_to_explain = [f"ID:{i} - {texts[i]}" for i in explain_ids]
        targets_to_explain = [targets[i] for i in explain_ids]
        out_ = self.__model(texts_to_explain, top_k=1)
        predicted_classes: list[int] = [int(idx[0]["score"] >= .5) for idx in out_]
        predicted_scores: list[float] = [idx[0]["score"] for idx in out_]
        labels = [f"not {out_label_name}", out_label_name]

        word_tokenizer = TweetTokenizer()
        bag_of_words = list(sorted(set([xx for x in texts for xx in word_tokenizer.tokenize(x)])))

        words_dict = {0: None}
        words_dict_reverse = {None: 0}
        for h, hh in enumerate(bag_of_words):
            words_dict[h + 1] = hh
            words_dict_reverse[hh] = h + 1

        predictor = SHAPexplainer(self.__model, self.__tokenizer, words_dict, words_dict_reverse, device=self.__device,
                                  use_logits=True)
        train_dt = np.array([predictor.split_string(x) for x in np.array(texts)])
        idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

        explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50),
                                         output_names=labels)

        texts_: list[list[str]] = [predictor.split_string(x) for x in texts]
        idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

        idx_texts_to_use: list[list[int]] = np.asarray([idx_texts[a] for a in explain_ids])
        tokenized_texts_: list[list[str]] = [texts_[a] for a in explain_ids]
        shap_values = explainer.shap_values(X=idx_texts_to_use, nsamples=96, l1_reg="aic")
        shap_values_to_explain: np.ndarray = shap_values[0]  # shap_values[1] + shap_values[0]

        shap_values_abs = np.abs(shap_values_to_explain)
        shap_values_signs = np.where(shap_values_to_explain < 0, -1, 1)
        shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
        shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change)
        shap_values_columns_indexes = pd.DataFrame(idx_texts_to_use)  # samples, features

        top_features: list[list[tuple[str, float, float]]] = list()
        # Explain each ID
        for j in range(len(explain_ids)):
            row = shap_values_relative_change_df.iloc[j, :]

            # Multiply % importance per signs
            percentage_importance = row * shap_values_signs[j, :]
            # Sort by absolute value
            sorted_magnitudes = percentage_importance.abs().sort_values(ascending=False)
            if effect_threshold is not None:
                # Take values with % magnitude > threshold
                sorted_magnitudes = sorted_magnitudes[sorted_magnitudes > effect_threshold * 100]
            # Take original signed values
            final_features = percentage_importance[sorted_magnitudes.index]

            # Find words with the highest importance, dropping None values (padding tokens)
            feature_values: pd.Series = pd.Series(
                [words_dict[wi] for wi in shap_values_columns_indexes.iloc[j, :].loc[final_features.index].tolist()],
                index=final_features.index).dropna()

            if top_k is not None:
                # Take top k values by magnitude (not None)
                final_features = final_features.loc[feature_values.index][:top_k]

            # Get the cumulative sum for the contribution of all other features and summarize them in two additional rows
            # Notice that other.abs().sum() + final_features.abs().sum() == 100
            other = percentage_importance[~(percentage_importance.index.isin(final_features.index))]
            final_features = pd.concat([final_features, feature_values], axis=1, join="inner")
            final_features.loc["OTHER_0", :] = [-other[other < 0].abs().sum(), other[other < 0].abs().mean()]
            final_features.loc["OTHER_1", :] = [other[other > 0].abs().sum(), other[other > 0].abs().mean()]
            # Add the feature tuples to the list for this example
            # tuple is made of: (feature_name, feature_importance, feature_value/mean feature_importance)
            top_features.append([(feat_name, *s.values.tolist()) for feat_name, s in list(final_features.iterrows())])

            pred = predicted_classes[j]
            len_ = len(tokenized_texts_[j])
            out = shap.force_plot(explainer.expected_value, shap_values_to_explain[j, :len_], tokenized_texts_[j],
                                  matplotlib=False, show=False, out_names=labels[pred])
            shap.save_html(str(self.__target_dir / f"shap_force_{explain_ids[j]}.htm"), out, full_html=True)

            print(" ".join(tokenized_texts_[j]), texts[explain_ids[j]])

        explanations: list[str] = list()
        for prob, y_pred, y_true, features, text in zip(predicted_scores, predicted_classes, targets_to_explain,
                                                        top_features, texts_to_explain):
            explanations.append(
                build_explanation(text, y_pred, y_true, prob * 100, features, None, label_names, words_mode=True))

        for ex in explanations:
            print("************************************************************")
            print(ex)

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

        predictor = SHAPexplainer(self.__model, self.__tokenizer, words_dict, words_dict_reverse, device=self.__device,
                                  use_logits=True)
        train_dt = np.array([predictor.split_string(x) for x in np.array(texts)])
        idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

        explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50),
                                         output_names=labels)

        texts_: list[list[str]] = [predictor.split_string(x) for x in texts]
        idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

        idx_texts_to_use: list[list[int]] = np.asarray([idx_texts[a] for a in explain_ids])
        tokenized_texts_: list[list[str]] = [texts_[a] for a in explain_ids]
        shap_values = explainer.shap_values(X=idx_texts_to_use, nsamples=96,
                                            l1_reg="aic")  # nsamples="auto" should be better

        # Explain each ID
        for idx, j in enumerate(range(len(explain_ids))):
            pred = int(out_[idx][0]["label"] == self.__target_label)
            pred_score = out_[idx][0]["score"]
            len_ = len(tokenized_texts_[j])
            print(" ".join(tokenized_texts_[j]))
            out = shap.force_plot(explainer.expected_value[pred], shap_values[pred][j, :len_], tokenized_texts_[j],
                                  matplotlib=False, show=False, out_names=labels[pred])
            shap.save_html(str(self.__target_dir / f"shap_force_{explain_ids[idx]}.htm"), out, full_html=True)
            text_j = tokenized_texts_[j]
            visualize_explanations.joint_visualization(self.__target_dir, text_j, shap_values[pred][j, :len_],
                                                       labels[pred],
                                                       pred_score, explain_ids[idx], fig_name="transhap",
                                                       pred_is_logit=False)

            if show:
                plt.show()
