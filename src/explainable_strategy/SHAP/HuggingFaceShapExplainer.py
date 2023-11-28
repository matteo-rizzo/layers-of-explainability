import os
import time
from typing import List

import numpy as np
import shap
from matplotlib import pyplot as plt
from shap.models import TransformersPipeline


class HuggingFaceShapExplainer:

    def __init__(self, pipeline, tokenizer, target_label: str):
        self.__pipeline = TransformersPipeline(pipeline, rescale_to_logits=True)
        self.__tokenizer = tokenizer
        self.__target_label = target_label
        self.__target_dir = os.path.join("plots", "Shap", "shap_{}".format(time.time()))
        os.makedirs(self.__target_dir, exist_ok=True)

    def __plot_explanations_by_sentence(self, corpus: List[str], shap_values: np.ndarray, show: bool):
        for i, text in enumerate(corpus):
            shap.plots.bar(shap_values[i, :, self.__target_label], max_display=20, show=False)
            plt.title(f"Sentence: {text}")
            plt.gcf().set_size_inches(10, 14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.__target_dir, f"shap_sentence_barplot_{i}.png"), dpi=400)
            if show:
                plt.show()
            # plt.clf()

    def __plot_explanations_summary(self, shap_values: np.ndarray, show: bool):
        shap.plots.bar(shap_values[:, :, self.__target_label].mean(0), max_display=20, show=False)
        plt.title("Mean relative importance of tokens for classification")
        plt.gcf().set_size_inches(10, 14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.__target_dir, "shap_mean_barplot.png"), dpi=400)
        if show:
            plt.show()
        # plt.clf()

    def run(self, corpus: list[str], show: bool = True):
        shap_values = shap.Explainer(self.__pipeline, self.__tokenizer)(corpus)
        self.__plot_explanations_by_sentence(corpus, shap_values, show)
        self.__plot_explanations_summary(shap_values, show)
