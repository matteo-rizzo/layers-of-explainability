from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt


class CategoricalShapExplainer:

    def __init__(self, model: object):
        self.__model = model
        self.__target_dir = Path("plots") / "CategoricalShap" / f"shap_{model.__class__.__name__}_{datetime.datetime.now().strftime('%Y-%M-%d_%H:%m:%s')}"
        self.__explainer = None
        os.makedirs(self.__target_dir, exist_ok=True)

    def __plot_explanations_summary(self, shap_values: np.ndarray, data: pd.DataFrame, out_names: list, show: bool, target_index: int = 1):
        # FORCE PLOT
        out = shap.force_plot(self.__explainer.expected_value[target_index], shap_values[target_index], data.values,
                              feature_names=data.columns, matplotlib=False, out_names=out_names[target_index],
                              contribution_threshold=0.1)
        #  Only features that the magnitude of their shap value is larger than contribution_threshold * (sum of all abs shap values) will be displayed
        shap.save_html(str(self.__target_dir / "shap_force.htm"), out, full_html=True)

        # SCATTERPLOT
        shap.summary_plot(
            shap_values=shap_values[target_index],
            features=data.values,
            feature_names=data.columns,
            plot_type="violin",
            # color="coolwarm",
            show=False,
            max_display=50,
            plot_size="auto",
            class_names=out_names[target_index]
        )

        plt.title(f"Impact of features on predicted class")
        plt.gcf().set_size_inches(10, 15)
        plt.tight_layout()
        plt.gcf().savefig(os.path.join(self.__target_dir, "shap_violin.png"), dpi=400)
        if show:
            plt.show()
        plt.clf()

        # BARPLOT
        shap.summary_plot(
            shap_values=shap_values,
            features=data.values,
            feature_names=data.columns,
            alpha=0,
            plot_type="bar",
            show=False,
            max_display=50,
            plot_size="auto",
            class_names=out_names
        )

        plt.title(f"Multi-output Barplot")
        plt.gcf().set_size_inches(10, 15)
        plt.tight_layout()
        plt.gcf().savefig(os.path.join(self.__target_dir, "shap_bar.png"), dpi=400)
        if show:
            plt.show()
        plt.clf()

    def run(self, train_data: pd.DataFrame, test_data: pd.DataFrame, output_names: list | None = None, show: bool = True, make_tensors: bool = False):
        x_train_summary = shap.kmeans(train_data, 50)

        target_samples = test_data
        # if make_tensors:
        #     TO BE DONE: to work with skorch we need tensors, but SHAP things break with tensors as output
        #     x_train_summary.data = torch.as_tensor(x_train_summary.data, dtype=torch.float32)
        #     x_train_summary.weights = torch.as_tensor(x_train_summary.weights, dtype=torch.float32)
        #     target_samples = torch.as_tensor(target_samples.values, dtype=torch.float32)

        self.__explainer = shap.KernelExplainer(self.__model.predict_proba, x_train_summary, output_names=output_names)

        shap_values = self.__explainer.shap_values(target_samples, nsamples=1000)
        self.__plot_explanations_summary(shap_values, target_samples, output_names, show, target_index=1)
