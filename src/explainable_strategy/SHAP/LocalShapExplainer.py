from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap


class LocalShapExplainer:

    def __init__(self, model: object):
        self.__model = model
        self.__target_dir = Path("plots") / "CategoricalShap" / f"shap_{model.__class__.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.explainer = None
        os.makedirs(self.__target_dir, exist_ok=True)

    def run(self, train_data: pd.DataFrame, test_data: pd.DataFrame, output_names: list | None = None, show: bool = True):
        # What number of background samples?
        x_train_summary = shap.kmeans(train_data, 50)

        target_samples = test_data
        fx = self.__model.predict_proba

        self.explainer = shap.KernelExplainer(fx, x_train_summary, output_names=output_names)

        shap_values = self.explainer.shap_values(target_samples, nsamples=1000)
        self.__plot_explanations_summary(shap_values, target_samples, output_names, show, target_index=1)

    def run_tree(self, test_data: pd.DataFrame, output_names: list | None = None, show: bool = True):
        self.explainer = shap.TreeExplainer(self.__model)
        k = 10

        shap_values: np.ndarray = self.explainer.shap_values(test_data)

        shap_values_df = pd.DataFrame(shap_values, columns=test_data.columns)  # - self.explainer.expected_value
        shap_values_abs = np.abs(shap_values)
        shap_values_signs = np.where(shap_values < 0, -1, 1)
        shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
        shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change, columns=test_data.columns)

        # TODO: WIP

        top_features = list()
        for i, row in shap_values_relative_change_df.iterrows():
            values = row * shap_values_signs[i, :]
            r = values.abs().sort_values(ascending=False)
            final_features = values[r.index][:k]
            top_features.append(list(final_features.items()))

        out = shap.force_plot(self.explainer.expected_value, shap_values[0, :], test_data.iloc[0, :])
        shap.save_html(str(self.__target_dir / "shap_local.htm"), out, full_html=True)

        pass

        # out = shap.force_plot(base_value=self.explainer.expected_value, shap_values=shap_values, features=test_data)

        # plt.title(f"Impact of features on predicted class")
        # plt.gcf().set_size_inches(10, 15)
        # plt.tight_layout()
        # plt.gcf().savefig(os.path.join(self.__target_dir, "shap_violin.png"), dpi=400)

        # plt.show()
        # plt.clf()
        # self.__plot_explanations_summary_tree(shap_values.expected, target_samples, output_names, show)
