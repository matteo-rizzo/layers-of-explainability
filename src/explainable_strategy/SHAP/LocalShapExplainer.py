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

    @staticmethod
    def __helper_explanation(features, y, feature_names, label_names) -> str:
        feature_importance_text = f"Top features that contributed towards '{label_names[y]}':\n"
        explanations = list()
        for f, pf in features:
            sc = f" - ({abs(pf):.1f}%) {feature_names[f]}"
            explanations.append(sc)
        feature_importance_text += "\n".join(explanations)
        return feature_importance_text

    @staticmethod
    def build_explanation(text: str, y_pred: int, y_true: int, prob: float, features: list[tuple[str, float]], feature_names: dict[str, str], label_names: dict[int, str]) -> str:
        s = (f"Example: '{text}'\n"
             f"Predicted class: '{label_names[y_pred]}' (confidence={prob if y_pred == 1 else 100 - prob:.1f}%)\n"
             f"True class: '{label_names[y_true]}'")

        fn_pos = float.__gt__ if y_pred == 1 else float.__lt__
        fn_neg = float.__lt__ if y_pred == 1 else float.__gt__

        pred_pos_features = [f for f in features if fn_pos(f[1], 0)]  # class predicted
        pred_neg_features = [f for f in features if fn_neg(f[1], 0)]  # class not predicted

        text_s = [s]
        if pred_pos_features:
            text_s.append(LocalShapExplainer.__helper_explanation(pred_pos_features, y_pred, feature_names, label_names))
        if pred_neg_features:
            text_s.append(LocalShapExplainer.__helper_explanation(pred_neg_features, 1 - y_pred, feature_names, label_names))
        return "\n".join(text_s)

    def run_tree(self, test_data: pd.DataFrame, texts: list[str], targets: list[int], feature_names: dict[str, str], label_names: dict[int, str]):
        self.explainer = shap.TreeExplainer(self.__model)
        k = 10

        shap_values: np.ndarray = self.explainer.shap_values(test_data)

        # shap_values_df = pd.DataFrame(shap_values, columns=test_data.columns)  # - self.explainer.expected_value
        shap_values_abs = np.abs(shap_values)
        shap_values_signs = np.where(shap_values < 0, -1, 1)
        shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
        shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change, columns=test_data.columns)

        # Is it correct to consider percentage over absolute value?

        top_features: list[list[tuple[str, float]]] = list()
        for i, row in shap_values_relative_change_df.iterrows():
            values = row * shap_values_signs[i, :]
            r = values.abs().sort_values(ascending=False)
            final_features = values[r.index][:k]
            top_features.append(list(final_features.items()))

        # Get model probabilities of positive label
        y_probs = self.__model.predict_proba(test_data)[:, 1].tolist()
        y_preds = self.__model.predict(test_data).tolist()

        explanations: list[str] = list()
        for prob, y_pred, y_true, features, text in zip(y_probs, y_preds, targets, top_features, texts):
            explanations.append(self.build_explanation(text, y_pred, y_true, prob * 100, features, feature_names, label_names))

        for ex in explanations:
            print("************************************************************")
            print(ex)

        out = shap.force_plot(self.explainer.expected_value, shap_values[0, :], test_data.iloc[0, :])
        shap.save_html(str(self.__target_dir / "shap_local.htm"), out, full_html=True)

        # out = shap.force_plot(base_value=self.explainer.expected_value, shap_values=shap_values, features=test_data)

        # plt.title(f"Impact of features on predicted class")
        # plt.gcf().set_size_inches(10, 15)
        # plt.tight_layout()
        # plt.gcf().savefig(os.path.join(self.__target_dir, "shap_violin.png"), dpi=400)

        # plt.show()
        # plt.clf()
        # self.__plot_explanations_summary_tree(shap_values.expected, target_samples, output_names, show)
