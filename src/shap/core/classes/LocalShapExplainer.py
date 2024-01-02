from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.shap.core.utils import build_explanation


class LocalShapExplainer:

    def __init__(self, model: object):
        self.__model = model
        self.__target_dir = Path(
            "plots") / "CategoricalShap" / f"shap_{model.__class__.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.explainer = None
        os.makedirs(self.__target_dir, exist_ok=True)

    def run_explain(self, test_data: pd.DataFrame, explain_ids: list[int], texts: list[str], targets: list[int],
                    feature_names: dict[str, str], label_names: dict[int, str],
                    top_k: int | None = None, effect_threshold: float | None = None,
                    quantized_features: pd.DataFrame | None = None) -> None:

        assert (quantized_features is None or (test_data.shape == quantized_features.shape)
                and (test_data.index == quantized_features.index).all()
                and (
                        test_data.columns == quantized_features.columns).all()), "'quantized_features' must have same shape as 'test_data'"
        values_df = test_data
        if quantized_features is not None:
            values_df = quantized_features.astype(str)  # Categorical causes problems, so use str

        assert (top_k is None) ^ (
                effect_threshold is None), "Exactly one of 'top_k' and 'effect_threshold' must be None"

        # Extract examples for explanation
        texts_to_explain = [f"ID:{i} - {texts[i]}" for i in explain_ids]
        targets_to_explain = [targets[i] for i in explain_ids]
        values_df = values_df.iloc[explain_ids, :]
        test_data = test_data.iloc[explain_ids, :]

        self.explainer = shap.TreeExplainer(self.__model)

        shap_values: np.ndarray = self.explainer.shap_values(test_data)

        shap_values_abs = np.abs(shap_values)
        shap_values_signs = np.where(shap_values < 0, -1, 1)
        shap_values_relative_change = shap_values_abs / shap_values_abs.sum(1).reshape(-1, 1) * 100
        shap_values_relative_change_df = pd.DataFrame(shap_values_relative_change, columns=test_data.columns,
                                                      index=test_data.index)

        top_features: list[list[tuple[str, float, float]]] = list()
        for i, (idx, row) in enumerate(shap_values_relative_change_df.iterrows()):
            # Multiply % importance per signs
            percentage_importance = row * shap_values_signs[i, :]
            # Sort by absolute value
            sorted_magnitudes = percentage_importance.abs().sort_values(ascending=False)
            if effect_threshold is not None:
                # Take values with % magnitude > threshold
                sorted_magnitudes = sorted_magnitudes[sorted_magnitudes > effect_threshold * 100]
            # Take original signed values
            final_features = percentage_importance[sorted_magnitudes.index]
            if top_k is not None:
                # Take top k values by magnitude
                final_features = final_features[:top_k]
            # Feature values (quantized or vanilla) for selected important features
            feature_values: pd.Series = values_df.loc[idx, final_features.index]
            # Get the cumulative sum for the contribution of all other features and summarize them in two additional rows
            # Notice that other.abs().sum() + final_features.abs().sum() == 100
            other = percentage_importance[~(percentage_importance.index.isin(final_features.index))]
            final_features = pd.concat([final_features, feature_values], axis=1)
            final_features.loc["OTHER_0", :] = [-other[other < 0].abs().sum(), other[other < 0].abs().mean()]
            final_features.loc["OTHER_1", :] = [other[other > 0].abs().sum(), other[other > 0].abs().mean()]
            # Add the feature tuples to the list for this example
            # tuple is made of: (feature_name, feature_importance, feature_value/mean feature_importance)
            top_features.append([(feat_name, *s.values.tolist()) for feat_name, s in list(final_features.iterrows())])

        # Get model probabilities of positive label
        y_probs = self.__model.predict_proba(test_data)[:, 1].tolist()
        y_preds = self.__model.predict(test_data).tolist()

        explanations: list[str] = list()
        for prob, y_pred, y_true, features, text in zip(y_probs, y_preds, targets_to_explain, top_features,
                                                        texts_to_explain):
            explanations.append(
                build_explanation(text, y_pred, y_true, prob * 100, features, feature_names, label_names))

        for ex in explanations:
            print("************************************************************")
            print(ex)

        out = shap.force_plot(self.explainer.expected_value, shap_values[0, :], test_data.iloc[0, :])
        shap.save_html(str(self.__target_dir / "shap_local.htm"), out, full_html=True)
