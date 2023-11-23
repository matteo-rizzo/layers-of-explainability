from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.utils import load_encode_dataset


def plot_permutation_importance(clf, X, y, ax, scorer: Callable, metric_name: str):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=4, scoring=scorer)[metric_name]
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


def scorer_wrapper(score_fn: Callable) -> Callable:
    def _fn(estimator, X, y) -> dict[str, float]:
        y_pred = estimator.predict(X)
        return score_fn(y, y_pred)

    return _fn


class FeatureImportance:
    """
    Perform importance tests on feature sets
    """

    def __init__(self, dataset: AbcDataset, out_path: str | Path):
        self.data_train, self.data_test = load_encode_dataset(dataset=dataset, scale=True)
        self.dataset_object: AbcDataset = dataset
        self.feature_names: list[str] = self.data_train.columns.tolist()
        if "y" in self.feature_names:
            self.feature_names.remove("y")
        self.output_path = Path(out_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def run_importance_test(self, clf, metric: str, decorrelate: bool = True):
        """
        Train the classifier removing one feature at a time, recording the performance metrics for each run.
        Returns the relative importance of each removed feature, based on the performance drop/increase obtaining after removing it.
        """

        suffix = f"{self.dataset_object.__class__.__name__}_{clf.__class__.__name__}"

        y_train = self.data_train.pop("y").to_numpy()
        y_test = self.data_test.pop("y").to_numpy()

        scorer = scorer_wrapper(self.dataset_object.compute_metrics)

        if not decorrelate:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(60, 30))

            data_train = self.data_train
            data_test = self.data_test

            plot_permutation_importance(clf, data_train, y_train, ax1, scorer=scorer, metric_name=metric)
            ax1.set_xlabel("Decrease in accuracy score (train set)")
            plot_permutation_importance(clf, data_test, y_test, ax2, scorer=scorer, metric_name=metric)
            ax2.set_xlabel("Decrease in accuracy score (test set)")
            fig.suptitle(f"Permutation importance on {self.dataset_object.__class__.__name__} features ({metric})")
            _ = fig.tight_layout()

            plt.savefig(self.output_path / f"importance_{suffix}.png")

            print(f"Baseline accuracy on test data: {clf.score(data_test, y_test):.3}")

            # plt.show()
        else:
            # Create hierarchy
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 50))
            all_data = pd.concat([self.data_train, self.data_test])
            # remove constant columns
            all_data = all_data.loc[:, (all_data != all_data.iloc[0]).any()]

            corr = spearmanr(all_data).correlation

            # Ensure the correlation matrix is symmetric
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)

            # We convert the correlation matrix to a distance matrix before performing
            # hierarchical clustering using Ward's linkage.
            distance_matrix = 1 - np.abs(corr)
            dist_linkage = hierarchy.ward(squareform(distance_matrix))
            dendro = hierarchy.dendrogram(
                dist_linkage, labels=all_data.columns.to_list(), ax=ax1, leaf_rotation=90
            )
            dendro_idx = np.arange(0, len(dendro["ivl"]))

            # Plot hierarchy

            ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
            ax2.set_xticks(dendro_idx)
            ax2.set_yticks(dendro_idx)
            ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
            ax2.set_yticklabels(dendro["ivl"])
            _ = fig.tight_layout()

            plt.savefig(self.output_path / f"correlation_dendrogram_{suffix}.png")
            plt.clf()

            # Clustering

            cluster_ids = hierarchy.fcluster(dist_linkage, 1.2, criterion="distance")
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            selected_features_names = self.data_train.columns[selected_features]

            # Write feature clusters to JSON
            feature_subsets: dict[str, list[str]] = {self.data_train.columns[fs[0]]: [self.data_train.columns[f] for f in fs[1:]] for fs in cluster_id_to_feature_ids.values()}
            with open(self.output_path / f"features_{suffix}.json", mode="w") as fo:
                json.dump(feature_subsets, fo)

            X_train_sel = self.data_train[selected_features_names]
            X_test_sel = self.data_test[selected_features_names]

            clf.fit(X_train_sel, y_train)

            print(f"Baseline accuracy on test data with feature subset: {clf.score(X_test_sel, y_test):.3}")

            fig, ax = plt.subplots(figsize=(10, 20))
            plot_permutation_importance(clf, X_test_sel, y_test, ax, scorer=scorer, metric_name=metric)
            ax.set_title("Permutation importance on selected subset of features (test set)")
            ax.set_xlabel(f"Decrease in {metric} score")
            ax.figure.tight_layout()

            plt.savefig(self.output_path / f"importance_reduced_set_{suffix}.png")
            plt.clf()
