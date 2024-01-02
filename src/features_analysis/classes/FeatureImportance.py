from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import Bunch

from src.datasets.classes.Dataset import AbcDataset
from src.text_classification.utils import load_encode_dataset


def plot_permutation_importance(clf, X, y, ax, scorer: Callable, metric_name: str,
                                rskf: RepeatedStratifiedKFold | None = None) -> tuple[plt.Axes, pd.DataFrame]:
    def process_fold(train_index, test_index) -> dict[str, float]:
        x_train_data = X.iloc[train_index, :]
        x_test_data = X.iloc[test_index, :]
        y_train_data = y[train_index]
        y_test_data = y[test_index]

        clf.fit(x_train_data, y_train_data)

        return \
            permutation_importance(clf, x_test_data, y_test_data, n_repeats=5, random_state=41, n_jobs=4,
                                   scoring=scorer)[
                metric_name]

    if rskf is not None:
        # Use the validation set and make importance on k-fold validation
        results = Parallel(n_jobs=4)(
            delayed(process_fold)(train_index, test_index) for train_index, test_index in rskf.split(X, y))
        concat_importance = np.concatenate([r.importances for r in results], axis=-1)
        result = Bunch(
            importances=concat_importance,
            importances_mean=concat_importance.mean(-1),
            importances_std=concat_importance.std(-1),
        )
    else:
        # Evaluate the importance of fitted model on the test set
        result = permutation_importance(clf, X, y, n_repeats=10, random_state=41, n_jobs=4, scoring=scorer)[metric_name]
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")

    stats: pd.DataFrame = pd.DataFrame(result.importances[perm_sorted_idx].T,
                                       columns=X.columns[perm_sorted_idx]).describe()
    result.pop("importances")
    importance_mean_df = pd.DataFrame.from_dict(result, orient="columns").set_index(X.columns[perm_sorted_idx])
    stats = pd.concat([stats, importance_mean_df.T], join="inner").sort_values(by=["50%", "importances_mean"],
                                                                               ascending=False, axis="columns").T
    return ax, stats


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
        self.data_train, self.data_test = load_encode_dataset(dataset=dataset, max_scale=True)
        self.dataset_object: AbcDataset = dataset
        self.feature_names: list[str] = self.data_train.columns.tolist()
        if "y" in self.feature_names:
            self.feature_names.remove("y")
        self.output_path = Path(out_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def run_importance_test(self, clf, metric: str, decorrelate: bool = True, use_validation: bool = False,
                            use_all_data: bool = True):
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
            ax1.set_xlabel(f"Decrease in {metric} score (train set)")
            plot_permutation_importance(clf, data_test, y_test, ax2, scorer=scorer, metric_name=metric)
            ax2.set_xlabel(f"Decrease in {metric} score (test set)")
            fig.suptitle(f"Permutation importance on {self.dataset_object.__class__.__name__} features ({metric})")
            _ = fig.tight_layout()

            plt.savefig(self.output_path / f"importance_{suffix}.png")

            print(f"Baseline {metric} on test data: {clf.score(data_test, y_test):.3}")

            # plt.show()
        else:
            # Create hierarchy
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 50))
            all_data = self.data_train
            if use_all_data:
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
            dendro = hierarchy.dendrogram(dist_linkage, labels=all_data.columns.to_list(), ax=ax1, leaf_rotation=90)
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

            cluster_ids = hierarchy.fcluster(dist_linkage, 1.1, criterion="distance")
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)
            selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            selected_features_names = self.data_train.columns[selected_features]

            # Write feature clusters to JSON
            feature_subsets: dict[str, list[str]] = {
                self.data_train.columns[fs[0]]: [self.data_train.columns[f] for f in fs[1:]] for fs in
                cluster_id_to_feature_ids.values()}
            with open(self.output_path / f"features_{suffix}.json", mode="w") as fo:
                json.dump(feature_subsets, fo)

            X_train_sel = self.data_train[selected_features_names]
            X_test_sel = self.data_test[selected_features_names]

            if use_validation is False:
                # Train on the training set and compute importance on testing
                clf.fit(X_train_sel, y_train)
                print(f"Baseline {metric} on test data with feature subset: {clf.score(X_test_sel, y_test):.3}")
                rskf = None
            else:
                # Pass training set, to do k-fold permutation
                rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851231)
                X_test_sel = X_train_sel
                y_test = y_train
                suffix += "_validation"

            fig, ax = plt.subplots(figsize=(12, 20))
            ax, stats = plot_permutation_importance(clf, X_test_sel, y_test, ax, scorer=scorer, metric_name=metric,
                                                    rskf=rskf)
            stats.to_csv(self.output_path / f"importance_reduced_set_{suffix}.csv", index_label="index")
            ax.set_title(
                f"Permutation importance on selected subset of features ({'test set' if not use_validation else 'validation'})")
            ax.set_xlabel(f"Decrease in {metric} score")
            ax.figure.tight_layout()

            plt.savefig(self.output_path / f"importance_reduced_set_{suffix}.png")
            plt.clf()
