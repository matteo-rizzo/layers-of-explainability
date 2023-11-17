from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.deep_learning_strategy.classes.Dataset import AbcDataset
from src.text_classification.classes.training.TrainingModelUtility import TrainingModelUtility
from src.text_classification.utils import load_encode_dataset


class FeatureAblator:
    """
    Perform ablation tests on feature sets, whatever it means.
    """

    def __init__(self, dataset: AbcDataset, train_config: dict, classifier_type: type, out_path: str | Path, classifier_kwargs: dict | None = None):
        self.data_train, self.data_test = load_encode_dataset(dataset=dataset, scale=True)
        self.dataset_object: AbcDataset = dataset
        self.feature_names: list[str] = self.data_train.columns.tolist()
        if "y" in self.feature_names:
            self.feature_names.remove("y")
        self.config: dict = train_config
        self.classifier_class: type = classifier_type
        self.classifier_kwargs: dict = classifier_kwargs if classifier_kwargs is not None else dict()
        self.output_path = Path(out_path)

        self.training_utility = TrainingModelUtility(self.config, self.classifier_class, self.classifier_kwargs)

    def run_ablation(self, feature_set=None):
        """
        Train the classifier removing one feature at a time, recording the performance metrics for each run.
        Returns the relative importance of each removed feature, based on the performance drop/increase obtaining after removing it.
        """
        self.training_utility.train_classifier(self.data_train.copy())
        baseline_metrics = self.training_utility.evaluate(self.data_test.copy(), self.dataset_object.compute_metrics, print_metrics=False)
        metric_names: list[str] = [m[0] for m in sorted(list(baseline_metrics.items()), key=lambda x: x[0])]

        all_metrics: dict[str, list[float]] = dict()
        all_metrics["base"] = [baseline_metrics[k] for k in metric_names]
        for f in tqdm(self.feature_names, desc="Feature removal"):
            data_train = self.data_train.copy().drop(columns=f)
            data_test = self.data_test.copy().drop(columns=f)
            self.training_utility.train_classifier(data_train)
            metrics = self.training_utility.evaluate(data_test, self.dataset_object.compute_metrics, print_metrics=False)
            # Select metrics in correct order and add it to the dictionary {feature_removed -> results}
            metric_list: list[float] = [metrics[k] for k in metric_names]
            all_metrics[f] = metric_list

        all_metrics_df: pd.DataFrame = pd.DataFrame.from_dict(all_metrics, orient="index", columns=metric_names).round(5)
        baseline_series: pd.Series = all_metrics_df.loc["base", :]
        # Sort both dataframes by drop descending
        all_metrics_df_ratio: pd.DataFrame = ((baseline_series - all_metrics_df) / baseline_series).sort_values(by=metric_names, ascending=False)
        all_metrics_df = all_metrics_df.loc[all_metrics_df_ratio.index, :]

        self.output_path.mkdir(exist_ok=True, parents=True)
        with pd.ExcelWriter(self.output_path / f"ablation_features_{self.dataset_object.__class__.__name__}.ods", engine="odf") as exc_writer:
            all_metrics_df_ratio.to_excel(exc_writer, sheet_name="drop_ratio", index=True)
            all_metrics_df.to_excel(exc_writer, sheet_name="metrics", index=True)
