from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from src.utils.yaml_manager import load_yaml


def rank_runs(path: str | Path, metric_name: str, top_k: int) -> None:
    """ Ranking utility to find the best hyperparameters """
    gs_config: dict = load_yaml("src/nlp/params/deep_learning_strategy.yml")["grid_search_params"]
    gs_params = list(gs_config.keys())

    path = Path(path)

    all_results = list()

    # Iterate over all items in the directory
    for item in os.listdir(path):
        # Construct the full path
        item_path = path / item
        # Check if it's a directory and matches the pattern 'checkpoint-X'
        results_path = item_path / "eval_results.json"
        config_path = item_path / "gs_params.yml"
        if item_path.is_dir() and results_path.is_file() and config_path.is_file():
            match_num = re.search(r"_(\d+)$", item)
            run_id: int | None
            if match_num is not None:
                run_id = int(match_num.groups()[0])
            else:
                run_id = None

            with open(results_path, mode="r", encoding="utf-8") as f:
                results = json.load(f)
            config = load_yaml(config_path)

            metric_value = results[f"eval_{metric_name}"]
            epoch_value = results["epoch"]

            values = [run_id, metric_value, epoch_value, *[config[p] for p in gs_params]]
            all_results.append(values)

    all_results = pd.DataFrame(all_results, columns=["id", metric_name, "epochs", *gs_params]).set_index(
        "id").sort_values(by=metric_name, ascending=False)

    all_results.to_csv(path / "gs_ranking.csv")

    print(all_results.iloc[:top_k, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to quickly obtain ranking from Grid Search results. Will create a result file in {"
                    "path}/gs_ranking.csv.")
    # parser.add_argument('--help', action='help', help=)
    parser.add_argument("path", type=str, help="Path to folder with Grid Search dumps")
    parser.add_argument("--top_k", "-k", nargs="?", type=int, default=10,
                        help="Best k results to print to console (the rest will be in CSV file)")
    parser.add_argument("--metric", "-m", nargs="?", type=str, default="f1",
                        help="Metric name to sort by, should reflect naming used in dumps")
    args = parser.parse_args()

    rank_runs(args.path, args.metric, args.top_k)
