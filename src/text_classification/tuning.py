from __future__ import annotations

import copy
import itertools
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch.cuda
from tqdm import tqdm

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.extraction import document_extraction
from topic_extraction.utils import load_yaml, dump_yaml, vector_rejection
import datetime as dt

# K-MEANS best (specter): 15
# HDBSCAN best (specter): 'min_samples': 3, 'min_cluster_size': 15, 'metric': 'euclidean', 'cluster_selection_method': 'eom'

USE_PASS_1_EMBEDDINGS = True  # use embeddings from guided topic modeling from the first model
ORTHOGONAL_SUBJECTS = True  # remove topics from the first model
EMB_PATH = "dumps/embeddings/allenai-specter.npy"
TEXT_COMPOSITION = ["t", "a", "k"]


def get_all_combination_indices(params: dict[str, list]) -> list[tuple]:
    """
    Get tuples with the combination of indices for a set of hyperparameters.

    :param params: dictionary of parameters, with name and list of values (no nested parameters)
    :return: list of tuples with all indices combinations
    """
    argument_name, argument_values = zip(*params.items())
    # Create list of indices 0...n-1 for each list of n values
    argument_values_indices = [list(range(len(a))) for a in argument_values]
    # Create tuples with all possible indices combinations
    arguments_index_combinations: list[tuple] = list(itertools.product(*argument_values_indices))
    return arguments_index_combinations


def tuning(normalize: bool, gs_config: Path | str):
    """
    Perform grid search over BERTopic hyperparameters.

    :param normalize: whether to use embedding L2 normalization
    :param gs_config: GS parameters. Should be organized as bertopic2 configuration file. If more than 1 methods are supported
        the config file should have one entry for each "choice". If only 1 possibility is present, then it should have parameters
        directly with no entry name for a module.
    """
    docs = document_extraction(TEXT_COMPOSITION)

    text_type_suffix = "_" + "".join(TEXT_COMPOSITION)
    suffix: str = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ("" if not normalize else "_norm") + text_type_suffix

    pl_path = Path("plots") / "validation"
    result_path = "gs_results_" + suffix

    grid_search_params = load_yaml(gs_config)
    base_params = load_yaml(f"topic_extraction/config/bertopic2{text_type_suffix}.yml")
    base_model_params = deepcopy(base_params["model"])
    tunable_parameters: list[str] = list(set(base_model_params.keys()) & set(grid_search_params.keys()))

    block_parameters: dict[str, list] = defaultdict(list)

    # For each module/block, generate all the combination of hyperparameters
    print("Preparing grid search...")
    for k in tqdm(tunable_parameters):
        k_choice: str | None = base_model_params[k].get("choice", None)
        if k_choice:
            grid_search_params[k] = grid_search_params[k][k_choice]

        grid_search_params[k]: dict[str, list]
        block_k_argument_names: list[str] = list(grid_search_params[k].keys())
        block_k_argument_values: list[list] = list(grid_search_params[k].values())

        # Get all combinations of parameters for block k
        param_k_combination_indices = get_all_combination_indices(grid_search_params[k])

        for index_combination in param_k_combination_indices:
            # index_combination is a tuple of indexes, 1 for each argument of block k
            combination_values: tuple = tuple(argument_values[i_comb] for argument_values, i_comb in zip(block_k_argument_values, index_combination))
            # Kwargs for a combination of the arguments of block k
            grid_search_combined_kv_args = dict(zip(block_k_argument_names, combination_values))
            block_parameters[k].append(grid_search_combined_kv_args)

    # Combine all hyperparameters in each module/block
    all_blocks_combination_indices = get_all_combination_indices(block_parameters)
    block_keys: list[str] = list(block_parameters.keys())
    block_values: list[list] = list(block_parameters.values())

    # For each module, start GRID SEARCH
    print("Starting grid search...")
    best_score: float = -1.0
    best_parameters: dict | None = None
    all_results = list()
    best_full_config: dict | None = None
    pl_path.mkdir(exist_ok=True, parents=True)

    if USE_PASS_1_EMBEDDINGS:
        embeddings = np.load("dumps/embeddings/gtm_embeddings.npy")
    elif Path(EMB_PATH).is_file():
        embeddings = np.load(EMB_PATH)
    else:
        raise ValueError("Embeddings were not loaded.")

    if ORTHOGONAL_SUBJECTS:
        theme_embeddings = np.load("dumps/embeddings/theme_embeddings.npy")
        embeddings = vector_rejection(embeddings, theme_embeddings)

    embeddings_c = copy.deepcopy(embeddings)

    for block_index_combination in tqdm(all_blocks_combination_indices):
        block_combination_values = tuple(block_argument_values[block_i_comb] for block_argument_values, block_i_comb in zip(block_values, block_index_combination))
        kv_args: dict[str, dict] = dict(zip(block_keys, block_combination_values))

        run_config = deepcopy(base_params)
        last_tested_arguments = dict()
        for k in tunable_parameters:
            k_choice: str | None = base_model_params[k].get("choice", None)
            if k_choice:
                run_config["model"][k]["params"][k_choice].update(**kv_args[k])
            else:
                run_config["model"][k]["params"].update(**kv_args[k])
            last_tested_arguments[k] = kv_args[k]

        extractor = BERTopicExtractor(plot_path=pl_path)
        extractor.prepare(config=run_config)

        topics, _ = extractor.train(docs, normalize=normalize, embeddings=embeddings)
        assert np.array_equal(embeddings_c, embeddings), "Embeddings changed!"

        # DBCV score
        bdcv_score = extractor._topic_model.hdbscan_model.relative_validity_
        n_clusters = max(topics) + 1
        n_outliers: int = len([t for t in topics if t < 0])

        # Score prioritize results with a good ratio between n_cluster and outliers, and have a good DBCV score
        score = (bdcv_score + (n_clusters / (math.log2(n_outliers + 1) + 1))) / 2
        # Flatten arguments to fit them in dataframe, and remove parameters that were not tuned
        flattened_args: dict[str, dict] = {f"{block_name}_{strategy_name}": strategy_config for block_name, block_conf in last_tested_arguments.items()
                                           for strategy_name, strategy_config in block_conf.items()}
        ext_args = {**flattened_args, "n_outliers": n_outliers, "n_clusters": n_clusters, "bdcv": bdcv_score, "score": score}
        # if we got a better score, store it and the parameters
        if score > best_score:
            best_score = score
            best_parameters = ext_args
            best_full_config = run_config
        all_results.append(ext_args)

        torch.cuda.empty_cache()

        pd.DataFrame.from_records(all_results).sort_values(by="score", ascending=False).to_csv(pl_path / f"{result_path}.csv", index=False)

    print("Best score: {:.4f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))

    dfr = pd.DataFrame.from_records(all_results).sort_values(by="score", ascending=False)

    print(f"Num clusters in best run: {dfr.iloc[0]['n_clusters']}")

    # Save complete results and the best configuration in YAML
    dfr.to_csv(pl_path / f"{result_path}.csv", index=False)
    dump_yaml(best_full_config, pl_path / f"best_config_{suffix}.yml")


if __name__ == "__main__":
    tuning(normalize=False, gs_config="topic_extraction/config/model_selection.yml")
