import itertools
from pathlib import Path
from typing import Dict

import torch

from src.deep_learning_strategy.classes.FineTuner import FineTuner
from src.deep_learning_strategy.utils import delete_checkpoints
from src.utils.yaml_manager import dump_yaml


class GridSearchFineTuner:

    def __init__(self, hyperparameters: Dict):
        hyperparameters["training"]["keep_n_best_models"] = 1
        gs_params = hyperparameters["grid_search_params"]
        hyp_combinations = [dict(zip(gs_params.keys(), cs)) for cs in itertools.product(*gs_params.values())]

        for combination in hyp_combinations:
            hyperparameters["training"].update(combination)

            fine_tuner = FineTuner(hyperparameters)
            fine_tuner.finetune()
            trainer = fine_tuner.get_trainer()

            dumps_dir = Path(trainer.args.output_dir)
            dump_yaml(hyperparameters["training"], dumps_dir / "gs_params.yml")
            delete_checkpoints(dumps_dir)

            del trainer
            torch.cuda.empty_cache()
