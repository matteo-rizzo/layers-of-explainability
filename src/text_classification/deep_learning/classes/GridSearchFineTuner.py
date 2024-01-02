import itertools
import os
import re
import shutil
from typing import Dict

import torch
from src.deep_learning_strategy import FineTuner

from src.utils.yaml_manager import dump_yaml


class GridSearchFineTuner:

    def __init__(self, hyperparameters: Dict):
        self.__hyperparameters = hyperparameters
        self.__hyperparameters["training"]["keep_n_best_models"] = 1
        gs_params = hyperparameters["grid_search_params"]
        self.__hyp_combinations = [dict(zip(gs_params.keys(), cs)) for cs in itertools.product(*gs_params.values())]

    def run(self):
        for combination in self.__hyp_combinations:
            self.__hyperparameters["training"].update(combination)

            fine_tuner = FineTuner(self.__hyperparameters)
            fine_tuner.run()
            trainer = fine_tuner.get_trainer()

            dumps_dir = trainer.args.output_dir
            dump_yaml(data=self.__hyperparameters["training"], path=os.path.join(dumps_dir, "gs_params.yml"))

            del trainer
            torch.cuda.empty_cache()
            self.__delete_checkpoints(dumps_dir)

    @staticmethod
    def __delete_checkpoints(path: str) -> None:
        """ Remove HF-formatted checkpoints 'checkpoint-XX' from a path """

        # Iterate over all items in the directory
        for item in os.listdir(path):
            # Construct the full path
            item_path = os.path.join(path, item)
            # Check if it's a directory and matches the pattern 'checkpoint-X'
            if os.path.isdir(item_path) and re.match(r"^checkpoint-\d+$", item):
                # Delete the directory
                shutil.rmtree(item_path)
