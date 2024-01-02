import argparse
import os

from src.text_classification.deep_learning.classes.FineTuner import FineTuner
from src.text_classification.deep_learning.classes.GridSearchFineTuner import GridSearchFineTuner
from src.utils.setup import PATH_TO_CONFIG, RANDOM_SEED, make_deterministic, print_namespace
from src.utils.yaml_manager import load_yaml

DO_GRID_SEARCH = False


def main(ns: argparse.Namespace):
    config = load_yaml(os.path.join(PATH_TO_CONFIG))
    config["dataset"] = ns.dataset
    GridSearchFineTuner(config).run() if ns.do_grid_search else FineTuner(config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--do_grid_search", type=bool, default=DO_GRID_SEARCH)
    parser.add_argument('--dataset', type=str, default="AMI2018")
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
