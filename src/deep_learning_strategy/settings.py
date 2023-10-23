import argparse
import os
import re

import numpy as np
import torch

SEPARATOR = {"stars": "".join(["*"] * 100), "dashes": "".join(["-"] * 100), "dots": "".join(["."] * 100)}


# --- Determinism (for reproducibility) ---

def make_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


# --- Device (cpu or cuda:n) ---

DEVICE_TYPE = "cuda:0"


def get_device() -> torch.device:
    if DEVICE_TYPE == "cpu":
        print("\n Running on device 'cpu' \n")
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", DEVICE_TYPE):
        if not torch.cuda.is_available():
            print("\n WARNING: running on cpu since device {} is not available \n".format(DEVICE_TYPE))
            return torch.device("cpu")

        print("\n Running on device '{}' \n".format(DEVICE_TYPE))
        return torch.device(DEVICE_TYPE)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(DEVICE_TYPE))


DEVICE = get_device()


# --- Print bash input parameters ---

def print_namespace(namespace: argparse.Namespace):
    print("\n" + SEPARATOR["dashes"])
    print("\n\t *** INPUT NAMESPACE PARAMETERS *** \n")
    for arg in vars(namespace):
        print(("\t - {} " + "".join(["."] * (25 - len(arg))) + " : {}").format(arg, getattr(namespace, arg)))
    print("\n" + SEPARATOR["dashes"] + "\n")


# --- GLOBAL VARIABLES ---

RANDOM_SEED = 0
PATH_TO_CONFIG = os.path.join("src", "deep_learning_strategy", "config.yml")
