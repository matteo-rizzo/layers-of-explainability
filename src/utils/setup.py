import random
import re

import numpy as np
import torch


def set_random_seed(seed: int, device: torch.device):
    """
    Set specific seed for reproducibility.

    :param seed: int, the seed to set
    :param device: torch.device, cuda:number or cpu
    :return:
    """
    torch.manual_seed(seed)
    if device.type == 'cuda:3':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_type: str) -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device)
    :param device_type: the id of the selected device (if cuda device, must match the regex "cuda:\d"
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device)
    """
    if device_type == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print(f"WARNING: running on cpu since device {device_type} is not available")
            return torch.device("cpu")
        return torch.device(device_type)

    raise ValueError(f"ERROR: {device_type} is not a valid device! Supported device are 'cpu' and 'cuda:n'")
