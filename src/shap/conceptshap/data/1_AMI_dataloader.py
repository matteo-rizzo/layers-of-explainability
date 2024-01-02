import argparse
import random

import pandas as pd

from src.datasets.classes.AMI2020Dataset import AMI2020Dataset
from src.utils.ami_2020_scripts.dataset_handling import train_val_test


def make_sliding_window_pkl(size, data: dict, savedir):
    windows = []
    labels = []

    x_data = data["x"]
    y_data = data["y"]

    # Shuffle data
    c = list(zip(x_data, y_data))
    random.shuffle(c)
    x_data, y_data = zip(*c)

    for i in range(size):
        split_review = x_data[i].split()
        label = y_data[i]
        for j in range(10, len(split_review)):
            sliding_window = split_review[j - 10:j]
            windows.append(sliding_window)
            labels.append(label)

    d = {"sentence": windows, "polarity": labels}
    pd.DataFrame.from_dict(d).to_pickle(savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000,
                        help="how many training and test sentences to run fragment extraction on")
    parser.add_argument("--train_dir", type=str, default="dumps/AMI_train_fragments.pkl",
                        help="path to save extracted sentence fragments")
    parser.add_argument("--test_dir", type=str, default="dumps/AMI_test_fragments.pkl",
                        help="path to save extracted sentence fragments")
    args = parser.parse_args()

    splits = train_val_test(target="M", add_synthetic_train=False, preprocessing_function=AMI2020Dataset.preprocessing)
    make_sliding_window_pkl(args.size, splits["train"], args.train_dir)
    make_sliding_window_pkl(args.size, splits["test"], args.test_dir)
    print("saved fragment files")
