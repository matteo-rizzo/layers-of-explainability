from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable

import evaluate
import numpy as np
import shap
from datasets import Dataset, NamedSplit, Features, Value, ClassLabel
from matplotlib import pyplot as plt
from shap.models import TransformersPipeline
from transformers import Trainer

from dataset.ami2020_misogyny_detection.scripts.dataset_handling import train_val_test


def create_hf_dataset(target: str = "M", add_synthetic: bool = False,
                      preprocessing_function: Callable[[str], str] = None) -> tuple[Dataset, Dataset]:
    """
    Load AMI dataset with specified the binary target and prepare HuggingFace Dataset wrapper

    :return: training and test dataset
    """
    dataset = train_val_test(target=target, add_synthetic_train=add_synthetic,
                             preprocessing_function=preprocessing_function)

    data_hf = {k: {"text": v["x"], "label": v["y"]} for k, v in dataset.items()}

    feat = Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["no", "yes"], names_file=None, id=None)}
    )

    ds_train = Dataset.from_dict(data_hf["train"], split=NamedSplit("train"), features=feat)
    ds_test = Dataset.from_dict(data_hf["test"], split=NamedSplit("test"), features=feat)

    return ds_train, ds_test


def compute_metrics(eval_pred):
    """
    Metrics computation for HF Trainer

    :param eval_pred: output of model
    :return: dictionary of metrics
    """
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("accuracy")
    metric4 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    accuracy = metric3.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric4.compute(predictions=predictions, references=labels)["f1"]
    return {"precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1}


def get_next_run_name(base_path: str | Path, model_name: str) -> Path:
    """
    Search correct sequence for checkpoint folders and return new name
    This is useful not to overwrite existing dumps.
    """
    base_path = Path(base_path)
    # Get a list of all items in the directory
    items = os.listdir(base_path)

    # Filter out items that match the pattern "path_{N}"
    matches = [item for item in items if re.fullmatch(rf"{model_name}_\d+", item)]

    if not matches:
        # If there are no matches, return "path_1"
        max_num = 0
    else:
        # If there are matches, extract the maximum number X and return "path_{X+1}"
        max_num = max(int(re.search(r"\d+", match).group()) for match in matches)
    return base_path / f"{model_name}_{max_num + 1}"


def log_results(trainer_results, trainer: Trainer):
    """ Log Trainer metrics to file json """
    # compute train results
    metrics = trainer_results.metrics

    # save train results
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # compute evaluation results
    metrics = trainer.evaluate()

    # save evaluation results
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # dataset statistics
    metrics["train_samples"] = len(trainer.train_dataset)  # min(max_val_samples, len(small_eval_dataset))
    metrics["eval_samples"] = len(trainer.eval_dataset)  # min(max_val_samples, len(small_eval_dataset))


def delete_checkpoints(path: Path | str) -> None:
    """ Remove HF-formatted checkpoints 'checkpoint-XX' from a path """
    path = Path(path)
    # Iterate over all items in the directory
    for item in os.listdir(path):
        # Construct the full path
        item_path = path / item
        # Check if it's a directory and matches the pattern 'checkpoint-X'
        if item_path.is_dir() and re.match(r"^checkpoint-\d+$", item):
            # Delete the directory
            shutil.rmtree(item_path)


def shap_explain(texts: list[str], model, tokenizer, target_label: str, show: bool = True):
    pmodel = shap.models.TransformersPipeline(model, rescale_to_logits=True)

    explainer = shap.Explainer(pmodel, tokenizer)
    shap_values = explainer(texts)

    target_dir = Path("plots") / "nlp"
    target_dir.mkdir(parents=True, exist_ok=True)

    # fig = plt.figure()
    # shap.plots.text(shap_values[0, :, target_label])

    for i, t in enumerate(texts):
        shap.plots.bar(shap_values[i, :, target_label], max_display=25, show=False)
        plt.title(f"Sentence: {t}")
        plt.gcf().set_size_inches(10, 14)
        plt.tight_layout()
        plt.savefig(target_dir / f"shap_sentence_barplot_{i}.png", dpi=400)
        plt.clf()

    shap.plots.bar(shap_values[:, :, target_label].mean(0), max_display=25, show=False)
    plt.title("Mean relative importance of tokens for classification")
    plt.gcf().set_size_inches(10, 14)
    plt.tight_layout()

    plt.savefig(target_dir / "shap_mean_barplot.png", dpi=400)
    if show:
        plt.show()
