import os
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import pipeline, Pipeline

from src.deep_learning_strategy.classes.HuggingFaceDataset import HuggingFaceDataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.utils.ami_2020_scripts.dataset_handling import compute_metrics
from src.utils.yaml_manager import load_yaml

config = load_yaml(os.path.join("src", "deep_learning_strategy", "config.yml"))
BATCH_SIZE = config["training"]["test_batch_size"]
TEST_MODEL_NAME = config["testing"]["task_m_model_name"]
TASK_A_MODEL_NAME = config["testing"]["task_a_model_name"]
TASK_B_MODEL_NAME = config["testing"]["task_b_model_name"]
TARGET_LABEL = config["testing"]["target_label"]
TASK = config["task"]
ADD_SYNTHETIC = config["add_synthetic"]
model_max_length: int = config["training"]["model_max_length"]
ACTIVATION_DIR = "dumps/AMI_embeddings.npy"


def main():
    print("*** PREDICTING MISOGYNY ***")
    pip = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    dataset = HuggingFaceDataset(augment_training=ADD_SYNTHETIC or TASK == "B")
    # predictions = pip.test(dataset.get_test_data(), TARGET_LABEL)

    hf_pipeline: Pipeline = pip.get()
    model = hf_pipeline.model

    extracted_activations = list()

    dropout = model.classifier.dropout

    def extract_activation_hook(model, input, output):
        output = torch.tanh(output)
        output = dropout(output)
        extracted_activations.append(output.cpu().numpy())

    def add_activation_hook(model, layer_idx):
        all_modules_list = list(model.modules())
        module = all_modules_list[layer_idx]
        module.register_forward_hook(extract_activation_hook)

    add_activation_hook(model, layer_idx=-3)

    print("running inference..")
    # run the whole model

    predictions = pip.test(dataset.get_test_data(), TARGET_LABEL)
    metrics = compute_metrics(y_pred=predictions, y_true=dataset.get_test_groundtruth())
    pprint(metrics)

    activations = np.concatenate(extracted_activations, axis=0)
    np.save(ACTIVATION_DIR, activations)

    # ce_loss = nn.BCEWithLogitsLoss()
    #
    # all_losses = []
    #
    # def tokenize_function(examples):
    #     return hf_pipeline.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model_max_length)
    #
    # tokenized_train_ds = dataset.test_data.map(tokenize_function, batched=True)
    # loader = DataLoader(tokenized_train_ds, batch_size=BATCH_SIZE)
    #
    # for batch in tqdm(loader):
    #     b_input_ids = batch["input_ids"]
    #     b_input_mask = batch["attention_mask"]
    #     b_labels = batch["label"]
    #     # print(torch.sum(b_labels).item())
    #     # outputs doesn't need to be saved
    #     with torch.no_grad():
    #         logits = model(**batch)
    #         loss_val_list = ce_loss(logits, b_labels)
    #         pred_loss = torch.mean(loss_val_list).item()
    #         all_losses.append(pred_loss)
    # print("inference loss:", np.mean(np.array(all_losses)))


if __name__ == "__main__":
    main()
