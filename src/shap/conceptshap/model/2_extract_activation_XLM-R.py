import os
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from src.deep_learning_strategy import HuggingFacePipeline
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import Pipeline

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
DEVICE = "cuda" if config["use_gpu"] else "cpu"


def main():
    pip = HuggingFacePipeline(TEST_MODEL_NAME, BATCH_SIZE)
    dataset = pd.read_pickle("dumps/AMI_train_fragments.pkl")

    hf_pipeline: Pipeline = pip.get()
    model = hf_pipeline.model.to(DEVICE)

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

    print("running inference...")
    # Run the whole model

    tokenized_dataset = hf_pipeline.tokenizer(text=dataset["sentence"].tolist(), padding="max_length", truncation=True,
                                              max_length=model_max_length, is_split_into_words=True)

    torch_data = TensorDataset(torch.IntTensor(tokenized_dataset.input_ids).to(DEVICE),
                               torch.tensor(tokenized_dataset.attention_mask).to(DEVICE))
    sampler = SequentialSampler(torch_data)
    loader = DataLoader(torch_data, sampler=sampler, batch_size=BATCH_SIZE)

    all_results = list()
    for batch in tqdm(loader):
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask).logits
            results = torch.sigmoid(logits)
            all_results.extend(results.cpu().detach().numpy().tolist())

    predictions = np.argmax(all_results, axis=1).tolist()
    metrics = compute_metrics(y_pred=predictions, y_true=dataset["polarity"].tolist())
    pprint(metrics)

    activations = np.concatenate(extracted_activations, axis=0)
    print(activations.shape)
    np.save(ACTIVATION_DIR, activations)


if __name__ == "__main__":
    main()
