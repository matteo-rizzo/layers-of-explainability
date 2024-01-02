import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import evaluate
import scipy as sp
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback, PreTrainedModel, PreTrainedTokenizer

from src.datasets import HuggingFaceAMI2018Dataset
from src.datasets import HuggingFaceAMI2020Dataset
from src.datasets import HuggingFaceCGReviewDataset
from src.datasets import HuggingFaceCallMeSexistDataset
from src.datasets import HuggingFaceIMDBDataset
from src.text_classification.base_models.classes import BinaryTrainer


class FineTuner:

    def __init__(self, hyperparameters: Dict):
        self.__model_name: str = hyperparameters["training"]["model_name"]
        self.__keep_n_best_models: int = hyperparameters["training"].get("keep_n_best_models", 1)
        self.__eval_size: float = hyperparameters["training"]["eval_size"]
        self.__checkpoint_path: str = hyperparameters["training"]["checkpoint"]
        self.__max_length: int = hyperparameters["training"]["model_max_length"]
        self.__batch_size: int = hyperparameters["training"]["batch_size"]
        self.__freeze_base: bool = hyperparameters["training"]["freeze_base"]
        self.__epochs: bool = hyperparameters["training"]["epochs"]
        self.__resume: bool = hyperparameters["training"]["resume"]
        self.__lr: float = hyperparameters["training"].get("learning_rate", 5.0e-5)
        self.__wd: float = hyperparameters["training"].get("decay", 0.0)
        self.__es_patience: bool = hyperparameters["training"]["es_patience"]
        self.__augment_training: bool = hyperparameters["training"]["add_synthetic"]
        self.__use_gpu: bool = hyperparameters["use_gpu"]
        self.__tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.__model_name)

        if hyperparameters["dataset"] == "AMI2018":
            self.__train_data: Dataset = (HuggingFaceAMI2018Dataset(
                augment_training=self.__augment_training).get_train_data())
        elif hyperparameters["dataset"] == "AMI2020":
            self.__train_data: Dataset = (HuggingFaceAMI2020Dataset(
                augment_training=self.__augment_training).get_train_data())
        elif hyperparameters["dataset"] == "CGReviews":
            self.__train_data: Dataset = (HuggingFaceCGReviewDataset().get_train_data())
        elif hyperparameters["dataset"] == "CallMeSexist":
            self.__train_data: Dataset = (HuggingFaceCallMeSexistDataset().get_train_data())
        elif hyperparameters["dataset"] == "IMDB":
            self.__train_data: Dataset = (HuggingFaceIMDBDataset().get_train_data())
        else:
            raise ValueError(f"Unsupported dataset with name: {hyperparameters['dataset']}")

        self.__data: Dict = self.__get_eval_data()
        self.__model: PreTrainedModel = self.__get_model()
        # Note: adamw_torch_fused has a bug and cannot be reloaded once trained, as workaround use normal version
        # self.__optimizer_name: str = "adamw_torch_fused" if self.__use_gpu else "adamw_torch"
        self.__optimizer_name: str = "adamw_torch"
        self.__trainer: Optional[Trainer] = None
        self.__dataset: Optional[Trainer] = None
        self.__dump_path: Path = Path("dumps") / "nlp_models"

    def run(self):
        training_args = TrainingArguments(
            output_dir=self.__get_checkpoint_path(), overwrite_output_dir=self.__resume,
            num_train_epochs=self.__epochs, seed=1234, data_seed=4321,
            per_device_train_batch_size=self.__batch_size, per_device_eval_batch_size=self.__batch_size,
            dataloader_num_workers=4, dataloader_pin_memory=True, optim=self.__optimizer_name,
            learning_rate=self.__lr, weight_decay=self.__wd, use_cpu=not self.__use_gpu,
            save_strategy="epoch", evaluation_strategy="epoch", logging_strategy="epoch",
            metric_for_best_model="eval_loss", save_total_limit=self.__keep_n_best_models,
            load_best_model_at_end=True)

        self.__trainer = BinaryTrainer(
            model=self.__model,
            args=training_args,
            train_dataset=self.__data["train"],
            eval_dataset=self.__data["test"],
            compute_metrics=self.__get_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.__es_patience)],
            tokenizer=self.__tokenizer)

        results = self.__trainer.train(resume_from_checkpoint=self.__resume)

        self.__log_results(results)

    def get_trainer(self) -> Trainer:
        return self.__trainer

    def __get_model(self) -> PreTrainedModel:
        num_labels = self.__data["train"].features["label"].num_classes
        if num_labels == 2:
            num_labels = 1
        model = AutoModelForSequenceClassification.from_pretrained(self.__model_name,
                                                                   num_labels=num_labels,
                                                                   problem_type="multi_label_classification")
        return self.__freeze_parameters(model) if self.__freeze_base else model

    def __get_eval_data(self) -> Dict:
        data = self.__tokenize_data()
        print(f" -- Using {int(data.num_rows * self.__eval_size)} examples as validation set --")
        return data.train_test_split(test_size=self.__eval_size, shuffle=True, seed=39, stratify_by_column="label")

    def __get_checkpoint_path(self) -> str:
        base_path = self.__dump_path  # os.path.join("src", "deep_learning_strategy", "saved_models")
        os.makedirs(base_path, exist_ok=True)
        return self.__checkpoint_path if self.__resume else self.__next_checkpoint_path(base_path, self.__model_name)

    @staticmethod
    def __next_checkpoint_path(base_path: str, model_name: str) -> str:
        # Filter out items that match the pattern "path_{N}"
        matches = [item for item in os.listdir(base_path) if re.fullmatch(rf"{model_name}_\d+", item)]

        if not matches:
            # If there are no matches, return "path_1"
            max_num = 0
        else:
            # If there are matches, extract the maximum number X and return "path_{X+1}"
            max_num = max(int(re.search(r"\d+", match).group()) for match in matches)

        return os.path.join(base_path, f"{model_name.replace('/', '_')}_{max_num + 1}")

    def __tokenize_data(self) -> Dataset:
        def __tokenize_fun(examples):
            return self.__tokenizer(examples["text"], padding="max_length",
                                    truncation=True, max_length=self.__max_length)

        return self.__train_data.map(__tokenize_fun, batched=True)

    @staticmethod
    def __freeze_parameters(model: PreTrainedModel) -> PreTrainedModel:
        for param in model.base_model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def __get_metrics(model_output: Tuple) -> Dict:
        logits, labels = model_output
        # predictions = np.argmax(logits, axis=-1)
        logits = logits.reshape(-1)
        predictions = sp.special.expit(logits)
        predictions = (predictions >= .5).astype(int)
        return {
            "precision": evaluate.load("precision").compute(predictions=predictions, references=labels)["precision"],
            "recall": evaluate.load("recall").compute(predictions=predictions, references=labels)["recall"],
            "accuracy": evaluate.load("accuracy").compute(predictions=predictions, references=labels)["accuracy"],
            "f1": evaluate.load("f1").compute(predictions=predictions, references=labels)["f1"]
        }

    def __log_results(self, train_results):
        metrics = train_results.metrics
        # Pls do NOT add "self=None" here. PyCharm is wrong, do not trust him :D
        self.__trainer.log_metrics(split="train", metrics=metrics)
        self.__trainer.save_metrics(split="train", metrics=metrics)

        metrics = self.__trainer.evaluate()
        self.__trainer.log_metrics(split="eval", metrics=metrics)
        self.__trainer.save_metrics(split="eval", metrics=metrics)
