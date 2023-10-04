from __future__ import annotations

import itertools
import os
from pathlib import Path

import torch
from src.deep_learning_strategy.pipeline import deep_preprocessing
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers.trainer import Trainer

from src.deep_learning_strategy.utils import create_hf_dataset, compute_metrics, get_next_run_name, log_results, \
    delete_checkpoints
from src.utils.yaml_manager import load_yaml, dump_yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DO_GRID_SEARCH: bool = True


def finetune(hyperparameters) -> Trainer:
    """ Start LM fine-tuning using HuggingFace Trainer and utilities """
    base_model: str = hyperparameters["training"]["model_name"]
    keep_n_best_models: int = hyperparameters["training"].get("keep_n_best_models", 1)
    eval_size: float = hyperparameters["training"]["eval_size"]
    model_max_length: int = hyperparameters["training"]["model_max_length"]
    batch_size: int = hyperparameters["training"]["batch_size"]
    freeze_base: bool = hyperparameters["training"]["freeze_base"]
    epochs: bool = hyperparameters["training"]["epochs"]
    resume: bool = hyperparameters["training"]["resume"]
    lr: float = hyperparameters["training"].get("learning_rate", 5.0e-5)
    wd: float = hyperparameters["training"].get("decay", 0.0)
    es_patience: bool = hyperparameters["training"]["es_patience"]
    use_gpu: bool = hyperparameters["use_gpu"]

    base_model_path: Path = Path("dumps") / "nlp_models"
    base_model_path.mkdir(parents=True, exist_ok=True)

    if resume:
        checkpoint_model_path = hyperparameters["training"]["checkpoint"]
    else:
        checkpoint_model_path = get_next_run_name(base_model_path, base_model.replace("/", "_"))

    train_ds, _ = create_hf_dataset(target="M", add_synthetic=hyperparameters["training"]["add_synthetic"],
                                    preprocessing_function=deep_preprocessing)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model_max_length)

    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    # tokenized_test_ds = test_ds.map(tokenize_function, batched=True) # test set not used here

    print(f"Using {int(tokenized_train_ds.num_rows * eval_size)} examples as validation set")
    train_eval_dataset = tokenized_train_ds.train_test_split(test_size=eval_size, shuffle=True, seed=39,
                                                             stratify_by_column="label")

    model = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                               num_labels=train_eval_dataset["train"].features[
                                                                   "label"].num_classes)
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    optim = "adamw_torch"  # "adamw_torch_fused" if use_gpu else "adamw_torch"
    training_args = TrainingArguments(output_dir=checkpoint_model_path,
                                      overwrite_output_dir=resume,
                                      num_train_epochs=epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      dataloader_num_workers=4, dataloader_pin_memory=True,
                                      optim=optim, learning_rate=lr, weight_decay=wd,
                                      use_cpu=not use_gpu, seed=943, data_seed=3211,
                                      save_strategy="epoch", evaluation_strategy="epoch", logging_strategy="epoch",
                                      metric_for_best_model="eval_loss", save_total_limit=keep_n_best_models,
                                      load_best_model_at_end=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_eval_dataset["train"],
        eval_dataset=train_eval_dataset["test"],  # validation set
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=es_patience)],
        tokenizer=tokenizer  # this is needed to load correctly for inference
    )

    r = trainer.train(resume_from_checkpoint=resume)

    log_results(r, trainer)

    return trainer


def grid_search_finetune(hyperparameters) -> None:
    """
    Grid Search implementation

    :param hyperparameters: configuration with training params; GS params should be listed in 'grid_search_params' field.
    """
    gs_params = hyperparameters["grid_search_params"]
    # Get all possible combinations of hyperparameters
    combinations = [dict(zip(gs_params.keys(), cs)) for cs in itertools.product(*gs_params.values())]
    # We do not need to save checkpoints for this
    hyperparameters["training"]["keep_n_best_models"] = 1

    for c in combinations:
        # Update config
        hyperparameters["training"].update(c)
        # Train and evaluate
        trainer: Trainer = finetune(hyperparameters)

        # Save the used config for reference
        dumps_dir = Path(trainer.args.output_dir)
        dump_yaml(hyperparameters["training"], dumps_dir / "gs_params.yml")
        # Delete heavy dumps
        delete_checkpoints(dumps_dir)

        del trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available :("

    config: dict = load_yaml("src/nlp/params/deep_learning_strategy.yml")
    if DO_GRID_SEARCH:
        grid_search_finetune(config)
    else:
        finetune(config)
