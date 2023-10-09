from src.utils.ami_2020_scripts.dataset_handling import train_val_test
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline

from src.deep_learning_strategy.utils import shap_explain
from src.utils.yaml_manager import load_yaml

NUM_EXPLANATIONS = 10

if __name__ == "__main__":
    config: dict = load_yaml("src/deep_learning_strategy/config.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]

    pipe_m = HuggingFacePipeline(config["testing"]["task_m_model_name"], batch_size=bs, top_k=None)
    dataset_m = train_val_test(target="M")

    hf_pipeline = pipe_m.get()
    shap_explain(dataset_m["test"]["x"][:NUM_EXPLANATIONS], model=hf_pipeline, tokenizer=hf_pipeline.tokenizer, target_label=target_label)
