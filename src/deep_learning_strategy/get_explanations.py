from src.deep_learning_strategy.classes.HuggingFaceDataset import HuggingFaceDataset
from src.utils.ami_2020_scripts.dataset_handling import train_val_test
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline

from src.deep_learning_strategy.explainations_utils import shap_explain, transhap_explain
from src.utils.yaml_manager import load_yaml

NUM_EXPLANATIONS = 100

if __name__ == "__main__":
    config: dict = load_yaml("src/deep_learning_strategy/config.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]

    pipe_m = HuggingFacePipeline(config["testing"]["task_m_model_name"], batch_size=bs, top_k=None)
    dataset_m = train_val_test(target="M")

    hf_pipeline = pipe_m.get()

    dataset_processed = [HuggingFaceDataset.preprocessing(t) for t in dataset_m["test"]["x"][:NUM_EXPLANATIONS]]

    # shap_explain(dataset_processed, model=hf_pipeline, tokenizer=hf_pipeline.tokenizer, target_label=target_label)
    transhap_explain(dataset_processed, explain_ids=[0, 99], model=hf_pipeline, tokenizer=hf_pipeline.tokenizer,
                     target_label=target_label,
                     device="cuda" if config["use_gpu"] else "cpu")
