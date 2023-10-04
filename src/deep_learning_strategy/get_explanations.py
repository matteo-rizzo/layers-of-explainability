from src.ami2020.dataset import train_val_test
from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.deep_learning_strategy.pipeline import HugghingFacePipeline

from src.deep_learning_strategy.utils import shap_explain

NUM_EXPLANATIONS = 10

if __name__ == "__main__":
    config: dict = load_yaml("src/nlp/params/deep_learning_strategy.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]

    pipe_m = HugghingFacePipeline(config["testing"]["task_m_model_name"], device=0 if use_gpu else "cpu", batch_size=bs,
                                  top_k=None)
    dataset_m = train_val_test(target="M")

    shap_explain(dataset_m["test"]["x"][:NUM_EXPLANATIONS], model=pipe_m, tokenizer=pipe_m.tokenizer,
                 target_label=target_label)
