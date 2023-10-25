import argparse
import os

from src.deep_learning_strategy.explainations_utils import transhap_explain

from src.deep_learning_strategy.classes.HuggingFaceAMI2020Dataset import HuggingFaceAMI2020Dataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.deep_learning_strategy.settings import make_deterministic, print_namespace, RANDOM_SEED, PATH_TO_CONFIG
from src.utils.yaml_manager import load_yaml

NUM_EXPLANATIONS = 100

if __name__ == "__main__":
    config: dict = load_yaml(PATH_TO_CONFIG)
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]

    pipeline = HuggingFacePipeline(model_name=config["testing"]["task_m_model_name"], batch_size=bs, top_k=None).get()
    data = HuggingFaceAMI2020Dataset().get_train_val_test_split()

    dataset_processed = [HuggingFaceAMI2020Dataset.preprocessing(t) for t in dataset_m["test"]["x"][:NUM_EXPLANATIONS]]

    # shap_explain(dataset_processed, model=hf_pipeline, tokenizer=hf_pipeline.tokenizer, target_label=target_label)
    transhap_explain(dataset_processed, explain_ids=[0, 99], model=hf_pipeline, tokenizer=hf_pipeline.tokenizer,
                     target_label=target_label,
                     device="cuda" if config["use_gpu"] else "cpu")


def main(ns: argparse.Namespace):
    config = load_yaml(os.path.join("src", "deep_learning_strategy", "config.yml"))
    GridSearchFineTuner(config).run() if ns.do_grid_search else FineTuner(config).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_explanations", type=int, default=NUM_EXPLANATIONS)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
