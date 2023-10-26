import argparse

from src.deep_learning_strategy.classes.HuggingFaceAMI2020Dataset import HuggingFaceAMI2020Dataset
from src.deep_learning_strategy.classes.HuggingFacePipeline import HuggingFacePipeline
from src.deep_learning_strategy.classes.ShapExplainer import ShapExplainer
from src.deep_learning_strategy.classes.TranShapExplainer import TranShapExplainer
from src.utils.setup import PATH_TO_CONFIG, RANDOM_SEED, make_deterministic, print_namespace
from src.utils.yaml_manager import load_yaml

NUM_EXPLANATIONS = 100
EXPLAINER = "transhap"
SHOW = True


def main(ns: argparse.Namespace):
    config: dict = load_yaml(PATH_TO_CONFIG)
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]
    num_explanations: int = ns.num_explanations
    explainer = ns.explainer
    show = ns.show

    device = "cuda" if use_gpu else "cpu"

    pipeline = HuggingFacePipeline(model_name=config["testing"]["task_m_model_name"], batch_size=bs, top_k=None).get()
    data = HuggingFaceAMI2020Dataset().get_train_val_test_split()

    dataset_processed = [HuggingFaceAMI2020Dataset.preprocessing(t) for t in data["test"]["x"][:num_explanations]]

    match explainer:
        case "transhap":
            explanation_method = TranShapExplainer(pipeline, pipeline.tokenizer, target_label, device)
            explanation_method.run(corpus=dataset_processed, explain_ids=[0, 99], show=show)
        case "shape":
            explanation_method = ShapExplainer(pipeline, pipeline.tokenizer, target_label)
            explanation_method.run(corpus=dataset_processed, show=True)
        case _:
            raise ValueError("Explainer '{}' not supported. Supported explainer are {}, {}"
                             .format(explainer, "transhap", "shap"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--num_explanations", type=int, default=NUM_EXPLANATIONS)
    parser.add_argument("--explainer", type=str, default=EXPLAINER)
    parser.add_argument("--show", type=bool, default=SHOW)
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
