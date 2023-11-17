from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.base
import torch.cuda
from sklearn import metrics
from sklearn.linear_model import RidgeClassifier

from src.text_classification.classes.other.TopicExtractor import TopicExtractor
from src.utils.yaml_manager import load_yaml

pd.set_option("display.max_columns", None)

NORMALIZE_INPUT_EMBEDDINGS = False  # L2-normalization of sentence embeddings
CLASSIFIER: type = RidgeClassifier


# def list_paper_per_cluster(documents: list[Document], topics: list[int] | np.ndarray[int]) -> dict[int, list[str]]:
#     if isinstance(topics, np.ndarray):
#         topics: list[int] = topics.tolist()
#
#     document_ids: list[str] = [d.id for d in documents]
#
#     doc_by_cluster: list[tuple[str, int]] = sorted(list(zip(document_ids, topics)), key=lambda x: x[1])
#
#     grouped_docs = dict()  # defaultdict(list)
#     for k, g in groupby(doc_by_cluster, key=lambda x: x[1]):
#         grouped_docs[k] = [doc_id for doc_id, _ in g]
#     return grouped_docs
#
#
# def get_word_relative_importance(words_topics: dict[int, list[tuple[str, float]]]) -> dict[int, list[tuple[str, float]]]:
#     """
#     Weight the importance of representative keywords
#
#     :param words_topics: topic keywords for each cluster
#     :return: dictionary as the input but with weighted importance
#     """
#
#     # sum_importance = {k: sum([s for _, s in ws]) for k, ws in words_topics.items()}
#
#     # words = {k: [(w, float(s / sum_importance[k])) for w, s in ws] for k, ws in words_topics.items()}
#     words = {k: [(w, float(s)) for w, s in ws] for k, ws in words_topics.items()}
#     # words_score = {k: [s / sum_importance[k] for _, s in ws] for k, ws in words_topics.items()}
#     return words


def compute_metrics(y_pred, y_true, sk_classifier_name: str = None) -> dict[str, float]:
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro")
    acc = metrics.accuracy_score(y_true, y_pred)
    if sk_classifier_name:
        print(f"{sk_classifier_name} accuracy: {acc:.3f}")
        print(f"{sk_classifier_name} precision: {precision:.3f}")
        print(f"{sk_classifier_name} recall: {recall:.3f}")
        print(f"{sk_classifier_name} F1-score: {f1_score:.3f}")

    return {"f1": f1_score, "accuracy": acc, "precision": precision, "recall": recall}


def extract_document_texts(dump_path: str | Path) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """ Extract text to be clustered. """

    train_data = sklearn.datasets.fetch_20newsgroups(data_home=str(dump_path), subset="train", shuffle=True, random_state=37, remove=('headers', 'footers', 'quotes'))
    test_data = sklearn.datasets.fetch_20newsgroups(data_home=str(dump_path), subset="test", shuffle=True, random_state=37, remove=('headers', 'footers', 'quotes'))
    all_data = sklearn.datasets.fetch_20newsgroups(data_home=str(dump_path), subset="all", shuffle=True, random_state=37, remove=('headers', 'footers', 'quotes'))
    return all_data["data"], all_data["target"], train_data["data"], train_data["target"], test_data["data"], test_data["target"]


def train_topic_model(texts: list[str], config_file: str | Path,
                      plot_path=Path("plots") / "topics",
                      dump_path=Path("dumps") / "clustering",
                      seed_topic_list: list[list[str]] | None = None, y: list | None = None) -> TopicExtractor:
    """
    Create the topic model, train it and return the trained model

    """
    # plot_path.mkdir(exist_ok=True, parents=True)
    extractor = TopicExtractor(plot_path=plot_path, dump_path=dump_path)
    extractor.prepare(config_file=config_file, seed_topic_list=seed_topic_list)

    embeddings = None
    if Path(extractor.embedding_save_path).is_file():
        embeddings = np.load(extractor.embedding_save_path)

    topics, _ = extractor.train(texts, embeddings=embeddings, normalize=NORMALIZE_INPUT_EMBEDDINGS, y=y)

    torch.cuda.empty_cache()

    bdcv_score = extractor.topic_model.hdbscan_model.relative_validity_
    n_clusters = max(topics) + 1
    n_outliers: int = len([t for t in topics if t < 0])

    print(f"Clustering results:\n N. clusters: {n_clusters}\n N. outliers: {n_outliers}\n BDCV score: {bdcv_score}\n")

    return extractor


def train_classifier(features: np.ndarray, truth: np.ndarray | list, config_file: str | Path,
                     dump_path=Path("dumps") / "classifier") -> sklearn.base.ClassifierMixin:
    """
    Takes doc features (probabilities of belonging to each cluster) and return trained classifier for inference
    """
    clf_config = load_yaml(config_file)
    clf_name = CLASSIFIER.__name__

    clf = CLASSIFIER(**clf_config[clf_name])

    clf.fit(features, truth)
    train_pred = clf.predict(features)

    dump_path.mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, dump_path / f"{clf_name}.bin")

    print("\nTrain metrics:")
    compute_metrics(train_pred, truth, "Experimental CLF (train)")

    return clf


def main():
    dump_path = Path("dumps") / "text_classification"
    dump_path.mkdir(exist_ok=True, parents=True)
    clustering_config = Path("src") / "text_classification" / "config" / "bertopic2.yml"

    all_docs, y_all, train_docs, y_train, test_docs, y_test = extract_document_texts(dump_path / "data")

    # Train topic model
    extractor = train_topic_model(train_docs, config_file=clustering_config, dump_path=dump_path / "clustering", y=y_train)

    # Inference
    topics, prob_main_topic, prob_all_topics, words_topics = extractor.predict(train_docs, -1, use_training_embeddings=True)

    # Compose features, save to file
    prob_all_topics: np.ndarray[float]
    np.save(dump_path, prob_all_topics)

    training_config = Path("src") / "text_classification" / "config" / "classifier.yml"

    # Create train and test split for classification
    # train_docs, test_docs, y_train, y_test, feature_train, feature_test = train_test_split(all_docs, y_all, prob_all_topics, shuffle=True, random_state=37, stratify=y_all)

    # Train classifier
    clf = train_classifier(prob_all_topics, truth=y_train, config_file=training_config, dump_path=dump_path / "classification")

    # Inference classifier
    _, _, prob_all_topics, _ = extractor.predict(test_docs, -1, use_training_embeddings=False)
    test_prediction = clf.predict(prob_all_topics)

    # Measure performance
    print("\nTest metrics:")
    compute_metrics(test_prediction, y_test, "Experimental Classifier (test)")


if __name__ == "__main__":
    main()
