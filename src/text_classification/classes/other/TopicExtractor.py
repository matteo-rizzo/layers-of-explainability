from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar, Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from umap import UMAP

from src.text_classification.classes.other import Document
from src.text_classification.classes.other.BERTopicExtended import BERTopicExtended
from src.text_classification.classes.other.BaseTopicExtractor import BaseTopicExtractor
from src.text_classification.utils import load_yaml
from src.text_classification.visualization.visualize_stacked_topics import visualize_stacked_topics

T = TypeVar("T")

logger = logging.getLogger(__name__)


def get_topic_probabilities(probabilities: np.ndarray[float], predictions: np.ndarray[int]) -> np.ndarray[float]:
    # Get indices of samples that got assigned -1 as the topic (outliers)
    outlier_indices: np.ndarray[int] = np.argwhere(np.asarray(predictions) == -1).reshape(-1)  # np.where(np.asarray(predictions) == -1, 1, 0)

    if len(outlier_indices) > 0:
        # Set the topic with the highest probability for outliers
        predictions[outlier_indices] = probabilities[outlier_indices].argmax(axis=1)

    assert not predictions[predictions < 0].any(), "There are still predictions at -1"

    # Get probability of the most matching topic
    best_probabilities = probabilities[:, predictions].diagonal()  # (N,)
    return best_probabilities


class TopicExtractor(BaseTopicExtractor):
    def save(self, path: str | Path, *args, **kwargs):
        self._topic_model.save(path, *args, **kwargs)

    def load(self, path: str | Path, *args, **kwargs):
        self._topic_model = BERTopic.load(path, *args, **kwargs)
        # TODO: assign components

    def __init__(self, plot_path: Path | str = Path("plots"), dump_path: Path | str = Path("dumps")):
        self._train_embeddings = None
        self._topic_model: BERTopicExtended = None
        self._reduction_model = None
        self._config = None
        self._embedding_model = None
        self._clustering_model = None
        self._vectorizer_model = None
        self._weighting_model = None
        self._representation_model = None
        self._plot_path: Path = plot_path
        self._dump_path: Path = dump_path
        self._instantiation_kwargs = None

    @staticmethod
    def tl_factory(tl_args: dict) -> BERTopic:
        return BERTopicExtended(**tl_args)

    @property
    def topic_model(self) -> BERTopic:
        return self._topic_model

    @property
    def embedding_save_path(self) -> str | Path:
        return self._embedding_save_path

    def prepare(self, *args, **kwargs):
        # Allow config overriding for grid search
        config: dict | None = kwargs.pop("config", None)
        if not config:
            config_path: str | Path = kwargs.pop("config_file")
            config = load_yaml(config_path)
        self._config = config

        run_config = self._config["run"]
        model_config = self._config["model"]

        print("*** Preparing everything ***")

        # Step 1 - Extract embeddings
        self._embedding_model = SentenceTransformer(model_config["sentence_transformer"])

        # Step 2 - Reduce dimensionality
        model_rd = kwargs.pop("dimensionality_reduction", None)
        if not model_rd:
            conf = model_config["dimensionality_reduction"]
            if conf["choice"] == "umap":
                model_rd = UMAP(**conf["params"][conf["choice"]])
            elif conf["choice"] == "pca":
                model_rd = PCA(**conf["params"][conf["choice"]])
        self._reduction_model = model_rd

        # Step 3 - Cluster reduced embeddings
        model_cl = kwargs.pop("clustering", None)
        if not model_cl:
            conf = model_config["clustering"]
            model_cl = None
            if conf["choice"] == "hdbscan":
                model_cl = HDBSCAN(**conf["params"][conf["choice"]])
            elif conf["choice"] == "kmeans":
                model_cl = KMeans(**conf["params"][conf["choice"]])
            elif conf["choice"] == "gmm":
                model_cl = GaussianMixture(**conf["params"][conf["choice"]])
        self._clustering_model = model_cl
        # if UMAP.n_components is increased may want to change metric in HDBSCAN

        # Step 4 - Tokenize topics
        self._vectorizer_model = CountVectorizer(**model_config["vectorizer"]["params"])

        # Step 5 - Create topic representation
        self._weighting_model = ClassTfidfTransformer(**model_config["weighting"]["params"])

        # Step 6 - (Optional) Fine-tune topic representations
        conf = model_config["representation"]
        model_ft = list()
        if "mmr" in conf["choice"]:
            model_ft.append(MaximalMarginalRelevance(**conf["params"]["mmr"]))
        elif "keybert" in conf["choice"]:
            model_ft.append(KeyBERTInspired(**conf["params"]["keybert"]))
        elif "pos" in conf["choice"]:
            model_ft.append(PartOfSpeech(**conf["params"]["pos"]))
        else:
            model_ft.append(None)
        self._representation_model = model_ft if len(model_ft) > 1 else model_ft[0]

        tl_args = dict(
            embedding_model=self._embedding_model,  # Step 1 - Extract embeddings
            vectorizer_model=self._vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=self._weighting_model,  # Step 5 - Extract topic words
        )
        if self._reduction_model is not None:
            tl_args["umap_model"] = self._reduction_model  # Step 2 - Reduce dimensionality
        if self._clustering_model is not None:
            tl_args["hdbscan_model"] = self._clustering_model  # Step 3 - Cluster reduced embeddings
        if self._representation_model is not None:
            tl_args[
                "representation_model"] = self._representation_model  # Step 6 - (Optional) Fine-tune topic representations

        self._instantiation_kwargs = {
            **tl_args,
            **model_config["bertopic"],
            **kwargs
        }
        self._topic_model = TopicExtractor.tl_factory(self._instantiation_kwargs)

        self._embedding_save_path = self._dump_path / "embeddings" / f"{model_config['sentence_transformer']}.npy"
        Path(self._embedding_save_path).parent.mkdir(exist_ok=True, parents=True)

    def train(self, documents: list[str], *args, **kwargs) -> Any:
        print("*** Generating embeddings ***")

        embeddings = kwargs.get("embeddings", None)
        fit_reduction = kwargs.get("fit_reduction", True)
        y = kwargs.get("y", None)

        if embeddings is None:
            # Precompute embeddings
            embeddings = self._embedding_model.encode(documents, show_progress_bar=False)
            if kwargs.get("normalize", False):
                # NOTE: only works when predict use the training embeddings
                embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

            np.save(self._embedding_save_path, embeddings)

        self._train_embeddings = embeddings

        print("*** Fitting the model ***")

        # Topic modelling
        # topics, probs = \
        return self._topic_model.fit(documents, embeddings=embeddings, fit_reduction=fit_reduction, y=y)
        # Further reduce topics
        # self._topic_model.reduce_topics(texts, nr_topics=3)

    def predict_one(self, document: Document, k: int, *args, **kwargs) -> list:
        pass

    def predict(self, documents: list[str], k: int, *args, **kwargs) -> tuple:
        """
        Compute topic clusters for each document
        
        :param documents: document to label
        :param k: ignored
        :param use_training_embeddings: if true assumes documents are the same used for training, otw must be set to false
        :param args: 
        :param kwargs: 
        :return:
        """
        print("*** Extracting topics ***")

        emb_train: bool = kwargs.get("use_training_embeddings", False)

        # Use pre-trained /reduced embeddings.
        emb = self._train_embeddings if emb_train else None

        topics, probs = self._topic_model.transform(documents, embeddings=emb)

        print(f"Outliers: {len([t for t in topics if t < 0])}")

        if kwargs.get("reduce_outliers", False):
            print("*** Reducing outliers ***")
            thr = kwargs.get("threshold", .0)
            topics = self._topic_model.reduce_outliers(documents, topics, probabilities=probs, strategy="probabilities", threshold=thr)
            self._topic_model.update_topics(documents, topics=topics,
                                            vectorizer_model=self._vectorizer_model,
                                            ctfidf_model=self._weighting_model,
                                            representation_model=self._representation_model,
                                            top_n_words=self._topic_model.top_n_words)
            print(f"Outliers post-reduction: {len([t for t in topics if t < 0])}")

        topic_probs = get_topic_probabilities(probs, np.asarray(topics))

        return topics, topic_probs, probs, self._topic_model.get_topics()

    def force_outlier_assignment(self, documents: list[str], topics: list[int], probabilities: np.ndarray, threshold: float, cluster_index: int) -> list[int]:
        # Check the correct use of parameters
        if probabilities is None:
            raise ValueError("Make sure to pass in `probabilities` in order to use the probabilities strategy")

        # Reduce outliers by extracting most likely topics through the topic-term probability matrix
        new_topics = [cluster_index if prob[cluster_index] >= threshold and topic == -1 else topic for topic, prob in zip(topics, probabilities)]

        self._topic_model.update_topics(documents, topics=new_topics,
                                        vectorizer_model=self._vectorizer_model,
                                        ctfidf_model=self._weighting_model,
                                        representation_model=self._representation_model)

        return new_topics

    def document_similarity(self, document_embeddings: np.ndarray, words: list[str], threshold: float) -> list[int]:

        words_concat = " ".join(words)
        words_embeddings = self._topic_model._extract_embeddings(words_concat)

        sim_matrix = cosine_similarity(document_embeddings, words_embeddings)

        return np.argwhere(sim_matrix.reshape(-1) > threshold).reshape(-1).tolist()

    def plot_wonders(self, documents: list[Document], **kwargs) -> pd.DataFrame:

        print("*** Plotting results ***")

        emb_train: bool = kwargs.get("use_training_embeddings", False)
        file_suffix: str | None = kwargs.get("file_suffix", "")

        self._plot_path.mkdir(parents=True, exist_ok=True)

        formatted_labels = self._topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=None,
                                                                   separator=" - ")
        self._topic_model.set_topic_labels(formatted_labels)

        texts = [d.body for d in documents]
        titles = [f"{d.id} - {d.title}" for d in documents]
        years: list[str] = [str(d.timestamp) for d in documents]

        # If documents are passed then those are embedded using the selected emb_model (else training emb are used)
        emb = self._train_embeddings
        if not emb_train:
            emb = self._embedding_model.encode(texts, show_progress_bar=False)

        # Reduce dimensions for document visualization
        reduced_embeddings = self._reduction_model.transform(emb)

        fig_topics = self._topic_model.visualize_topics(width=1200, height=1200)
        fig_topics.write_html(self._plot_path / f"topic_space_{file_suffix}.html")
        fig_doc_topics = self._topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings,
                                                               hide_annotations=True, custom_labels=True, width=1800, height=1200)
        fig_doc_topics.write_html(self._plot_path / f"document_clusters_{file_suffix}.html")

        topics_over_time = self._topic_model.topics_over_time(texts, years, nr_bins=20, datetime_format="%Y")
        fig_time = self._topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=None, custom_labels=True,
                                                                normalize_frequency=True, relative_frequency=False, width=1600, height=800)
        fig_time.write_html(self._plot_path / f"topic_evolution_{file_suffix}.html")

        # fig_hier = self._topic_model.visualize_hierarchy(top_n_topics=None, custom_labels=True)
        # fig_hier.write_html(self._plot_path / "topic_hierarchy.html")

        # topics_per_class = self._topic_model.topics_per_class(texts, classes=GROUND_TRUTH)
        # fig_class = self._topic_model.visualize_topics_per_class(custom_labels=True)
        # fig_class.write_html(self._plot_path / "topic_hierarchy.html")

        if kwargs.get("add_doc_classes", None):
            l2_topics = kwargs["add_doc_classes"]
            fig = visualize_stacked_topics(self._topic_model, titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, custom_labels=True, width=1800, height=1200,
                                           stacked_topics=l2_topics, stacked_symbols=[(0, "circle"), (1, "x")])
            fig.write_html(self._plot_path / f"topic_stacked_{file_suffix}.html")
