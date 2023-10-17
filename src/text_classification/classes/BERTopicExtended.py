from __future__ import annotations

import logging
from typing import Union, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic
from bertopic._bertopic import TopicMapper
from bertopic._utils import check_is_fitted, check_documents_type, check_embeddings_shape
from bertopic.backend._utils import select_backend
from bertopic.cluster import BaseCluster
from bertopic.cluster._utils import is_supported_hdbscan, hdbscan_delegator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from src.text_classification.visualization.plotly_utils import visualize_topics_over_time_ext

logger = logging.getLogger(__name__)


class BERTopicExtended(BERTopic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reduced_train_embeddings: np.ndarray | None = None

    def visualize_topics_over_time(self,
                                   topics_over_time: pd.DataFrame,
                                   top_n_topics: int = None,
                                   topics: list[int] = None,
                                   normalize_frequency: bool = False,
                                   relative_frequency: bool = False,
                                   custom_labels: bool = False,
                                   title: str = "<b>Topics over Time</b>",
                                   width: int = 1250,
                                   height: int = 450) -> go.Figure:
        """ Visualize topics over time

        Arguments:
            topics_over_time: The topics you would like to be visualized with the
                              corresponding topic representation
            top_n_topics: To visualize the most frequent topics instead of all
            topics: Select which topics you would like to be visualized
            normalize_frequency: Whether to normalize each topic's frequency individually
            relative_frequency: Whether to show the relative frequency. Overrides normalize_frequency
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure including all traces

        Examples:

        To visualize the topics over time, simply run:

        ```python
        topics_over_time = topic_model.topics_over_time(docs, timestamps)
        topic_model.visualize_topics_over_time(topics_over_time)
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_topics_over_time(topics_over_time)
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return visualize_topics_over_time_ext(self,
                                              topics_over_time=topics_over_time,
                                              top_n_topics=top_n_topics,
                                              topics=topics,
                                              normalize_frequency=normalize_frequency,
                                              relative_frequency=relative_frequency,
                                              custom_labels=custom_labels,
                                              title=title,
                                              width=width,
                                              height=height)

    def fit(self, *args, **kwargs) -> Any:
        return self.fit_transform(*args, **kwargs)

    def fit_transform(self,
                      documents: list[str],
                      embeddings: np.ndarray = None,
                      y: Union[list[int], np.ndarray] = None, fit_reduction: bool = True) -> tuple[list[int], Union[np.ndarray, None]]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Arguments:
            documents: A list of documents to fit on
            fit_reduction: whether UMAP should be fitted (or used as already fit)
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
                           If `calculate_probabilities` in BERTopic is set to True, then
                           it calculates the probabilities of all topics across all documents
                           instead of only the assigned topic. This, however, slows down
                           computation and may increase memory usage.

        Examples:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings)
        ```
        """
        check_documents_type(documents)
        check_embeddings_shape(embeddings, documents)

        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        if embeddings is None:
            self.embedding_model = select_backend(self.embedding_model,
                                                  language=self.language)
            embeddings = self._extract_embeddings(documents.Document,
                                                  method="document",
                                                  verbose=self.verbose)
            logger.info("Transformed documents to Embeddings")
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)

        # Reduce dimensionality
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)

        if fit_reduction:
            umap_embeddings = self._reduce_dimensionality(embeddings, y)
        else:
            umap_embeddings = self.umap_model.transform(embeddings)
            logger.info("Reduced dimensionality")
            umap_embeddings = np.nan_to_num(umap_embeddings)

        # Cluster reduced embeddings
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents)

        # Resulting output
        self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, self.probabilities_

    # def fit(self,
    #         documents: list[str],
    #         embeddings: np.ndarray = None,
    #         reduced_embeddings: np.ndarray | str = None,
    #         y: Union[list[int], np.ndarray] = None) -> BERTopicExtended:
    #     self.fit_transform(documents, embeddings, reduced_embeddings, y)
    #     return self
    #
    def transform(self,
                  documents: Union[str, list[str]],
                  embeddings: np.ndarray = None,
                  images: list[str] = None) -> tuple[list[int], np.ndarray]:
        """ After having fit a model, use transform to predict new instances

        Arguments:
            documents: A single document or a list of documents to predict on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model.
            images: A list of paths to the images to predict on or the images themselves

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution which is returned by default.
                           If `calculate_probabilities` in BERTopic is set to False, then the
                           probabilities are not calculated to speed up computation and
                           decrease memory usage.

        Examples:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic().fit(docs)
        topics, probs = topic_model.transform(docs)
        ```

        If you want to use your own embeddings:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic().fit(docs, embeddings)
        topics, probs = topic_model.transform(docs, embeddings)
        ```
        """
        check_is_fitted(self)
        check_embeddings_shape(embeddings, documents)

        if isinstance(documents, str) or documents is None:
            documents = [documents]

        if embeddings is None:
            embeddings = self._extract_embeddings(documents,
                                                  images=images,
                                                  method="document",
                                                  verbose=self.verbose)

        # Check if an embedding model was found
        if embeddings is None:
            raise ValueError("No embedding model was found to embed the documents."
                             "Make sure when loading in the model using BERTopic.load()"
                             "to also specify the embedding model.")

        # Transform without hdbscan_model and umap_model using only cosine similarity
        elif type(self.hdbscan_model) == BaseCluster:
            sim_matrix = cosine_similarity(embeddings, np.array(self.topic_embeddings_))
            predictions = np.argmax(sim_matrix, axis=1) - self._outliers

            if self.calculate_probabilities:
                probabilities = sim_matrix
            else:
                probabilities = np.max(sim_matrix, axis=1)

        # Transform with the full pipeline
        else:
            umap_embeddings = self.umap_model.transform(embeddings)
            logger.info("Reduced dimensionality")

            # Extract predictions and probabilities if it is a HDBSCAN-like model
            if is_supported_hdbscan(self.hdbscan_model):
                predictions, probabilities = hdbscan_delegator(self.hdbscan_model, "approximate_predict", umap_embeddings)

                # Calculate probabilities
                if self.calculate_probabilities:
                    probabilities = hdbscan_delegator(self.hdbscan_model, "membership_vector", umap_embeddings)
                    logger.info("Calculated probabilities with HDBSCAN")
            else:
                predictions = self.hdbscan_model.predict(umap_embeddings)
                probabilities = None
                # ******************** MY ADDITION ***************************
                if hasattr(self.hdbscan_model, "predict_proba"):
                    probabilities = self.hdbscan_model.predict_proba(umap_embeddings)  # (N, clusters)
                # ******************** END ADDITION ***************************
            logger.info("Predicted clusters")

            # Map probabilities and predictions
            probabilities = self._map_probabilities(probabilities, original_topics=True)
            predictions = self._map_predictions(predictions)
        return predictions, probabilities

    def _cluster_embeddings(self,
                            umap_embeddings: np.ndarray,
                            documents: pd.DataFrame,
                            partial_fit: bool = False,
                            y: np.ndarray = None) -> tuple[pd.DataFrame, np.ndarray]:
        """ Cluster UMAP embeddings with HDBSCAN

        Arguments:
            umap_embeddings: The reduced sentence embeddings with UMAP
            documents: Dataframe with documents and their corresponding IDs
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            documents: Updated dataframe with documents and their corresponding IDs
                       and newly added Topics
            probabilities: The distribution of probabilities
        """
        if partial_fit:
            self.hdbscan_model = self.hdbscan_model.partial_fit(umap_embeddings)
            labels = self.hdbscan_model.labels_
            documents['Topic'] = labels
            self.topics_ = labels
        else:
            try:
                self.hdbscan_model.fit(umap_embeddings, y=y)
            except TypeError:
                self.hdbscan_model.fit(umap_embeddings)

            # ******************************** Add support for GMM
            if isinstance(self.hdbscan_model, GaussianMixture):
                labels = self.hdbscan_model.predict(umap_embeddings)
            # ******************************** END addition
            else:
                try:
                    labels = self.hdbscan_model.labels_
                except AttributeError:
                    labels = y
            documents['Topic'] = labels
            self._update_topic_size(documents)

        # Some algorithms have outlier labels (-1) that can be tricky to work
        # with if you are slicing data based on that labels. Therefore, we
        # track if there are outlier labels and act accordingly when slicing.
        self._outliers = 1 if -1 in set(labels) else 0

        # Extract probabilities
        probabilities = None
        if hasattr(self.hdbscan_model, "probabilities_"):
            probabilities = self.hdbscan_model.probabilities_

            if self.calculate_probabilities and is_supported_hdbscan(self.hdbscan_model):
                probabilities = hdbscan_delegator(self.hdbscan_model, "all_points_membership_vectors")

        if not partial_fit:
            self.topic_mapper_ = TopicMapper(self.topics_)
        logger.info("Clustered reduced embeddings")
        return documents, probabilities
