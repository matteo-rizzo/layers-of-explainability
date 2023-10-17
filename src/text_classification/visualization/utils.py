import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP


def visualize_topic_space_data(topic_model, umap_model,
                               topics: list[int] = None,
                               top_n_topics: int = None) -> pd.DataFrame:
    """ Visualize topics, their sizes, and their corresponding words

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_topics()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/viz.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    # I use sentence embeddings because Tf-IDF is too problematic
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    # embeddings = topic_model.c_tf_idf_.toarray()[indices]
    embeddings = np.array(topic_model.topic_embeddings_)[indices]
    # embeddings = MinMaxScaler().fit_transform(embeddings)
    print(embeddings.shape)
    if not umap_model.fitted:
        umap_model.fit(embeddings)
        umap_model.fitted = True
    embeddings = umap_model.transform(embeddings)
    # embeddings = umap_model.fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})
    return df
