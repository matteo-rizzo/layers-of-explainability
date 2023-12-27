from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.text_classification.TMP_simple_model.text_features import TextFeatureExtractor


def make_pipeline(sk_classifier: ClassifierMixin) -> Pipeline:
    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('classifier', sk_classifier)])
    return pipe


def naive_classifier(sk_classifier: ClassifierMixin, training_data: dict[str, dict[str, list]],
                     return_pipe: bool = False, predict: bool = True) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    pipe = make_pipeline(sk_classifier)

    print("------ Training")

    pipe.fit(training_data["train"]["x"], training_data["train"]["y"])

    predicted = None
    if predict:
        print("------ Testing")

        # Predicting with a test dataset
        predicted = pipe.predict(training_data["test"]["x"])

    if not return_pipe:
        return predicted
    else:
        return predicted, pipe
