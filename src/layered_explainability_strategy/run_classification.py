import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.feature_extraction.text_features import TextFeatureExtractor
from src.utils.yaml_manager import load_yaml


def load_data(df: pd.DataFrame,
              keep_text: bool = False,
              binarize: bool = False,
              threshold: float = 0.5,
              fit_vectorizer: bool = True,
              vectorizer: TfidfVectorizer = None,
              return_all: bool = False):
    # -----------------------------------------------
    # FIXME: brutta but idc atm
    shuffled = df.sample(frac=1)
    ds = shuffled.drop(columns=["text"])
    if binarize:
        # Skip label column
        for column in ds.columns[:-1]:
            ds[column] = (ds[column] > threshold).astype(int)
    ds = ds.to_numpy()
    features_X, y = ds[:, 0:4], ds[:, 4]

    if keep_text or return_all:
        if vectorizer is None:
            raise ValueError("if keep_text is true, a vectorizer must be specified")
        text = shuffled["text"].to_list()
        vectors_X = (vectorizer.fit_transform(text) if fit_vectorizer else vectorizer.transform(text)).toarray()
        if return_all:
            return vectors_X, features_X, y
        # Consider other ways of concatenating
        features_X = np.concatenate((features_X, vectors_X), axis=1)
    return features_X, y


def classify_subfeatures():
    # -----------------------------------------------
    train = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv")
    # fex = TextFeatureExtractor(language="en")
    # bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
    #                                  ngram_range=(1, 3),
    #                                  max_features=10000,
    #                                  token_pattern=None)
    #
    # train_x, train_y = load_data(train, keep_text=True,
    #                              vectorizer=bow_vectorizer, fit_vectorizer=True)
    # test_x, test_y = load_data(test, keep_text=True,
    #                            vectorizer=bow_vectorizer, fit_vectorizer=False)
    train_x, train_y = load_data(train, keep_text=False, binarize=False)
    test_x, test_y = load_data(test, keep_text=False, binarize=False)
    # -----------------------------------------------
    params: dict = load_yaml("src/layered_explainability_strategy/config.yml")
    # DecisionTreeClassifier
    # RidgeClassifier
    gs = GridSearchCV(DecisionTreeClassifier(), n_jobs=4,
                      param_grid=params["gs"]["DecisionTreeClassifier"], verbose=10, refit=True)
    gs.fit(X=train_x, y=train_y)
    predictions = gs.predict(X=test_x)
    # -----------------------------------------------
    f1 = f1_score(y_true=test_y, y_pred=predictions)
    acc = accuracy_score(y_true=test_y, y_pred=predictions)
    print("---------------------------------------")
    print(gs.best_params_)
    print(f"f1: {f1:.3f} acc: {acc:.3f}")
    # -----------------------------------------------


def classify_tfidf():
    train = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv")

    fex = TextFeatureExtractor(language="en")
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    train_x, _, train_y = load_data(train, return_all=True, vectorizer=bow_vectorizer, fit_vectorizer=True)
    test_x, _, test_y = load_data(test, return_all=True, vectorizer=bow_vectorizer, fit_vectorizer=False)

    # -----------------------------------------------
    params: dict = load_yaml("src/layered_explainability_strategy/config.yml")
    # DecisionTreeClassifier
    # RidgeClassifier
    gs = GridSearchCV(RidgeClassifier(), n_jobs=4,
                      param_grid=params["gs"]["RidgeClassifier"], verbose=10, refit=True)
    gs.fit(X=train_x, y=train_y)
    predictions = gs.predict(X=test_x)
    # -----------------------------------------------
    f1 = f1_score(y_true=test_y, y_pred=predictions)
    acc = accuracy_score(y_true=test_y, y_pred=predictions)
    print("---------------------------------------")
    print(gs.best_params_)
    print(f"f1: {f1:.3f} acc: {acc:.3f}")
    # -----------------------------------------------


def mini_ensemble():
    train = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv")
    params: dict = load_yaml("src/layered_explainability_strategy/config.yml")

    fex = TextFeatureExtractor(language="en")
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    train_vectors_x, train_features_x, train_y = load_data(train, return_all=True, vectorizer=bow_vectorizer,
                                                           fit_vectorizer=True)
    test_vectors_x, test_features_x, test_y = load_data(test, return_all=True, vectorizer=bow_vectorizer,
                                                        fit_vectorizer=False)
    feature_clf = DecisionTreeClassifier(**params["DecisionTreeClassifier"])
    tfd_idf_clf = RidgeClassifier(**params["RidgeClassifier"])

    feature_clf.fit(train_features_x, train_y)
    tfd_idf_clf.fit(train_vectors_x, train_y)

    feature_predictions = feature_clf.predict(X=test_features_x)
    vector_predictions = tfd_idf_clf.predict(X=test_vectors_x)

    # -----------------------------------------------
    feature_f1 = f1_score(y_true=test_y, y_pred=feature_predictions)
    feature_acc = accuracy_score(y_true=test_y, y_pred=feature_predictions)

    vector_f1 = f1_score(y_true=test_y, y_pred=vector_predictions)
    vector_acc = accuracy_score(y_true=test_y, y_pred=vector_predictions)
    print("---------------------------------------")
    print(f"[FEATURES] f1: {feature_f1:.3f} acc: {feature_acc:.3f} "
          f"/ [VECTORS] f1: {vector_f1:.3f} acc: {vector_acc:.3f}")
    # -----------------------------------------------
    train_feature_predictions = feature_clf.predict(X=train_features_x)
    train_vector_predictions = tfd_idf_clf.predict(X=train_vectors_x)

    test_feature_predictions = feature_predictions
    test_vector_predictions = vector_predictions

    gs = GridSearchCV(DecisionTreeClassifier(), n_jobs=4,
                      param_grid=params["gs"]["DecisionTreeClassifier"], verbose=10, refit=True)
    gs.fit(X=np.stack((train_feature_predictions, train_vector_predictions), axis=1), y=train_y)
    predictions = gs.predict(X=np.stack((test_feature_predictions, test_vector_predictions), axis=1))
    # -----------------------------------------------
    aggregate_f1 = f1_score(y_true=test_y, y_pred=predictions)
    aggregate_acc = accuracy_score(y_true=test_y, y_pred=predictions)
    print("---------------------------------------")
    print(gs.best_params_)
    print(f"[AGGREGATE] f1: {aggregate_f1:.3f} acc: {aggregate_acc:.3f}")
    # -----------------------------------------------


if __name__ == "__main__":
    # classify_subfeatures()
    # classify_tfidf()
    mini_ensemble()
