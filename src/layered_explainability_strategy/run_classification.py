import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.feature_extraction.text_features import TextFeatureExtractor
from src.layered_explainability_strategy.bad_word_finder import examine_docs
from src.utils.yaml_manager import load_yaml


def load_data(df: pd.DataFrame,
              binarize: bool = False,
              threshold: float = 0.5,
              fit_vectorizer: bool = True,
              vectorizer: TfidfVectorizer = None,
              bad_words: bool = False):
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

    if vectorizer is None:
        raise ValueError("if keep_text is true, a vectorizer must be specified")
    text = shuffled["text"].to_list()
    vectors_X = (vectorizer.fit_transform(text) if fit_vectorizer else vectorizer.transform(text)).toarray()
    if bad_words:
        bad_word_count: list[float] = examine_docs(text)
        features_X = np.hstack((np.array(bad_word_count)[:, np.newaxis], features_X))
    return vectors_X, features_X, y


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
    _, train_x, train_y = load_data(train, binarize=False)
    _, test_x, test_y = load_data(test, binarize=False)
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

    train_x, _, train_y = load_data(train, vectorizer=bow_vectorizer, fit_vectorizer=True)
    test_x, _, test_y = load_data(test, vectorizer=bow_vectorizer, fit_vectorizer=False)

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


def fit_predict_metric(clf, train_X, test_X, train_y, test_y):
    clf.fit(train_X, train_y)
    preds = clf.predict(X=test_X)
    f1 = f1_score(y_true=test_y, y_pred=preds)
    acc = accuracy_score(y_true=test_y, y_pred=preds)
    return f1, acc, preds


def mini_ensemble():
    train = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv")
    params: dict = load_yaml("src/layered_explainability_strategy/config.yml")
    # ----------------------------------------------------------
    # Feture Extration
    fex = TextFeatureExtractor(language="en")
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    train_vectors_x, train_features_x, train_y = load_data(train, vectorizer=bow_vectorizer,
                                                           bad_words=True,
                                                           fit_vectorizer=True)
    test_vectors_x, test_features_x, test_y = load_data(test, vectorizer=bow_vectorizer,
                                                        bad_words=True,
                                                        fit_vectorizer=False)
    # ----------------------------------------------------------
    # Classification
    feature_clf = DecisionTreeClassifier(**params["DecisionTreeClassifier"])
    tfd_idf_clf = RidgeClassifier(**params["RidgeClassifier"])

    feature_f1, feature_acc, feature_predictions = fit_predict_metric(feature_clf,
                                                                      train_X=train_features_x,
                                                                      test_X=test_features_x,
                                                                      train_y=train_y,
                                                                      test_y=test_y)

    vector_f1, vector_acc, vector_predictions = fit_predict_metric(tfd_idf_clf,
                                                                   train_X=train_vectors_x,
                                                                   test_X=test_vectors_x,
                                                                   train_y=train_y,
                                                                   test_y=test_y)
    print("---------------------------------------")
    print(f"[FEATURES] f1: {feature_f1:.3f} acc: {feature_acc:.3f} "
          f"/ [VECTORS] f1: {vector_f1:.3f} acc: {vector_acc:.3f}")
    # -----------------------------------------------
    # Ensemble - Pred + Pred
    train_feature_predictions = feature_clf.predict(X=train_features_x)
    train_vector_predictions = tfd_idf_clf.predict(X=train_vectors_x)

    test_feature_predictions = feature_predictions
    test_vector_predictions = vector_predictions

    gs = GridSearchCV(DecisionTreeClassifier(), n_jobs=4,
                      param_grid=params["gs"]["DecisionTreeClassifier"], verbose=0, refit=True)
    gs.fit(X=np.stack((train_feature_predictions, train_vector_predictions), axis=1), y=train_y)
    predictions = gs.predict(X=np.stack((test_feature_predictions, test_vector_predictions), axis=1))

    aggregate_f1 = f1_score(y_true=test_y, y_pred=predictions)
    aggregate_acc = accuracy_score(y_true=test_y, y_pred=predictions)
    print("---------------------------------------")
    print(gs.best_params_)
    print(f"[Pred + Pred] f1: {aggregate_f1:.3f} acc: {aggregate_acc:.3f}")
    # # -----------------------------------------------
    # Ensemble - Feat + Pred
    gs = GridSearchCV(DecisionTreeClassifier(), n_jobs=4,
                      param_grid=params["gs"]["DecisionTreeClassifier"], verbose=0, refit=True)
    gs.fit(X=np.hstack((train_features_x, train_feature_predictions[:, np.newaxis])), y=train_y)
    predictions = gs.predict(X=np.hstack((test_features_x, test_vector_predictions[:, np.newaxis])))

    aggregate_f1 = f1_score(y_true=test_y, y_pred=predictions)
    aggregate_acc = accuracy_score(y_true=test_y, y_pred=predictions)
    print("---------------------------------------")
    print(gs.best_params_)
    print(f"[Feat + Pred] f1: {aggregate_f1:.3f} acc: {aggregate_acc:.3f}")
    # -----------------------------------------------


if __name__ == "__main__":
    # classify_subfeatures()
    # classify_tfidf()
    mini_ensemble()
