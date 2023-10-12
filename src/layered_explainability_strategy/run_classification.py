import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.feature_extraction.text_features import TextFeatureExtractor
from src.utils.yaml_manager import load_yaml


def load_data(df: pd.DataFrame,
              keep_text: bool = False,
              fit_vectorizer: bool = True,
              vectorizer: TfidfVectorizer = None):
    # -----------------------------------------------
    shuffled = df.sample(frac=1)
    ds = shuffled.drop(columns=["text"]).to_numpy()
    X, y = ds[:, 0:4], ds[:, 4]

    if keep_text:
        if vectorizer is None:
            raise ValueError("if keep_text is true, a vectorizer must be specified")
        text = shuffled["text"].to_list()
        vectors = (vectorizer.fit_transform(text) if fit_vectorizer else vectorizer.transform(text)).toarray()
        # Consider other ways of concatenating
        X = np.concatenate((X, vectors), axis=1)
    return X, y


def main():
    # -----------------------------------------------
    train = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv")
    fex = TextFeatureExtractor(language="en")
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    train_x, train_y = load_data(train, keep_text=True, vectorizer=bow_vectorizer, fit_vectorizer=True)
    test_x, test_y = load_data(test, keep_text=True, vectorizer=bow_vectorizer, fit_vectorizer=False)
    # -----------------------------------------------
    params: dict = load_yaml("src/layered_explainability_strategy/config.yml")
    gs = GridSearchCV(DecisionTreeClassifier(), n_jobs=4,
                      param_grid=params["DecisionTreeClassifier"], verbose=10, refit=True)
    gs.fit(X=train_x, y=train_y)
    predictions = gs.predict(X=test_x)
    # -----------------------------------------------
    f1 = f1_score(y_true=test_y, y_pred=predictions)
    acc = accuracy_score(y_true=test_y, y_pred=predictions)
    print("---------------------------------------")
    print(gs.best_params_)
    print(f"f1: {f1:.3f} acc: {acc:.3f}")
    # -----------------------------------------------


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    main()
