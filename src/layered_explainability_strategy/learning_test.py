import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.utils.yaml_manager import load_yaml


def main():
    # -----------------------------------------------
    params: dict = load_yaml("src/layered_explainability_strategy/config.yml")
    train = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv")
    # -----------------------------------------------
    train = train.sample(frac=1)
    train_ds = train.drop(columns=["text"]).to_numpy()
    train_x, train_y = train_ds[:, 0:4], train_ds[:, 4]
    # -----------------------------------------------
    test = test.sample(frac=1)
    test_ds = test.drop(columns=["text"]).to_numpy()
    test_x, test_y = test_ds[:, 0:4], test_ds[:, 4]
    # -----------------------------------------------
    gs = GridSearchCV(DecisionTreeClassifier(), n_jobs=8,
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
    main()
