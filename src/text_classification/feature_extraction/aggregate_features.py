from pathlib import Path

import pandas as pd

from src.text_classification.feature_extraction.TextEmotion import TextEmotion
from src.text_classification.feature_extraction.TextGrammarErrors import TextGrammarErrors
from src.text_classification.feature_extraction.TextPolarity import TextPolarity


def compute_features():
    data_path = Path("dataset") / "fake_reviews_dataset.csv"

    data = pd.read_csv(data_path)

    texts = data["text_"].tolist()

    feature_transforms = [TextPolarity, TextGrammarErrors, TextEmotion]
    feature_transforms = [f() for f in feature_transforms]

    all_features = dict()
    for f in feature_transforms:
        res = f.extract(texts)
        all_features.update(res)

    pd.DataFrame.from_dict(all_features).to_csv(data_path.parent / "fake_reviews_features.csv", index=False)


if __name__ == "__main__":
    compute_features()
