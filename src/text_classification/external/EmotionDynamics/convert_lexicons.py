from pathlib import Path

import pandas as pd

LEXICONS = ["NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "NRC-VAD-Lexicon.txt", "NRC-Hashtag-Emotion-Lexicon-v0.2.txt"]
BASE_FOLDER = Path("src") / "text_classification" / "external" / "EmotionDynamics" / "lexicons"

if __name__ == "__main__":
    for lexicon in LEXICONS:
        if lexicon == "NRC-VAD-Lexicon.txt":
            df = pd.read_csv(BASE_FOLDER / lexicon, sep="\t", header=None,
                             names=["word", "valence", "arousal", "domination"])
            df.to_csv((BASE_FOLDER / lexicon).with_suffix(".csv"), index=False)
        elif lexicon == "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt":
            data = pd.read_csv(BASE_FOLDER / lexicon, sep="\t", header=None, names=["word", "emotion", "value"])
            df = data.pivot(columns="emotion", index="word", values="value")
            df = df[df.index.notnull()]
            df.to_csv((BASE_FOLDER / lexicon).with_suffix(".csv"), index=True)
        elif lexicon == "NRC-Hashtag-Emotion-Lexicon-v0.2.txt":
            data = pd.read_csv(BASE_FOLDER / lexicon, sep="\t", header=None, names=["emotion", "word", "value"])
            df = data.pivot(columns="emotion", index="word", values="value")
            df = df[df.index.notnull()].fillna(.0)  # is it correct to set this to 0?
            df.to_csv((BASE_FOLDER / lexicon).with_suffix(".csv"), index=True)
        else:
            ValueError(f"Unrecognized lexicon '{lexicon}'.\nSupported lexicon are: [{', '.join(LEXICONS)}]")
