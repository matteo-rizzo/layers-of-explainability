from pathlib import Path

import language_tool_python
import pandas as pd

from src.features_extraction.classes.Feature import Feature
from src.text_classification.external.EmotionDynamics.avgEmoValues import read_lexicon, prep_dim_lexicon, get_vals


class EmotionLex(Feature):
    LEXICONS = ["NRC-Emotion-Lexicon-Wordlevel-v0.92.csv", "NRC-VAD-Lexicon.csv",
                "NRC-Hashtag-Emotion-Lexicon-v0.2.csv"]
    BASE_FOLDER = Path("src") / "text_classification" / "external" / "EmotionDynamics" / "lexicons"

    def __init__(self, use_correction: bool = False, *args, **kwargs):
        self.correct_text = use_correction
        self.pipe = language_tool_python.LanguageTool("en-US")

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        dataframes = list()

        if self.correct_text:
            texts = [self.pipe.correct(t) for t in texts]

        for lex in self.LEXICONS:
            lexnames = pd.read_csv(self.BASE_FOLDER / lex).columns.tolist()
            lexnames.remove("word")

            lexicon = read_lexicon(self.BASE_FOLDER / lex, lexnames)

            for LEXNAME in lexnames:
                lexdf = prep_dim_lexicon(lexicon, LEXNAME)

                # Compute lexicon matching
                resrows = [get_vals(x, lexdf) for x in texts]

                resdf = pd.DataFrame(resrows, columns=["numTokens", "numLexTokens", "avgLexVal"])
                resdf = resdf.fillna(.0)
                resdf.drop(columns=["numTokens", "numLexTokens"], inplace=True)
                resdf.rename(
                    columns={k: f"{self.__class__.__name__}_{str(Path(lex).with_suffix(''))}_{LEXNAME}_{k}" for k in
                             ["avgLexVal"]}, inplace=True)
                dataframes.append(resdf)

        result_df = pd.concat(dataframes, axis=1)
        assert not result_df.isna().any().any(), "There are NaN values in produced features"
        return result_df.to_dict(orient="list")

    @classmethod
    def label_description(cls) -> dict[str, str]:
        labels = ['EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_anger_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_anticipation_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_disgust_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_fear_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_joy_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_negative_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_positive_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_sadness_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_surprise_avgLexVal',
                  'EmotionLex_NRC-Emotion-Lexicon-Wordlevel-v0.92_trust_avgLexVal',
                  'EmotionLex_NRC-VAD-Lexicon_valence_avgLexVal', 'EmotionLex_NRC-VAD-Lexicon_arousal_avgLexVal',
                  'EmotionLex_NRC-VAD-Lexicon_domination_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_anger_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_anticipation_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_disgust_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_fear_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_joy_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_sadness_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_surprise_avgLexVal',
                  'EmotionLex_NRC-Hashtag-Emotion-Lexicon-v0.2_trust_avgLexVal']

        emotions = [f"text '{e.split('_')[2]}' score" for e in labels]
        d = dict(zip(labels, emotions))

        return d
