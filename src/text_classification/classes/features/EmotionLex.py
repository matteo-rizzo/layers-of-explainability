from pathlib import Path

import language_tool_python
import pandas as pd

from src.text_classification.classes.features.Feature import Feature
from src.text_classification.external.EmotionDynamics.avgEmoValues import read_lexicon, get_vals, prep_dim_lexicon


class EmotionLex(Feature):
    LEXICONS = ["NRC-Emotion-Lexicon-Wordlevel-v0.92.csv", "NRC-VAD-Lexicon.csv", "NRC-Hashtag-Emotion-Lexicon-v0.2.csv"]
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
                # resdf = resdf[resdf[f"{lexname}_numLexTokens"] >= 1]

                # resdf["lexRatio"] = resdf["numLexTokens"] / resdf["numTokens"]
                resdf = resdf.fillna(.0)

                # if len(dataframes) == 0:
                #     resdf["text"] = texts

                resdf.drop(columns=["numTokens", "numLexTokens"], inplace=True)
                resdf.rename(columns={k: f"{self.__class__.__name__}_{str(Path(lex).with_suffix(''))}_{LEXNAME}_{k}" for k in ["avgLexVal"]}, inplace=True)
                dataframes.append(resdf)
                # resdf.to_csv(os.path.join(savePath, LEXNAME + '.csv'), index=False)

        result_df = pd.concat(dataframes, axis=1)
        assert not result_df.isna().any().any(), "There are NaN values in produced features"
        return result_df.to_dict(orient="list")
