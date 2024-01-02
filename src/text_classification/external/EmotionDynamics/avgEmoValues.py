import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

parser = ArgumentParser()
parser.add_argument("--dataPath", help="path to CSV data file with texts",
                    default="src/text_classification/external/EmotionDynamics/sample_data/sample_input.csv")
parser.add_argument("--lexPath", help="path to lexicon. CSV with columns \"word\" plus emotion columns",
                    default="src/text_classification/external/EmotionDynamics/lexicons/NRC-VAD-Lexicon.csv")
parser.add_argument("--lexNames", nargs="*", type=str, help="Names of the lexicons/column names in the lexicon CSV")
parser.add_argument("--savePath", help="path to save folder",
                    default="src/text_classification/external/EmotionDynamics/output")


def read_lexicon(path, LEXNAMES):
    df = pd.read_csv(path)
    df = df[~df['word'].isna()]
    df = df[['word'] + LEXNAMES]
    df['word'] = [x.lower() for x in df['word']]
    return df
    # df = df[~df['val'].isna()]


def prep_dim_lexicon(df, dim):
    ldf = df[['word'] + [dim]]
    ldf = ldf[~ldf[dim].isna()]
    ldf.drop_duplicates(subset=['word'], keep='first', inplace=True)
    ldf[dim] = [float(x) for x in ldf[dim]]
    ldf.rename({dim: 'val'}, axis='columns', inplace=True)
    ldf.set_index('word', inplace=True)
    return ldf


def get_alpha(token):
    return token.isalpha()


def get_vals(twt, lexdf):
    tt = twt.lower().split(" ")
    at = [w for w in tt if w.isalpha()]

    pw = [x for x in tt if x in lexdf.index]
    pv = [lexdf.loc[w]['val'] for w in pw]

    numTokens = len(at)
    numLexTokens = len(pw)

    avgLexVal = np.mean(pv)  # nan for 0 tokens

    return [numTokens, numLexTokens, avgLexVal]


def process_df(df, lexdf):
    logging.info("Number of rows: " + str(len(df)))

    resrows = [get_vals(x, lexdf) for x in df["text"]]
    resrows = [x + y for x, y in zip(df.values.tolist(), resrows)]

    resdf = pd.DataFrame(resrows, columns=df.columns.tolist() + ["numTokens", "numLexTokens", "avgLexVal"])
    # resdf = resdf[resdf[f"{lexname}_numLexTokens"] >= 1]

    resdf["lexRatio"] = resdf["numLexTokens"] / resdf["numTokens"]
    resdf = resdf.fillna(.0)
    return resdf


def extract_emotions(dataPath, LEXICON, LEXNAMES, savePath):
    os.makedirs(savePath, exist_ok=True)

    logfile = os.path.join(savePath, "log.txt")

    logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    df = pd.read_csv(dataPath)

    dataframes = list()

    for LEXNAME in LEXNAMES:
        lexdf = prep_dim_lexicon(LEXICON, LEXNAME)
        logging.info(LEXNAME + " lexicon length: " + str(len(lexdf)))
        resdf = process_df(df, lexdf)
        if len(dataframes) > 0:
            resdf = resdf.loc[:, ["numLexTokens", "avgLexVal", "lexRatio"]]
        resdf.rename(columns={k: f"{LEXNAME}_{k}" for k in ["numLexTokens", "avgLexVal", "lexRatio"]}, inplace=True)
        dataframes.append(resdf)
        # resdf.to_csv(os.path.join(savePath, LEXNAME + '.csv'), index=False)

    result_df = pd.concat(dataframes, axis=1)
    result_df.to_csv(os.path.join(savePath, "all_lex" + '.csv'), index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    dataPath = args.dataPath
    lexPath = args.lexPath

    LEXNAMES = args.lexNames
    LEXICON = read_lexicon(lexPath, LEXNAMES)

    savePath = args.savePath

    extract_emotions(dataPath, LEXICON, LEXNAMES, savePath)
