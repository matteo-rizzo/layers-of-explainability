from pathlib import Path

import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

from src.layered_explainability_strategy.test_pipeline import preprocess

logging.set_verbosity_error()


def get_pred(model, all_texts, opposite: bool = False, task: str = ""):
    batch_size = 4
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model).to("cuda:0")
    texts = [preprocess(t) for t in all_texts]

    scores = []
    for i in tqdm(range((len(all_texts) // batch_size) + 1), desc=f"{task.title()} inference"):
        low = i * batch_size
        high = (i + 1) * batch_size
        current_batch = texts[low:high]
        if not current_batch:
            continue
        encoded_input = tokenizer(current_batch, return_tensors='pt', padding=True).to("cuda:0")
        output = model(**encoded_input)
        scores.extend(output[0].cpu().detach().tolist())
    scores = softmax(scores, axis=1)

    score_label_1 = scores[:, 1].tolist()
    if opposite:
        score_label_1 = [1 - x for x in score_label_1]

    del model
    torch.cuda.empty_cache()
    return score_label_1


def extract_lm_features(all_texts):
    model = 'cardiffnlp/twitter-roberta-base-irony'
    irony = get_pred(model, all_texts, task="irony")
    model = 'helinivan/english-sarcasm-detector'
    sarcasm = get_pred(model, all_texts, task="sarcasm")
    model = 'cardiffnlp/twitter-roberta-base-offensive'
    offensiveness = get_pred(model, all_texts, task="offense")
    model = 'thaile/roberta-base-md_gender_bias-saved'
    gender_female = get_pred(model, all_texts, opposite=True, task="gender")

    return irony, sarcasm, offensiveness, gender_female


def create_compound_df(ds):
    ds_list = []
    # for idx, row in ds.iterrows():
    ids, text, misogynous, misogyny_category, target = ds["id"].tolist(), ds["text"].tolist(), ds[
        "misogynous"].tolist(), ds["misogyny_category"].tolist(), ds["target"].tolist()
    irony, sarcasm, offensiveness, gender_female = extract_lm_features(text)
    # ds_list.append((text, irony, sarcasm, offensiveness, gender_female, misogynous))
    ds_list = [x for x in zip(text, irony, sarcasm, offensiveness, gender_female, misogynous)]
    df = pd.DataFrame(ds_list, columns=["text", "irony", "sarcasm", "offensive", "female", "label"])
    return df


def create_compound_dataset():
    train = pd.read_table("dataset/ami2018_misogyny_detection/en_training_anon.tsv")
    test = pd.read_table("dataset/ami2018_  misogyny_detection/en_testing_labeled_anon.tsv")

    print("Generating train dataset...")
    train_df = create_compound_df(train)
    print("Generating test dataset...")
    test_df = create_compound_df(test)

    print("Saving to file...")
    root = Path("dataset/ami2018_misogyny_detection/processed")
    root.mkdir(exist_ok=True)
    train_df.to_csv(root / "en_training_anon.tsv", sep='\t', index=False)
    test_df.to_csv(root / "en_testing_labeled_anon.tsv", sep='\t', index=False)
    print("")


if __name__ == "__main__":
    create_compound_dataset()
