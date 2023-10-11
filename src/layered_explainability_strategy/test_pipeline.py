import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, logging

logging.set_verbosity_error()


# Preprocess text (username and link placeholders)
def preprocess(text):
    # FIXME: placeholder
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def run_model(model, task, text, labels):
    print("-------------------------------------------------------")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)

    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    print(f"*** {task.title()} ***")
    for s, l in zip(scores, labels):
        print(f"{l.title():<10}: {s:.3f}")


def run_t5_model(model, task, text):
    print("-------------------------------------------------------")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=3)

    dec = [tokenizer.decode(ids) for ids in output]

    if "derision" in dec[0]:
        label = "sarcasm"
    elif "normal" in dec[0]:
        label = "non-sarcasm"
    else:
        raise ValueError("Error in T5 output")
    print(f"*** {task.title()} ***")
    print(f"{label.title():<10}")


def test_pipolino():
    # text = "My Mather is a slut bitch"
    t1 = "Please tell me why the bitch next to me in the piercing line is so judgmental about everyone she fucking sees. SHUT THE FUCK UP."
    t2 = "<MENTION_1> <MENTION_2> Bitch shut the fuck up"
    t3 = "<MENTION_1> Dear cunt, please shut the fuck up."
    t4 = "RT <MENTION_1> Pls shut the fuck up bitch"
    t5 = 'RT <MENTION_1> "when u gonna get your license" SHUT THE FUCK UP BITCH I AINT GOT TIME DAMN GET OFF MY DICK'
    t = [t1, t2, t3, t4, t5]
    for text in t:
        print("Text:")
        print(f"'{text}'")
        # -----------------------------------------------
        task = 'irony'  # irony, non irony
        model = 'cardiffnlp/twitter-roberta-base-irony'
        labels = ['non-irony', 'irony']
        run_model(model, task, text, labels)
        # -----------------------------------------------
        # task = 'sarcasm'  # sarcasm, non sarcasm
        # model = 'mrm8488/t5-base-finetuned-sarcasm-twitter'
        # run_t5_model(model, task, text)
        # # -----------------------------------------------
        task = 'sarcasm'  # sarcasm, non sarcasm
        model = 'helinivan/english-sarcasm-detector'
        labels = ['non-sarcasm', 'sarcasm']
        run_model(model, task, text, labels)
        # -----------------------------------------------
        # task = 'sentiment-latest'  # negative, neutral, positive
        # model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        # labels = ['negative', 'neutral', 'positive']
        # run_model(model, task, text, labels)
        # # task = 'polarity'  # 0 negative, 1 positive
        # # model = 'VictorSanh/roberta-base-finetuned-yelp-polarity'
        # # -----------------------------------------------
        # task = 'hate'  # non-hate / hate
        # model = 'cardiffnlp/twitter-roberta-base-hate'
        # labels = ['non-hate', 'hate']
        # run_model(model, task, text, labels)
        # -----------------------------------------------
        task = 'offensive'  # non-offensive, offensive
        model = 'cardiffnlp/twitter-roberta-base-offensive'
        labels = ['non-offensive', 'offensive']
        run_model(model, task, text, labels)
        # -----------------------------------------------
        task = 'gender'  # Male, Female
        # model = 'padmajabfrl/Gender-Classification'
        # labels = ["male", "female"]
        model = 'thaile/roberta-base-md_gender_bias-saved'
        labels = ["female", "male"]
        run_model(model, task, text, labels)


def get_pred(model, all_texts, opposite: bool = False, task: str = ""):
    batch_size = 128
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model).to("cuda:0")
    texts = [preprocess(t) for t in all_texts]

    scores = []
    for i in tqdm(range((len(all_texts) // batch_size) + 1), desc=f"{task.title()} inference"):
        low = i * batch_size
        high = (i + 1) * batch_size
        current_batch = texts[low:high]
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
    test = pd.read_table("dataset/ami2018_misogyny_detection/en_testing_labeled_anon.tsv")

    print("Generating train dataset...")
    train_df = create_compound_df(train)
    print("Generating test dataset...")
    test_df = create_compound_df(test)

    print("Saving to file...")
    train_df.to_csv("dataset/ami2018_misogyny_detection/processed/en_training_anon.tsv", sep='\t', index=False)
    test_df.to_csv("dataset/ami2018_misogyny_detection/processed/en_testing_labeled_anon.tsv", sep='\t', index=False)
    print("")


if __name__ == "__main__":
    create_compound_dataset()
