from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Preprocess text (username and link placeholders)
def preprocess(text):
    # FIXME: plaholder
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

    print(f"*** {task} ***")
    for s, l in zip(scores, labels):
        print(f"{l.title():<10}: {s:.3f}")


def main():
    text = "What takes up 12 parking spaces? 6 women drivers"
    print("Text:")
    print(text)
    # -----------------------------------------------
    task = 'irony'  # irony, non irony
    model = 'cardiffnlp/twitter-roberta-base-irony'
    labels = ['non-irony', 'irony']
    run_model(model, task, text, labels)
    # -----------------------------------------------
    task = 'sentiment-latest'  # negative, neutral, positive
    model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    labels = ['negative', 'neutral', 'positive']
    run_model(model, task, text, labels)
    # task = 'polarity'  # 0 negative, 1 positive
    # model = 'VictorSanh/roberta-base-finetuned-yelp-polarity'
    # -----------------------------------------------
    task = 'hate'  # non-hate / hate
    model = 'cardiffnlp/twitter-roberta-base-hate'
    labels = ['non-hate', 'hate']
    run_model(model, task, text, labels)
    # -----------------------------------------------
    task = 'offensive'  # non-offensive, offensive
    model = 'cardiffnlp/twitter-roberta-base-offensive'
    labels = ['non-offensive', 'offensive']
    run_model(model, task, text, labels)
    # -----------------------------------------------
    task = 'gender'  # Male, Female
    model = 'padmajabfrl/Gender-Classification'
    labels = ["female", "male"]
    run_model(model, task, text, labels)


if __name__ == "__main__":
    main()
