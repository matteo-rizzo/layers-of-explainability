from pathlib import Path

import numpy as np
import shap
from matplotlib import pyplot as plt
from src.explainable_strategy.transhap.explainers import visualize_explanations
from nltk import TweetTokenizer

from src.explainable_strategy.transhap.explainers.SHAP_for_text import SHAPexplainer


def shap_explain(texts: list[str], model, tokenizer, target_label: str, show: bool = True):
    pmodel = shap.models.TransformersPipeline(model, rescale_to_logits=True)

    explainer = shap.Explainer(pmodel, tokenizer)
    shap_values = explainer(texts)

    target_dir = Path("plots") / "nlp"
    target_dir.mkdir(parents=True, exist_ok=True)

    # fig = plt.figure()
    # shap.plots.text(shap_values[0, :, target_label])

    for i, t in enumerate(texts):
        shap.plots.bar(shap_values[i, :, target_label], max_display=25, show=False)
        plt.title(f"Sentence: {t}")
        plt.gcf().set_size_inches(10, 14)
        plt.tight_layout()
        plt.savefig(target_dir / f"shap_sentence_barplot_{i}.png", dpi=400)
        plt.clf()

    shap.plots.bar(shap_values[:, :, target_label].mean(0), max_display=25, show=False)
    plt.title("Mean relative importance of tokens for classification")
    plt.gcf().set_size_inches(10, 14)
    plt.tight_layout()

    plt.savefig(target_dir / "shap_mean_barplot.png", dpi=400)
    if show:
        plt.show()


def transhap_explain(texts: list[str], explain_ids: list[int], model, tokenizer, target_label: str, show: bool = True, device: str = None):
    word_tokenizer = TweetTokenizer()
    bag_of_words = set([xx for x in texts for xx in word_tokenizer.tokenize(x)])

    words_dict = {0: None}
    words_dict_reverse = {None: 0}
    for h, hh in enumerate(bag_of_words):
        words_dict[h + 1] = hh
        words_dict_reverse[hh] = h + 1

    predictor = SHAPexplainer(model, tokenizer, words_dict, words_dict_reverse, device=device)
    train_dt = np.array([predictor.split_string(x) for x in np.array(texts)])
    idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)

    explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=50))

    texts_ = [predictor.split_string(x) for x in texts]
    idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

    idx_texts_to_use = np.asarray([idx_texts[a] for a in explain_ids])
    tokenized_texts_ = [texts_[a] for a in explain_ids]
    shap_values = explainer.shap_values(X=idx_texts_to_use, nsamples=64, l1_reg="aic")

    # Explain each ID
    for idx, j in enumerate(range(len(explain_ids))):
        len_ = len(tokenized_texts_[j])
        d = {i: sum(x > 0 for x in shap_values[i][j, :len_]) for i, x in enumerate(shap_values)}
        m = max(d, key=d.get)
        print(" ".join(tokenized_texts_[j]))
        shap.force_plot(explainer.expected_value[m], shap_values[m][j, :len_], tokenized_texts_[j], matplotlib=True)

        # Barplot

        text_j = tokenized_texts_[j]
        idx_to_use = idx_texts_to_use[j].reshape(1, -1)
        f = predictor.predict(idx_to_use)
        pred_f = np.argmax(f[0])

        visualize_explanations.joint_visualization("plots/nlp", text_j, shap_values[pred_f][j, :len(text_j)], ["Non-hateful", "Hateful"][int(pred_f)], f[0][pred_f], idx)
        plt.show()
