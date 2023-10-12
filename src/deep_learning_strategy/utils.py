from __future__ import annotations

from pathlib import Path

import shap
from matplotlib import pyplot as plt
from shap.models import TransformersPipeline


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
