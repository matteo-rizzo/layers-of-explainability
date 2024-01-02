from __future__ import annotations

from numbers import Number

from src.text_classification.utils import capitalize_first_letter


def build_explanation(text: str, y_pred: int, y_true: int, prob: float, features: list[tuple[str, float, float]],
                      feature_names: dict[str, str] | None,
                      label_names: dict[int, str], words_mode: bool = False) -> str:
    s = (f"Example: '{text}'\n"
         f"Predicted class: '{label_names[y_pred]}' (confidence={prob if y_pred == 1 else 100 - prob:.1f}%)\n"
         f"True class: '{label_names[y_true]}'")

    fn_pos = float.__gt__ if y_pred == 1 else float.__lt__
    fn_neg = float.__lt__ if y_pred == 1 else float.__gt__

    # Features contributing to predicted class
    # - if predicted == 0 we select all features < 0, since negative values are < 0.5 after sigmoid
    # - if predicted == 1 we select all features > 0, since negative values are > 0.5 after sigmoid
    pred_pos_features = [f for f in features if fn_pos(f[1], 0)]
    # Features contributing to the non-predicted class
    pred_neg_features = [f for f in features if fn_neg(f[1], 0)]

    text_s = [s]
    if pred_pos_features:
        text_s.append(__helper_explanation(pred_pos_features, y_pred, feature_names, label_names, words_mode))
    if pred_neg_features:
        text_s.append(__helper_explanation(pred_neg_features, 1 - y_pred, feature_names, label_names, words_mode))
    return "\n".join(text_s)


def __helper_explanation(features, y, feature_names, label_names, words_mode: bool) -> str:
    feature_importance_text = f"Top {'words' if words_mode else 'features'} that contributed towards '{label_names[y]}':\n"
    explanations = list()
    for f, pf, vf in features:
        # vf is the original value of the feature f, unless f=OTHER_0/1, where it is the average % importance on all other features
        # here we dynamically compose template
        key_name = "mean" if f in ["OTHER_0", "OTHER_1"] else "score"
        suffix = "%" if key_name == "mean" else ""
        # If the feature does not have a description, we keep its name
        pretty_name = capitalize_first_letter(feature_names[f]) if feature_names is not None and (
                f in feature_names or key_name == "mean") else f
        # Fill explanation template
        value = vf if not isinstance(vf, Number) else f"{vf:.2f}"
        if words_mode and key_name != "mean":
            sc = f" - [importance={abs(pf):.1f}%] {value}"
        else:
            sc = f" - [importance={abs(pf):.1f}%] {pretty_name if not words_mode else 'Other words'} ({key_name}={value}{suffix})"
        explanations.append(sc)
    feature_importance_text += "\n".join(explanations)
    return feature_importance_text
