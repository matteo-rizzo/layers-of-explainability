from collections import defaultdict

from transformers import pipeline

from src.text_classification.classes.features.Feature import Feature


class TextEmotion(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 256, *args, **kwargs):
        self.pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device="cuda" if use_gpu else "cpu", top_k=None, batch_size=batch_size)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        labels_and_score = self.pipe(texts, truncation=True, padding=True, max_length=512)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{emotion["label"]}'].append(emotion["score"]) for labels_list_dict in labels_and_score for emotion in labels_list_dict]
        return feature_df
