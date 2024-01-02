from collections import defaultdict

from transformers import pipeline

from src.features_extraction.classes.Feature import Feature


class TextEmotion(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 256, *args, **kwargs):
        self.pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions",
                             device="cuda" if use_gpu else "cpu", top_k=None, batch_size=batch_size)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        labels_and_score = self.pipe(texts, truncation=True, padding=True, max_length=512)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{emotion["label"]}'].append(emotion["score"]) for labels_list_dict in
         labels_and_score for emotion in labels_list_dict]
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        return {
            f"{cls.__name__}_disappointment": "text shows 'disappointment'",
            f"{cls.__name__}_sadness": "text shows 'sadness'",
            f"{cls.__name__}_annoyance": "text shows 'annoyance'",
            f"{cls.__name__}_neutral": "no emotions detected",
            f"{cls.__name__}_disapproval": "text shows 'disapproval'",
            f"{cls.__name__}_realization": "text shows 'realization'",
            f"{cls.__name__}_nervousness": "text shows 'nervousness'",
            f"{cls.__name__}_approval": "text shows 'approval'",
            f"{cls.__name__}_joy": "text shows 'joy'",
            f"{cls.__name__}_anger": "text shows 'anger'",
            f"{cls.__name__}_embarrassment": "text shows 'embarrassment'",
            f"{cls.__name__}_caring": "text shows 'caring'",
            f"{cls.__name__}_remorse": "text shows 'remorse'",
            f"{cls.__name__}_disgust": "text shows 'disgust'",
            f"{cls.__name__}_grief": "text shows 'grief'",
            f"{cls.__name__}_confusion": "text shows 'confusion'",
            f"{cls.__name__}_relief": "text shows 'relief'",
            f"{cls.__name__}_desire": "text shows 'desire'",
            f"{cls.__name__}_admiration": "text shows 'admiration'",
            f"{cls.__name__}_optimism": "text shows 'optimism'",
            f"{cls.__name__}_fear": "text shows 'fear'",
            f"{cls.__name__}_love": "text shows 'love'",
            f"{cls.__name__}_excitement": "text shows 'excitement'",
            f"{cls.__name__}_curiosity": "text shows 'curiosity'",
            f"{cls.__name__}_amusement": "text shows 'amusement'",
            f"{cls.__name__}_surprise": "text shows 'surprise'",
            f"{cls.__name__}_gratitude": "text shows 'gratitude'",
            f"{cls.__name__}_pride": "text shows 'pride'"
        }
