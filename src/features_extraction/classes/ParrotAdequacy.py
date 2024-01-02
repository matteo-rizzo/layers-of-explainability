from collections import defaultdict

from transformers import pipeline

from src.features_extraction.classes.Feature import Feature


class ParrotAdequacy(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 256, *args, **kwargs):
        self.pipe = pipeline("text-classification", model="prithivida/parrot_adequacy_model",
                             device="cuda" if use_gpu else "cpu",
                             top_k=None, batch_size=batch_size)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        labels_and_score = self.pipe(texts, truncation=True, padding=True, max_length=512)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{emotion["label"]}'].append(emotion["score"]) for labels_list_dict in
         labels_and_score for emotion in labels_list_dict]
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        return {
            f"{cls.__name__}_neutral": "text does not contain relations",
            f"{cls.__name__}_contradiction": "text contains a contradiction",
            f"{cls.__name__}_entailment": "text shows entailment of two parts of text"
        }
