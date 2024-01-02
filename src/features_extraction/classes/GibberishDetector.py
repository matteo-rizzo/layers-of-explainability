from collections import defaultdict

from transformers import pipeline

from src.features_extraction.classes.Feature import Feature


class GibberishDetector(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 256, *args, **kwargs):
        self.pipe = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457",
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
            f"{cls.__name__}_clean": "words form a complete and meaningful sentence",
            f"{cls.__name__}_mild gibberish": "text has grammatical errors, word sense errors, or syntactical abnormalities that reduce the overall coherence",
            f"{cls.__name__}_word salad": "words make sense independently, but do not produce a coherent meaning",
            f"{cls.__name__}_noise": "text appears to be made up of random characters"
        }
