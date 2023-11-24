from collections import defaultdict

from transformers import pipeline

from src.text_classification.classes.features.Feature import Feature


class TopicLM(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 256, *args, **kwargs):
        self.pipe = pipeline("text-classification", model="cardiffnlp/tweet-topic-21-multi", device="cuda" if use_gpu else "cpu",
                             top_k=None, batch_size=batch_size)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        labels_and_score = self.pipe(texts, truncation=True, padding=True, max_length=512)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{emotion["label"]}'].append(emotion["score"]) for labels_list_dict in labels_and_score for emotion in labels_list_dict]
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        return {
            f"{cls.__name__}_sports": "text discuss topics about sport",
            f"{cls.__name__}_news_&_social_concern": "text discuss topics about news and social concerns",
            f"{cls.__name__}_fitness_&_health": "text discuss topics about fitness and health",
            f"{cls.__name__}_youth_&_student_life": "text discuss topics about student life and youths",
            f"{cls.__name__}_learning_&_educational": "text discuss topics about education",
            f"{cls.__name__}_science_&_technology": "text discuss topics about science and technology",
            f"{cls.__name__}_celebrity_&_pop_culture": "text discuss topics about celebrities and pop culture",
            f"{cls.__name__}_travel_&_adventure": "text discuss topics about travel and adventures",
            f"{cls.__name__}_diaries_&_daily_life": "text discuss topics about daily life",
            f"{cls.__name__}_food_&_dining": "text discuss topics about food and dining",
            f"{cls.__name__}_gaming": "text discuss topics about gaming",
            f"{cls.__name__}_business_&_entrepreneurs": "text discuss topics about business and entrepreneurs",
            f"{cls.__name__}_family": "text discuss topics about family",
            f"{cls.__name__}_relationships": "text discuss topics about relationships",
            f"{cls.__name__}_fashion_&_style": "text discuss topics about fashion/style",
            f"{cls.__name__}_music": "text discuss topics about music",
            f"{cls.__name__}_film_tv_&_video": "text discuss topics about film, TV and videos",
            f"{cls.__name__}_other_hobbies": "text discuss topics about hobbies",
            f"{cls.__name__}_arts_&_culture": "text discuss topics about arts and culture"
        }
