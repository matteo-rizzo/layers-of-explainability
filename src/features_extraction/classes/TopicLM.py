from collections import defaultdict

from transformers import pipeline

from src.features_extraction.classes.Feature import Feature


class TopicLM(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 256, *args, **kwargs):
        self.pipe = pipeline("text-classification", model="cardiffnlp/tweet-topic-21-multi",
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
            f"{cls.__name__}_sports": "text talks about 'sport'",
            f"{cls.__name__}_news_&_social_concern": "text talks about 'news & social concerns'",
            f"{cls.__name__}_fitness_&_health": "text talks about 'fitness & health'",
            f"{cls.__name__}_youth_&_student_life": "text talks about 'student life & youths'",
            f"{cls.__name__}_learning_&_educational": "text talks about 'education'",
            f"{cls.__name__}_science_&_technology": "text talks about 'science & technology'",
            f"{cls.__name__}_celebrity_&_pop_culture": "text talks about 'celebrities & pop culture'",
            f"{cls.__name__}_travel_&_adventure": "text talks about 'travel & adventures'",
            f"{cls.__name__}_diaries_&_daily_life": "text talks about 'daily life'",
            f"{cls.__name__}_food_&_dining": "text talks about 'food & dining'",
            f"{cls.__name__}_gaming": "text talks about 'gaming'",
            f"{cls.__name__}_business_&_entrepreneurs": "text talks about 'business & entrepreneurs'",
            f"{cls.__name__}_family": "text talks about 'family'",
            f"{cls.__name__}_relationships": "text talks about 'relationships'",
            f"{cls.__name__}_fashion_&_style": "text talks about 'fashion & style'",
            f"{cls.__name__}_music": "text talks about 'music'",
            f"{cls.__name__}_film_tv_&_video": "text talks about 'TV & videos'",
            f"{cls.__name__}_other_hobbies": "text talks about 'hobbies'",
            f"{cls.__name__}_arts_&_culture": "text talks about 'arts & culture'"
        }
