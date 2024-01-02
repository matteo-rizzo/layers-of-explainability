from collections import defaultdict

from empath import Empath

from src.features_extraction.classes.Feature import Feature


class EmpathFeatures(Feature):
    def __init__(self, *args, **kwargs):
        self.lexicon = Empath()

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        processed_texts: list[dict[str, float]] = list()
        default_score = self.lexicon.analyze("Hi", normalize=True)
        default_score = {k: .0 for k in default_score}
        for t in texts:
            data = self.lexicon.analyze(t, normalize=True)
            processed_texts.append(data if data is not None else default_score)
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in processed_texts for
         aspect, score in label_dict.items()]
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        cats = ['help', 'office', 'violence', 'dance', 'money', 'wedding', 'valuable', 'domestic_work', 'sleep',
                'medical_emergency', 'cold', 'hate', 'cheerfulness', 'aggression',
                'occupation', 'envy', 'anticipation', 'family', 'crime', 'attractive', 'masculine', 'prison', 'health',
                'pride', 'dispute', 'nervousness', 'government', 'weakness',
                'horror', 'swearing_terms', 'leisure', 'suffering', 'royalty', 'wealthy', 'white_collar_job', 'tourism',
                'furniture', 'school', 'magic', 'beach', 'journalism',
                'morning', 'banking', 'social_media', 'exercise', 'night', 'kill', 'art', 'play', 'computer', 'college',
                'traveling', 'stealing', 'real_estate', 'home', 'divine',
                'sexual', 'fear', 'monster', 'irritability', 'superhero', 'business', 'driving', 'pet', 'childish',
                'cooking', 'exasperation', 'religion', 'hipster', 'internet',
                'surprise', 'reading', 'worship', 'leader', 'independence', 'movement', 'body', 'noise', 'eating',
                'medieval', 'zest', 'confusion', 'water', 'sports', 'death',
                'healing', 'legend', 'heroic', 'celebration', 'restaurant', 'ridicule', 'programming',
                'dominant_heirarchical', 'military', 'neglect', 'swimming', 'exotic', 'love',
                'hiking', 'communication', 'hearing', 'order', 'sympathy', 'hygiene', 'weather', 'anonymity', 'trust',
                'ancient', 'deception', 'fabric', 'air_travel', 'fight',
                'dominant_personality', 'music', 'vehicle', 'politeness', 'toy', 'farming', 'meeting', 'war',
                'speaking', 'listen', 'urban', 'shopping', 'disgust', 'fire', 'tool',
                'phone', 'gain', 'sound', 'injury', 'sailing', 'rage', 'science', 'work', 'appearance', 'optimism',
                'warmth', 'youth', 'sadness', 'fun', 'emotional', 'joy',
                'affection', 'fashion', 'lust', 'shame', 'torment', 'economics', 'anger', 'politics', 'ship',
                'clothing', 'car', 'strength', 'technology', 'breaking',
                'shape_and_size', 'power', 'vacation', 'animal', 'ugliness', 'party', 'terrorism', 'smell',
                'blue_collar_job', 'poor', 'plant', 'pain', 'beauty', 'timidity',
                'philosophy', 'negotiate', 'negative_emotion', 'cleaning', 'messaging', 'competing', 'law', 'friends',
                'payment', 'achievement', 'alcohol', 'disappointment',
                'liquid', 'feminine', 'weapon', 'children', 'ocean', 'giving', 'contentment', 'writing', 'rural',
                'positive_emotion', 'musical']

        emotions = [f"Lexical category '{e}' score" for e in cats]
        labels = [f"{cls.__name__}_{aspect}" for aspect in cats]
        d = dict(zip(labels, emotions))

        return d
