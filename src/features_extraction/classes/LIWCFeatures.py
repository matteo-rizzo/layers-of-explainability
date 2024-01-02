from collections import defaultdict
from pathlib import Path

from src.features_extraction.classes.Feature import Feature
from src.text_classification.external.nlpkit import LIWCExtractor
from src.text_classification.external.nlpkit.liwc import Liwc


class LIWCFeatures(Feature):
    LIWC_FILEPATH = Path("src") / "text_classification" / "external" / "nlpkit" / "liwc" / "resources" / "liwc.dic"

    def __init__(self, *args, **kwargs):
        liwc = Liwc(LIWCFeatures.LIWC_FILEPATH)
        self.extractor = LIWCExtractor(liwc)

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        # [4233, 4776]
        ext_data: list[dict] = self.extractor.fit_transform(texts)
        # Fill missing categories when not all aspects are returned
        all_categories = {k: .0 for k in self.extractor.liwc.categories.values()}
        for d in ext_data:
            d.update(all_categories)
        # ext_data = [d.update(all_categories) for d in ext_data]
        feature_df: dict[str, list[float]] = defaultdict(list)
        [feature_df[f'{self.__class__.__name__}_{aspect}'].append(score) for label_dict in ext_data for aspect, score in
         label_dict.items()]
        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        # See manual at: https://www.liwc.app/help/psychometrics-manuals
        liwc_spheres = {
            "funct": "all functional words",
            "pronoun": "pronouns",
            "ppron": "personal pronouns",
            "i": "1st person singular",
            "we": "1st person plural",
            "you": "2nd person",
            "shehe": "3rd person singular",
            "they": "3rd person plural",
            "ipron": "impersonal pronouns",
            "article": "articles",
            "verb": "verbs",
            "auxverb": "auxiliary verbs",
            "past": "past focus",
            "present": "present focus",
            "future": "future focus",
            "adverb": "Adverbs",
            "preps": "prepositions",
            "conj": "conjunctions",
            "negate": "Negations",
            "quant": "quantities",
            "number": "numbers",
            "swear": "swear words",
            "social": "social processes",
            "family": "family",
            "friend": "friends",
            "humans": "human beings",  #
            "affect": "affections",
            "posemo": "positive emotions",
            "negemo": "negative emotions",
            "anx": "anxiety",
            "anger": "anger",
            "sad": "sadness",
            "cogmech": "cognitive processes",
            "insight": "insight",
            "cause": "causation",
            "discrep": "discrepancy",
            "tentat": "tentative",
            "certain": "certitude",
            "inhib": "inhibition",  #
            "incl": "inclusion",  #
            "excl": "exclusion",  #
            "percept": "perception",
            "see": "visual perception",
            "hear": "auditory perception",
            "feel": "feeling",
            "bio": "physical",
            "body": "human body",  #
            "health": "health",
            "sexual": "sexuality",
            "ingest": "food",
            "relativ": "time, space & motion",
            "motion": "motion",
            "space": "space perception",
            "time": "time",
            "work": "work",  #
            "achieve": "achievement",
            "leisure": "leisure",
            "home": "home-place",
            "money": "money",
            "relig": "religion",
            "death": "death",
            "assent": "assent",
            "nonfl": "nonfluencies",
            "filler": "filler words/exclamations"
        }

        return {f"{cls.__name__}_{s}": f"LIWC detection of words about/in category '{d}'" for s, d in
                liwc_spheres.items()}
