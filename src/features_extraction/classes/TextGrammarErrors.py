from collections import Counter, defaultdict

import language_tool_python
from tqdm import tqdm

from src.features_extraction.classes.Feature import Feature


class TextGrammarErrors(Feature):
    def __init__(self, *args, **kwargs):
        self.pipe = language_tool_python.LanguageTool("en-US")

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        def default_list() -> list[int]:
            return [0] * len(texts)

        feature_df: dict[str, list[float]] = defaultdict(default_list)
        for i, t in tqdm(enumerate(texts), desc="grammar analysis", total=len(texts)):
            matches = self.pipe.check(t)
            error_count_by_cat: dict[str, int] = Counter([m.category for m in matches])
            feature_df[f"{self.__class__.__name__}_num_errors"][i] = len(matches)
            for cat, n in error_count_by_cat.items():
                feature_df[f"{self.__class__.__name__}_error_{cat}"][i] = n

        return feature_df

    @classmethod
    def label_description(cls) -> dict[str, str]:
        error_types = [
            "ACADEMIC_WRITING",
            "CAPITALIZATION",
            "COLLOCATIONS",
            "COMMONLY_CONFUSED_WORDS",
            "COMPOUNDING",
            "CONFUSED_WORDS",
            "CREATIVE_WRITING",
            "GRAMMAR",
            "MISCELLANEOUS",
            "NONSTANDARD_PHRASES",
            "PLAIN_ENGLISH",
            "POSSIBLE_TYPO",
            "PUNCTUATION",
            "REDUNDANCY",
            "REPETITIONS",
            "SEMANTICS",
            "STYLE",
            "STYLISTIC_HINTS_FOR_CREATIVE_WRITING",
            "TEXT_ANALYSIS",
            "TYPOGRAPHY",
            "UPPER/LOWERCASE",
            "WIKIPEDIA",
            "AMERICAN_ENGLISH_STYLE",
            "BRITISH_ENGLISH"
        ]

        d = defaultdict(lambda: "Other LanguageTool errors")
        d.update(
            {f"{cls.__name__}_error_{cat}": f"LanguageTool detected errors of type '{cat.lower().replace('_', ' ')}'"
             for cat in error_types})
        d[f"{cls.__name__}_error_num_errors"] = "Quantity of errors detected by LanguageTool"
        d[f"{cls.__name__}_MISC"] = "LanguageTool detected 'miscellaneous' errors"
        return d
