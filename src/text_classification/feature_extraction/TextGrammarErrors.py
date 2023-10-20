from collections import Counter, defaultdict

import language_tool_python
from tqdm import tqdm

from src.text_classification.feature_extraction.Feature import Feature


class TextGrammarErrors(Feature):
    def __init__(self, use_gpu: bool = True, batch_size: int = 512):
        self.pipe = language_tool_python.LanguageTool("en-US")

    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        def default_list() -> list[int]:
            return [0] * len(texts)

        feature_df: dict[str, list[float]] = defaultdict(default_list)
        for i, t in tqdm(enumerate(texts), desc="grammar analysis", total=len(texts)):
            matches = self.pipe.check(t)
            error_count_by_cat: dict[str, int] = Counter([m.category for m in matches])
            feature_df["num_errors"][i] = len(matches)
            for cat, n in error_count_by_cat.items():
                feature_df[f"error_{cat}"][i] = n

        return feature_df
