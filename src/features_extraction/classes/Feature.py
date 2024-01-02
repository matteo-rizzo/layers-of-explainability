from __future__ import annotations

import abc


class Feature(abc.ABC):

    @abc.abstractmethod
    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        pass

    @classmethod
    def label_description(cls) -> dict[str, str] | None:
        """
        Return a user-friendly short description of each possible feature returned by this feature extractor
        """
        return None
