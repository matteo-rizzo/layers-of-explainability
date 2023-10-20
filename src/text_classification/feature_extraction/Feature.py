import abc
from typing import Any


class Feature(abc.ABC):

    @abc.abstractmethod
    def extract(self, texts: list[str]) -> Any:
        pass
