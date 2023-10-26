import abc


class Feature(abc.ABC):

    @abc.abstractmethod
    def extract(self, texts: list[str]) -> dict[str, list[float]]:
        pass
