import abc


class BaseTopicExtractor(abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs) -> None:
        pass

    def train(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def predict_one(self, document: str, k: int, *args, **kwargs) -> list:
        pass

    @abc.abstractmethod
    def predict(self, documents: list[str], k: int, *args, **kwargs) -> list[list]:
        pass

    def plot_wonders(self, documents: list) -> None:
        pass
