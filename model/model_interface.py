from abc import ABC, abstractmethod

class IModel(ABC):

    @abstractmethod
    def predict(self, board: list[int]):
        pass

    def update(self, var) -> None:
        pass