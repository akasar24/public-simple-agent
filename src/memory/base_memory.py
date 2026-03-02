from abc import ABC, abstractmethod

class BaseMemory(ABC):
    @abstractmethod
    def add_messages(self, messages: list) -> None:
        pass

    @abstractmethod
    def get_context(self) -> list[dict]:
        pass