from memory.base_memory import BaseMemory


class FullHistoryMemory(BaseMemory):

    def __init__(self):
        self._messages = []

    def add_messages(self, messages: list) -> None:
        self._messages.extend(messages)

    def get_context(self) -> list[dict]:
        return self._messages