from memory.base_memory import BaseMemory


class WindowMemory(BaseMemory):

    def __init__(self, window_size: int = 10):
        self._messages = []
        self._window_size = window_size

    def add_messages(self, messages: list) -> None:
        self._messages.extend(messages)

    def get_context(self) -> list[dict]:
        window = self._messages[-self._window_size:]
        # ensure we start on a user message
        for i, msg in enumerate(window):
            if msg["role"] == "user":
                return window[i:]
        return window