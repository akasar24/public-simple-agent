from langchain.chat_models import init_chat_model
from memory.base_memory import BaseMemory


class SummaryMemory(BaseMemory):

    def __init__(
        self,
        summarizer_model_str: str = "anthropic:claude-haiku-4-5-20251001",
        threshold: int = 10,
        keep_recent: int = 4,
    ):
        self._messages = []
        self._summary = None
        self._threshold = threshold
        self._keep_recent = keep_recent
        self._summarizer = init_chat_model(summarizer_model_str)

    def _summarize(self) -> None:
        # everything except the most recent messages gets compressed
        to_summarize = self._messages[:-self._keep_recent]
        recent = self._messages[-self._keep_recent:]

        # if we already have a summary, fold it in so we don't lose older context
        if self._summary:
            prompt = f"Previous summary (keep this concise): {self._summary}\n\nNew messages to incorporate, update the summary in 2-3 sentences maximum:\n"
        else:
            prompt = "Summarize this conversation in 2-3 sentences maximum, preserving key facts, names, and preferences:\n\n"

        prompt += "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)

        result = self._summarizer.invoke(prompt)
        self._summary = result.content

        # keep only recent messages. remove what we have summarized
        self._messages = recent

    def add_messages(self, messages: list) -> None:
        self._messages.extend(messages)
        if len(self._messages) >= self._threshold:
            self._summarize()

    def get_context(self) -> list[dict]:
        if self._summary:
            summary_msg = {"role": "user", "content": f"[Conversation summary: {self._summary}]"}
            return [summary_msg] + self._messages
        return self._messages