from agent.core import BaseConversationStore


class InMemoryConversation(BaseConversationStore):
    """История диалогов в памяти процесса: исчезает при перезапуске."""

    def __init__(self, max_history: int = 100) -> None:
        self.db: dict[str, list[dict]] = {}
        self.max_history = max_history

    async def load(self, conversation_id: str) -> list[dict]:
        return self.db.get(conversation_id, [])[-self.max_history :]

    async def save(
        self,
        conversation_id: str,
        data: list[dict],
    ) -> None:
        self.db[conversation_id] = data

    async def append(self, conversation_id: str, data: list[dict]) -> None:
        self.db.setdefault(conversation_id, []).extend(data)
