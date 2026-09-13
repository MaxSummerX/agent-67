from agent.core import BaseConversationStore


class InMemoryConversation(BaseConversationStore):
    """История диалогов в памяти процесса: исчезает при перезапуске."""

    def __init__(self) -> None:
        self.db: dict[str, list[dict]] = {}

    async def load(self, conversation_id: str) -> list[dict]:
        return self.db.get(conversation_id, [])

    async def save(
        self,
        conversation_id: str,
        data: list[dict],
    ) -> None:
        self.db[conversation_id] = data
