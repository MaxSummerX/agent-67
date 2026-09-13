from abc import ABC, abstractmethod


class BaseConversationStore(ABC):
    """Хранилище истории диалога по идентификатору беседы."""

    @abstractmethod
    async def load(self, conversation_id: str) -> list[dict]:
        """Возвращает список сообщений беседы (пустой, если беседы нет)."""
        ...

    @abstractmethod
    async def save(self, conversation_id: str, data: list[dict]) -> None:
        """Полностью перезаписывает историю беседы."""
        ...
