from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ContextInput:
    """Входные данные для сборки контекста: новое сообщение, память и история."""

    message: str
    memories: list[dict]
    conversation: list[dict]


@dataclass
class ContextOutput:
    """Готовый список сообщений для отправки в LLM."""

    messages: list[dict]


class BaseContextBuilder(ABC):
    """Собирает итоговый промпт: системный промпт + память + история + сообщение."""

    @abstractmethod
    async def build_context(self, data: ContextInput) -> ContextOutput:
        """Собирает итоговый список сообщений: система + память + история + новое сообщение."""
        ...
