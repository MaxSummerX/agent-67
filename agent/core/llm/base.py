from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class BaseLLMConfig:
    """Параметры подключения к провайдеру: URL, модель, ключ."""

    def __init__(self, base_url: str, model: str, api_token: str):
        self.api_key = api_token
        self.base_url = base_url
        self.model = model


@dataclass
class ToolCall:
    """Вызов инструмента от модели: id для ответа, имя и уже распарсенные аргументы."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Ответ модели за один запрос: текст и/или вызовы инструментов."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None


class BaseLLM(ABC):
    """Интерфейс общения с моделью: один запрос чата с опциональными инструментами."""

    def __init__(self, config: BaseLLMConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Один запрос чата; возвращает ChatResponse с текстом и/или вызовами инструментов."""
        ...
