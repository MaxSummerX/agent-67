from abc import ABC, abstractmethod


class BaseLLMConfig:
    """Параметры подключения к провайдеру: URL, модель, ключ."""

    def __init__(self, base_url: str, model: str, api_token: str):
        self.api_key = api_token
        self.base_url = base_url
        self.model = model


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
    ) -> dict:
        """Один запрос чата; возвращает {"message": ..., "finish_reason": ...}."""
        ...
