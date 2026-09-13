from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Инструмент агента: имя, описание, JSON-схема и исполнение."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя инструмента, на которое ссылается LLM."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Краткое описание для модели: что и когда использовать."""
        ...

    @abstractmethod
    def schema(self) -> dict:
        """JSON-схема tool-calling (OpenAI-формат)."""
        ...

    @abstractmethod
    async def execute(self, args: dict) -> str:
        """Выполняет инструмент с распакованными аргументами, возвращает текст-результат."""
        ...
