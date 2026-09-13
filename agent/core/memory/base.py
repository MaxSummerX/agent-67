from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Долгосрочная память агента: поиск релевантных записей и добавление новых."""

    @abstractmethod
    async def search(self, query: str) -> list[dict]:
        """Возвращает записи, релевантные запросу."""
        ...

    @abstractmethod
    async def add(self, data: dict) -> None:
        """Сохраняет запись в память."""
        ...
