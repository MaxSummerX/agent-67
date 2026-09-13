from agent.core.memory.base import BaseMemory


class NullMemory(BaseMemory):
    """Память выключена - всегда пустой результат."""

    async def search(self, query: str) -> list[dict]:
        return []

    async def add(self, data: dict) -> None:
        return
