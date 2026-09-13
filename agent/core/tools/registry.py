from agent.core.tools.base import BaseTool


class ToolRegistry:
    """Реестр инструментов по имени: отдаёт схемы для LLM и исполняет вызовы."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self.registry: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        """Схемы всех инструментов для передачи в LLM."""
        return [tool.schema() for tool in self.registry.values()]

    async def execute(self, name: str, args: dict) -> str:
        """Находит инструмент и выполняет его."""
        tool = self.registry.get(name)

        if tool is None:
            return f"Ошибка: неизвестный инструмент {name!r}"

        try:
            return await tool.execute(args)

        except Exception as e:
            return f"Ошибка инструмента {name!r}: {e}"
