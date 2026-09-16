from agent.core.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """Реестр инструментов по имени: отдаёт схемы для LLM и исполняет вызовы."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self.registry: dict[str, BaseTool] = {}
        for tool in tools:
            if tool.name in self.registry:
                raise ValueError(f"Дублирующееся имя инструмента: {tool.name!r}")
            self.registry[tool.name] = tool

    def schemas(self) -> list[dict]:
        """Схемы всех инструментов для передачи в LLM."""
        return [tool.schema() for tool in self.registry.values()]

    async def execute(self, name: str, args: dict) -> ToolResult:
        """Находит инструмент и выполняет его."""
        _HINT = "\n\n[Проанализируй ошибку и попробуй другой подход.]"

        tool = self.registry.get(name)

        if tool is None:
            return ToolResult.error(f"Неизвестный инструмент {name!r}. Доступны: {', '.join(self.registry)}" + _HINT)

        errors = tool.validate_params(args)
        if errors:
            return ToolResult.error(
                f"Ошибка валидации аргументов: {'; '.join(errors)}. Повтори вызов с корректными аргументами." + _HINT
            )

        try:
            return await tool.execute(args)

        except Exception as e:
            return ToolResult.error(f"Ошибка инструмента {name!r}: {e}" + _HINT)
