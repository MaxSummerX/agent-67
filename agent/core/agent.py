from agent.core.loops.base import AgentDependencies, BaseAgentLoop


class Agent:
    """Готовый к использованию агент: связывает цикл выполнения с зависимостями."""

    def __init__(
        self,
        loop: BaseAgentLoop,
        dependencies: AgentDependencies,
    ):
        self.loop = loop
        self.dependencies = dependencies

    async def run(self, message: str, conversation_id: str) -> str:
        """Прогоняет сообщение через цикл агента и возвращает итоговый текст."""
        return await self.loop.run(message, self.dependencies, conversation_id)
