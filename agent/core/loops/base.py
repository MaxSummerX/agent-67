from abc import ABC, abstractmethod
from dataclasses import dataclass

from agent.core.context.base import BaseContextBuilder
from agent.core.conversations.base import BaseConversationStore
from agent.core.llm.base import BaseLLM
from agent.core.memory.base import BaseMemory
from agent.core.tools.registry import ToolRegistry


@dataclass
class AgentDependencies:
    """Все зависимости агента: модель, инструменты, контекст, память, история."""

    llm: BaseLLM
    tools: ToolRegistry
    context: BaseContextBuilder
    memory: BaseMemory
    conversations: BaseConversationStore


class BaseAgentLoop(ABC):
    """Цикл выполнения: сборка контекста, вызовы LLM и инструментов до финального ответа."""

    @abstractmethod
    async def run(self, message: str, dependencies: AgentDependencies, conversation_id: str | None = None) -> str:
        """Обрабатывает сообщение до финального ответа; conversation_id включает историю."""
        ...
