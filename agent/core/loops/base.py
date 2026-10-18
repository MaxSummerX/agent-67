from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from agent.core.context.base import BaseContextBuilder
from agent.core.conversations.base import BaseConversationStore
from agent.core.decision import BaseDecisionEngine
from agent.core.llm.base import BaseLLM
from agent.core.memory.base import BaseMemory
from agent.core.tools.registry import ToolRegistry


class TraceWriter(Protocol):
    """
    Запись событий трейса: всё, что нужно циклу - уметь дописывать события.
    """

    async def append(self, conversation_id: str, data: list[dict]) -> None: ...


@dataclass
class UsageStats:
    """Статистика расхода токенов за последний запрос LLM."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: int = 0
    cache_write: int = 0

    @property
    def cache_hit_pct(self) -> float:
        return (self.cached / self.prompt_tokens * 100) if self.prompt_tokens > 0 else 0.0


@dataclass
class AgentDependencies:
    """
    Все зависимости агента: модель, инструменты, контекст, память, история, трейс.

    trace - приёмник сырых событий цикла, опционален: None -> цикл работает без записи трейса.
    decision_engine - ось типизированных решений (System One), опциональна:
    None -> все точки потребления деградируют в прежнее поведение.
    """

    llm: BaseLLM
    tools: ToolRegistry
    context: BaseContextBuilder
    memory: BaseMemory
    conversations: BaseConversationStore
    trace: TraceWriter | None = None
    decision_engine: BaseDecisionEngine | None = None


class BaseAgentLoop(ABC):
    """Цикл выполнения: сборка контекста, вызовы LLM и инструментов до финального ответа."""

    @abstractmethod
    async def run(
        self, message: str, dependencies: AgentDependencies, conversation_id: str | None = None
    ) -> tuple[str, UsageStats]:
        """Обрабатывает сообщение до финального ответа; conversation_id включает историю."""
        ...
