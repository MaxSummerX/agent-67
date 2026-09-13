from collections.abc import Callable

import pytest

from agent.core import Agent, BaseLLM, BaseLLMConfig, BaseTool
from agent.core.context.builder import ContextBuilder
from agent.core.loops.base import AgentDependencies
from agent.core.loops.loop import AgentLoop
from agent.core.memory.null_memory import NullMemory
from agent.core.tools.registry import ToolRegistry
from agent.infrastructure.conversations.in_memory import InMemoryConversation
from agent.prompts import SYSTEM_PROMPT


class FakeLLM(BaseLLM):
    """
    Отвечает по сценарию: список заранее заготовленных ответов, выдают по одному.

    Запоминает последнюю историю сообщений — для проверок содержимого запросов к модели.
    """

    def __init__(self, script: list[dict]) -> None:
        super().__init__(BaseLLMConfig("fake", "fake", "fake"))
        self.script = list(script)
        self.last_messages: list[dict] = []

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None, tool_choice: str = "auto") -> dict:
        self.last_messages = messages
        return self.script.pop(0)


class UpperTool(BaseTool):
    """Тестовый инструмент: возвращает текст заглавными буквами."""

    @property
    def name(self) -> str:
        return "upper"

    @property
    def description(self) -> str:
        return "Возвращает текст заглавными буквами"

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    async def execute(self, args: dict) -> str:
        return str(args["text"]).upper()


@pytest.fixture
def fake_llm_factory() -> Callable[[list[dict]], FakeLLM]:
    """Фабрика LLM с заданным сценарием ответов."""
    return FakeLLM


@pytest.fixture
def build_agent() -> Callable[..., Agent]:
    """Собирает агента с тестовым инструментом `upper`, данным LLM и опциональным циклом."""

    def _build(llm: BaseLLM, loop: AgentLoop | None = None) -> Agent:
        return Agent(
            loop or AgentLoop(),
            AgentDependencies(
                llm=llm,
                tools=ToolRegistry([UpperTool()]),
                context=ContextBuilder(SYSTEM_PROMPT),
                memory=NullMemory(),
                conversations=InMemoryConversation(),
            ),
        )

    return _build
