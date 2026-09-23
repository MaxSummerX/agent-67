from pathlib import Path

from httpx2 import AsyncClient

from agent.core import Agent, AgentDependencies, AgentLoop, BaseAgentLoop, BaseLLM, ToolRegistry
from agent.core.context.builder import ContextBuilder
from agent.core.memory.null_memory import NullMemory
from agent.infrastructure.conversations.in_memory import InMemoryConversation
from agent.infrastructure.conversations.json_store import JSONConversationStore
from agent.infrastructure.llm.giga_chat import GigaChatClient, GigaChatConfig
from agent.infrastructure.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleLLM
from agent.infrastructure.tools.fetch_url import FetchURLTool
from agent.infrastructure.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.infrastructure.tools.shell import ExecTool
from agent.infrastructure.tools.web_search import SearchWebTool
from agent.prompts import SYSTEM_PROMPT
from agent.settings import (
    API_KEY,
    BASE_URL,
    GIGACHAT_CREDENTIALS,
    GIGACHAT_MODEL,
    GIGACHAT_SCOPE,
    MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


_CACHEABLE = ("anthropic", "gemini", "qwen")


def create_llm(provider: str, http_client: AsyncClient) -> BaseLLM:
    """Собирает LLM-клиент по имени провайдера; неизвестный провайдер - KeyError, open_router без ключа - сразу RuntimeError."""
    if provider == "open_router" and not API_KEY:
        raise RuntimeError("OpenRouter: задайте API_KEY в .env")

    provider_map = {
        "open_router": OpenAICompatibleLLM(
            config=OpenAICompatibleConfig(
                model=MODEL,
                api_key=API_KEY,
                base_url=BASE_URL,
                temperature=0.7,
                max_tokens=8192,
                headers={
                    # Content-Type и Authorization ставит клиент автоматически
                    "HTTP-Referer": "https://github.com/MaxSummerX/agent-67",
                    "X-Title": "agent-67",
                },
                cache_system_prompt=any(
                    s in MODEL.lower() for s in _CACHEABLE
                ),  # префикс-кэш для anthropic/gemini/qwen
            ),
            http_client=http_client,
        ),
        "ollama": OpenAICompatibleLLM(
            config=OpenAICompatibleConfig(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                api_key="",  # локальный сервер ключа не требует
                temperature=0.7,
                max_tokens=4096,
            ),
            http_client=http_client,
        ),
        "giga_chat": GigaChatClient(
            config=GigaChatConfig(
                base_url="https://api.giga.chat/v1/chat/completions",
                model=GIGACHAT_MODEL,
                api_key="",  # авторизация через credentials (OAuth), не api_key
                temperature=0.7,
                max_tokens=4096,
                credentials=GIGACHAT_CREDENTIALS,
                scope=GIGACHAT_SCOPE,
            ),
            http_client=http_client,
        ),
    }
    return provider_map[provider]


def create_agent(
    http_client: AsyncClient,
    persist: bool = False,
    workspace: str | Path | None = None,
    loop: BaseAgentLoop | None = None,
    provider: str | None = None,
) -> Agent:
    """
    Точка сборки агента: связывает LLM, инструменты, память и историю.

    persist=True - сохранять диалоги в JSON (./history), иначе - только в памяти.
    provider - имя LLM-провайдера из реестра create_llm; по умолчанию open_router.
    """
    workspace = Path(workspace or "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    fs_args = {"workspace": workspace, "allowed_dir": workspace}

    llm = create_llm(provider or "open_router", http_client)

    tools = ToolRegistry(
        [
            FetchURLTool(http_client),
            SearchWebTool(),
            ExecTool(working_dir=str(workspace)),
            EditFileTool(**fs_args),
            ReadFileTool(**fs_args),
            WriteFileTool(**fs_args),
            ListDirTool(**fs_args),
        ]
    )
    memory = NullMemory()
    conversations = JSONConversationStore() if persist else InMemoryConversation()
    context = ContextBuilder(prompt=SYSTEM_PROMPT, max_history=200)

    dependencies = AgentDependencies(
        llm=llm,
        tools=tools,
        context=context,
        memory=memory,
        conversations=conversations,
    )

    loop = loop or AgentLoop(max_rounds=25)

    return Agent(
        loop=loop,
        dependencies=dependencies,
    )
