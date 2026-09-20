from pathlib import Path

from httpx2 import AsyncClient

from agent.core import Agent, AgentDependencies, AgentLoop, BaseAgentLoop, ToolRegistry
from agent.core.context.builder import ContextBuilder
from agent.core.memory.null_memory import NullMemory
from agent.infrastructure.conversations.in_memory import InMemoryConversation
from agent.infrastructure.conversations.json_store import JSONConversationStore
from agent.infrastructure.llm.open_router import OpenRouterConfig, OpenRouterLLM
from agent.infrastructure.tools.fetch_url import FetchURLTool
from agent.infrastructure.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.infrastructure.tools.shell import ExecTool
from agent.infrastructure.tools.web_search import SearchWebTool
from agent.prompts import SYSTEM_PROMPT
from agent.settings import API_KEY, BASE_URL, MODEL


def create_agent(
    http_client: AsyncClient,
    persist: bool = False,
    workspace: str | Path | None = None,
    loop: BaseAgentLoop | None = None,
) -> Agent:
    """
    Точка сборки агента: связывает LLM, инструменты, память и историю.

    persist=True - сохранять диалоги в JSON (./history), иначе - только в памяти.
    """
    workspace = Path(workspace or "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    fs_args = {"workspace": workspace, "allowed_dir": workspace}
    llm_config = OpenRouterConfig(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.7, max_tokens=8192)

    llm = OpenRouterLLM(config=llm_config, http_client=http_client)
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
