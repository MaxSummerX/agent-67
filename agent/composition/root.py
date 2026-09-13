from httpx2 import AsyncClient

from agent.core import Agent, AgentDependencies, AgentLoop, ToolRegistry
from agent.core.context.builder import ContextBuilder
from agent.core.memory.null_memory import NullMemory
from agent.infrastructure.conversations.in_memory import InMemoryConversation
from agent.infrastructure.conversations.json_store import JSONConversationStore
from agent.infrastructure.llm.open_router import OpenRouterConfig, OpenRouterLLM
from agent.infrastructure.tools.fetch_url import FetchURLTool
from agent.infrastructure.tools.web_search import SearchWebTool
from agent.prompts import SYSTEM_PROMPT
from agent.settings import API_KEY, BASE_URL, MODEL


def create_agent(http_client: AsyncClient, persist: bool = False) -> Agent:
    """
    Точка сборки агента: связывает LLM, инструменты, память и историю.

    persist=True - сохранять диалоги в JSON (./history), иначе - только в памяти.
    """

    llm_config = OpenRouterConfig(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    llm = OpenRouterLLM(config=llm_config, http_client=http_client)
    tools = ToolRegistry([SearchWebTool(), FetchURLTool()])
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

    loop = AgentLoop(max_round=25)

    return Agent(
        loop=loop,
        dependencies=dependencies,
    )
