import json

from agent.core.context.base import ContextInput
from agent.core.loops.base import AgentDependencies, BaseAgentLoop, UsageStats


class AgentLoop(BaseAgentLoop):
    """
    Цикл «модель -> tool calls -> модель», пока не придёт текстовый ответ.

    Останавливается на finish_reason == "stop" с непустым текстом,
    на content_filter или после max_rounds итераций.
    Каждое сообщение записывается в историю в момент появления (append-only).
    """

    def __init__(self, max_rounds: int = 25) -> None:
        if max_rounds < 1:
            raise ValueError(f"max_rounds должен быть >= 1, передано {max_rounds}")
        self.max_rounds = max_rounds

    async def run(
        self, message: str, dependencies: AgentDependencies, conversation_id: str | None = None
    ) -> tuple[str, UsageStats]:
        """Обрабатывает сообщение, исполняет цепочки tool calls, возвращает ответ модели."""
        messages = []
        if conversation_id:
            messages = await dependencies.conversations.load(conversation_id)

        memory = await dependencies.memory.search(message)
        profile = await dependencies.memory.load_profile()

        context = await dependencies.context.build_context(
            ContextInput(message=message, memories=memory, conversation=messages, profile=profile)
        )

        messages = context.messages

        async def add_message() -> None:
            """Записывает последнее сообщение в историю."""
            if conversation_id:
                await dependencies.conversations.append(conversation_id, [messages[-1]])

        await add_message()

        usage = UsageStats()
        for _ in range(self.max_rounds):
            response = await dependencies.llm.chat(messages=messages, tools=dependencies.tools.schemas())
            usage = self.parse_usage(response.usage)

            finish = response.finish_reason
            text = response.content.strip()

            if response.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                            for call in response.tool_calls
                        ],
                    }
                )
                await add_message()

                for call in response.tool_calls:
                    result = await dependencies.tools.execute(call.name, call.arguments)
                    if result.is_error:
                        pass  # наблюдение (observability): счётчик/логгинг появятся вместе с телеметрией
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    await add_message()
                continue

            messages.append({"role": "assistant", "content": text or ""})

            if finish == "stop" and text:
                await add_message()
                return text, usage

            if finish == "content_filter":
                await add_message()
                return text or "[Ответ заблокирован фильтром провайдера]", usage

            messages.append({"role": "user", "content": "Ответ пустой или обрезан. Заверши задачу и дай полный текст."})
            await add_message()

        return f"Не удалось выполнить задачу за {self.max_rounds} кругов.", usage

    @staticmethod
    def parse_usage(usage: dict | None) -> UsageStats:
        """Собирает UsageStats из сырого usage-словаря ChatResponse."""
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        details = usage.get("prompt_tokens_details", {})
        cached = details.get("cached_tokens", 0)
        cache_write = details.get("cache_write_tokens", 0)

        return UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached=cached,
            cache_write=cache_write,
        )
