import json
from time import time
from typing import Any

from agent.core.context.base import ContextInput
from agent.core.loops.base import AgentDependencies, BaseAgentLoop, UsageStats


class AgentLoop(BaseAgentLoop):
    """
    Цикл «модель -> tool calls -> модель», пока не придёт текстовый ответ.

    Останавливается на finish_reason == "stop" с непустым текстом,
    на content_filter или после max_rounds итераций.
    Каждое сообщение записывается в историю в момент появления (append-only).
    Параллельно пишет трейс: сырые события цикла с отметкой времени.
    """

    def __init__(self, max_rounds: int = 25) -> None:
        if max_rounds < 1:
            raise ValueError(f"max_rounds должен быть >= 1, передано {max_rounds}")
        self.max_rounds = max_rounds

    async def run(
        self, message: str, dependencies: AgentDependencies, conversation_id: str | None = None
    ) -> tuple[str, UsageStats]:
        """Обрабатывает сообщение, исполняет цепочки tool calls, возвращает ответ модели."""
        trace_id = conversation_id or "no_session"
        await self._trace(
            dependencies,
            trace_id,
            event_type="user",
            content=message,
        )
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
        await self._trace(
            dependencies,
            trace_id,
            event_type="context_built",
            memories_found=len(memory),
        )
        usage = UsageStats()
        for iteration in range(self.max_rounds):
            try:
                response = await dependencies.llm.chat(messages=messages, tools=dependencies.tools.schemas())
            except Exception as exc:
                await self._trace(dependencies, trace_id, event_type="error", stage="llm_chat", error=repr(exc))
                raise
            usage = self.parse_usage(response.usage)

            finish = response.finish_reason
            text = response.content.strip()

            await self._trace(
                dependencies,
                trace_id,
                event_type="llm_response",
                iteration=iteration,
                finish_reason=finish,
                text=response.content,
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in response.tool_calls or []],
                usage=usage.__dict__,
            )
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
                    try:
                        result = await dependencies.tools.execute(call.name, call.arguments)
                    except Exception as exc:
                        await self._trace(
                            dependencies,
                            trace_id,
                            event_type="error",
                            stage="tool_execute",
                            tool=call.name,
                            error=repr(exc),
                        )
                        raise
                    await self._trace(
                        dependencies,
                        trace_id,
                        event_type="tool_result",
                        iteration=iteration,
                        tool=call.name,
                        arguments=call.arguments,
                        output=str(result),
                        is_error=result.is_error,
                    )
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    await add_message()
                continue

            messages.append({"role": "assistant", "content": text or ""})

            if finish == "stop" and text:
                await add_message()
                await self._trace(
                    dependencies,
                    trace_id,
                    event_type="final",
                    iteration=iteration,
                )
                return text, usage

            if finish == "content_filter":
                await add_message()
                await self._trace(
                    dependencies, trace_id, event_type="blocked", iteration=iteration, reason="content_filter"
                )
                return text or "[Ответ заблокирован фильтром провайдера]", usage

            messages.append({"role": "user", "content": "Ответ пустой или обрезан. Заверши задачу и дай полный текст."})
            await add_message()
            await self._trace(dependencies, trace_id, event_type="empty_response_retry", iteration=iteration)

        await self._trace(dependencies, trace_id, event_type="max_rounds_exceeded")
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

    async def _trace(self, deps: AgentDependencies, trace_id: str, **fields: Any) -> None:
        """Пишет событие с отметкой времени; None - запись отключена"""
        if deps.trace is not None:
            await deps.trace.append(trace_id, [{"ts": time(), **fields}])
