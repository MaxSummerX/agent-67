import json

from agent.core.context.base import ContextInput
from agent.core.loops.base import AgentDependencies, BaseAgentLoop


class AgentLoop(BaseAgentLoop):
    """Цикл «модель -> tool calls -> модель», пока не придёт текстовый ответ.

    Останавливается на finish_reason == "stop" с непустым тексте,
    на content_filter или после max_round итераций. История сохраняется
    в finally — даже при падении.
    """

    def __init__(self, max_round: int = 25) -> None:
        self.max_round = max_round

    async def run(self, message: str, dependencies: AgentDependencies, conversation_id: str | None = None) -> str:
        """Обрабатывает сообщение, исполняет цепочки tool calls, возвращает ответ модели."""
        messages = []
        if conversation_id:
            messages = await dependencies.conversations.load(conversation_id)

        memory = await dependencies.memory.search(message)

        context = await dependencies.context.build_context(
            ContextInput(message=message, memories=memory, conversation=messages)
        )

        messages = context.messages

        try:
            for _ in range(self.max_round):
                response = await dependencies.llm.chat(messages=messages, tools=dependencies.tools.schemas())

                msg = response["message"]
                finish = response["finish_reason"]

                calls = msg.get("tool_calls") or []
                text = (msg.get("content") or "").strip()
                if calls:
                    clean: list[dict] = []
                    messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": clean})
                    for call in calls:
                        func = call["function"]
                        clean.append(
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {"name": func["name"], "arguments": func["arguments"]},
                            }
                        )
                        try:
                            args = json.loads(func.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "content": f"Ошибка: невалидный JSON аргументов: {func.get('arguments')}",
                                }
                            )
                            continue
                        result = await dependencies.tools.execute(func["name"], args)
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                    continue

                messages.append({"role": "assistant", "content": text or None})
                if finish == "stop" and text:
                    return text

                if finish == "content_filter":
                    return text or "[Ответ заблокирован фильтром провайдера]"

                messages.append(
                    {"role": "user", "content": "Ответ пустой или обрезан. Заверши задачу и дай полный текст."}
                )

            return f"Не удалось выполнить задачу за {self.max_round} кругов."

        finally:
            if conversation_id:
                await dependencies.conversations.save(
                    conversation_id,
                    messages,
                )
