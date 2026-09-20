from datetime import datetime

from agent.core.context.base import BaseContextBuilder, ContextInput, ContextOutput


class ContextBuilder(BaseContextBuilder):
    """Собирает контекст: системный промпт с текущей датой, память и история диалога."""

    def __init__(self, prompt: str, max_history: int = 100) -> None:
        self.prompt = prompt
        self.max_history = max_history

    async def build_context(self, data: ContextInput) -> ContextOutput:
        system = self.prompt
        if data.memories:
            system += "\n" + "\n".join(memory["content"] for memory in data.memories)

        history = [message for message in data.conversation if message["role"] != "system"][-self.max_history :]

        while history and history[0]["role"] == "tool":
            history.pop(0)

        return ContextOutput(
            [
                {"role": "system", "content": system},
                *history,
                {
                    "role": "user",
                    "content": f"[ Текущие дата и время сообщения: {datetime.now():%Y-%m-%d %H:%M}]\n{data.message}",
                },
            ]
        )
