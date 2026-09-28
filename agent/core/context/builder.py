from datetime import datetime

from agent.core.context.base import BaseContextBuilder, ContextInput, ContextOutput


class ContextBuilder(BaseContextBuilder):
    """Собирает контекст: системный промпт с текущей датой, память и история диалога."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    async def build_context(self, data: ContextInput) -> ContextOutput:
        system = self.prompt
        if data.memories:
            system += "\n" + "\n".join(memory["content"] for memory in data.memories)

        start = 0
        while start < len(data.conversation) and data.conversation[start]["role"] == "tool":
            start += 1

        history = data.conversation[start:]

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
