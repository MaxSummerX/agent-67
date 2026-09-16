import asyncio
import json

from httpx2 import AsyncClient, HTTPStatusError

from agent.core import BaseLLM, BaseLLMConfig
from agent.core.llm.base import ChatResponse, ToolCall


class OpenRouterConfig(BaseLLMConfig):
    """Конфиг подключения к OpenRouter API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
    ) -> None:
        super().__init__(base_url, model, api_key)


class OpenRouterLLM(BaseLLM):
    """Реализация LLM через OpenRouter Chat Completions API."""

    RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
    RETRIES = 2
    BASE_DELAY = 3.0

    def __init__(
        self,
        config: OpenRouterConfig,
        http_client: AsyncClient,
    ) -> None:
        super().__init__(config)
        self.http_client = http_client

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Отправляет запрос, возвращает ChatResponse с текстом и/или вызовами инструментов."""

        payload = {"model": self.config.model, "messages": messages}
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}

        if tools:
            payload["tools"] = tools

        for attempt in range(self.RETRIES + 1):
            try:
                response = await self.http_client.post(
                    self.config.base_url,
                    json=payload,
                    headers=headers,
                )

                response.raise_for_status()

                result = response.json()

                return self._parse_response(result)
            except HTTPStatusError as e:
                retryable = e.response.status_code in self.RETRYABLE_STATUSES
                if not retryable or attempt == self.RETRIES:
                    raise
                await asyncio.sleep(self.BASE_DELAY * 2**attempt)
            except ConnectionError as e:
                if attempt == self.RETRIES:
                    raise e
                await asyncio.sleep(self.BASE_DELAY * 2**attempt)

        raise AssertionError("unreachable")

    @staticmethod
    def _parse_response(data: dict) -> ChatResponse:
        """Разбирает сырой JSON OpenAI-формата в ChatResponse; невалидный JSON аргументов становится {}."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            func = raw.get("function", {})
            try:
                args = json.loads(func.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=raw.get("id", ""), name=func.get("name", ""), arguments=args))

        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            model=data.get("model"),
            usage=data.get("usage"),
        )
