import asyncio

from httpx2 import AsyncClient, HTTPStatusError

from agent.core import BaseLLM, BaseLLMConfig


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
    ) -> dict:
        """Отправляет запрос, возвращает {"message": ..., "finish_reason": ...}."""

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

                return {
                    "message": result["choices"][0]["message"],
                    "finish_reason": result["choices"][0]["finish_reason"],
                }
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
