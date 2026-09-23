"""База для OpenAI-совместимых провайдеров"""

import json
from typing import Any

from httpx2 import AsyncClient

from agent.core import BaseLLM, BaseLLMConfig, ChatResponse, ToolCall
from agent.infrastructure.llm._http import post_with_retries


class OpenAICompatibleConfig(BaseLLMConfig):
    """Конфиг OpenAI-совместимого эндпоинта; headers - статичные, провайдер-специфичные."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        headers: dict[str, str] | None = None,
        cache_system_prompt: bool = False,
    ) -> None:
        super().__init__(base_url, model, api_key, temperature, max_tokens)
        self.headers = headers or {}
        self.cache_system_prompt = cache_system_prompt


class OpenAICompatibleLLM(BaseLLM):
    """Общий chat() для всех, кто говорит в OpenAI Chat Completions формате."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        http_client: AsyncClient,
        debug: bool = False,
    ) -> None:
        super().__init__(config)
        self.config: OpenAICompatibleConfig = config
        self.http_client = http_client
        self.debug = debug

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Отправляет запрос в формате OpenAI Chat Completions и возвращает ChatResponse."""
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        headers = dict(self.config.headers)
        if self.config.api_key:
            headers.setdefault("Authorization", f"Bearer {self.config.api_key}")

        if self.config.cache_system_prompt:
            payload["messages"] = self._annotate_system_for_cache(messages)

        data = await post_with_retries(
            self.http_client,
            self.config.base_url,
            json=payload,
            headers=headers or None,
        )
        return self._parse_response(data)

    def _annotate_system_for_cache(self, messages: list[dict]) -> list[dict]:
        """Оборачивает system-промпт в content-блок с cache_control: ephemeral - запрос сервера закэшировать префикс (Anthropic/Gemini/Qwen)."""
        return [
            {
                **msg,
                "content": [
                    {
                        "type": "text",
                        "text": msg.get("content", ""),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
            if msg.get("role") == "system" and isinstance(msg.get("content"), str)
            else msg
            for msg in messages
        ]

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        """Разбирает OpenAI-ответ; невалидный JSON аргументов инструментов превращает в {}."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            func = raw.get("function", {})
            try:
                args = json.loads(func.get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append(ToolCall(id=raw.get("id", ""), name=func.get("name", ""), arguments=args))

        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            model=data.get("model"),
            usage=data.get("usage"),
        )
