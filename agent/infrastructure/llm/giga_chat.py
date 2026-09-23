"""Клиент GigaChat: OAuth-авторизация и конвертация OpenAI в GigaChat-формат."""

import json
import time
import uuid
from typing import Any

from httpx2 import AsyncClient

from agent.core import BaseLLM, BaseLLMConfig, ChatResponse, ToolCall
from agent.infrastructure.llm._http import post_with_retries


class GigaChatConfig(BaseLLMConfig):
    """Конфиг GigaChat: credentials/scope вместо api_key (OAuth, base64 client_id:client_secret)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        credentials: str | None = None,
        scope: str | None = None,
    ) -> None:
        super().__init__(base_url, model, api_key, temperature, max_tokens)
        self.credentials = credentials
        self.scope = scope


class GigaChatClient(BaseLLM):
    """Клиент GigaChat: OAuth-токен, function_call-формат, своя конвертация сообщений."""

    GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def __init__(
        self,
        config: GigaChatConfig,
        http_client: AsyncClient,
        timeout: float = 120,
        debug: bool = False,
    ) -> None:
        """Инициализирует клиент; без credentials падает сразу."""
        self.config: GigaChatConfig = config
        self.http_client = http_client
        self.debug = debug
        self._access_token: str | None = None
        self._token_expires_at: float = 0

        if not self.config.credentials:
            raise ValueError(
                "GigaChat: credentials не заданы. Укажите переменную окружения "
                "GIGACHAT_CREDENTIALS или передайте параметр credentials."
            )

    async def _get_gigachat_token(self) -> str:
        """Возвращает действующий OAuth-токен, при истечении получает новый."""
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        data = await post_with_retries(
            self.http_client,
            self.GIGACHAT_AUTH_URL,
            data={"scope": self.config.scope},
            headers={
                "Authorization": f"Basic {self.config.credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
                "RqUID": str(uuid.uuid4()),
                "Accept": "application/json",
            },
        )

        token = data.get("access_token") or ""
        if not token:
            raise ValueError("GigaChat: не удалось получить access_token")
        self._access_token = token

        expires_at = data.get("expires_at", 0)
        self._token_expires_at = expires_at / 1000 if expires_at > 1e12 else expires_at

        return token

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Отправляет запрос к GigaChat, конвертируя сообщения из OpenAI-формата, и возвращает ChatResponse."""
        payload = {
            "model": self.config.model,
            "messages": [self._to_gigachat_message(m) for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "agent-67",
        }

        token = await self._get_gigachat_token()
        headers["Authorization"] = f"Bearer {token}"

        if tools:
            payload["functions"] = [t["function"] for t in tools]
            payload["function_call"] = "auto"

        result = await post_with_retries(
            self.http_client,
            self.config.base_url,
            json=payload,
            headers=headers,
        )
        return self._parse_response(result)

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        """Разбирает ответ GigaChat: function_call вместо tool_calls, id берётся из functions_state_id."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        calls: list[ToolCall] = []
        function_call = message.get("function_call")
        if function_call:
            args = function_call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args or "{}")
            calls = [
                ToolCall(
                    id=message.get("functions_state_id") or uuid.uuid4().hex,
                    name=function_call["name"],
                    arguments=args,
                )
            ]

        return ChatResponse(
            content=message.get("content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            model=data.get("model"),
            usage=data.get("usage"),
        )

    def _to_gigachat_message(self, msg: dict) -> dict:
        """Конвертирует OpenAI-сообщение в GigaChat-формат: tool → function, tool_calls → function_call + functions_state_id."""
        if msg.get("role") == "tool":
            content = msg["content"]
            try:
                json.loads(content)
            except (json.JSONDecodeError, TypeError):
                content = json.dumps({"result": str(content)}, ensure_ascii=False)
            return {"role": "function", "content": content}
        if "tool_calls" in msg:
            call = msg["tool_calls"][0]  # GigaChat не параллелит функции, берём первую
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "functions_state_id": call["id"],
                "function_call": {
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"] or "{}"),
                },
            }
        return msg
