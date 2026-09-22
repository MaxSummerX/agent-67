import asyncio
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from httpx2 import AsyncClient, HTTPStatusError, TimeoutException


RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
RETRIES = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0
MIN_DELAY = 0.5
JITTER = 0.5


def _parse_retry_after(err: HTTPStatusError) -> float | None:
    """
    Возвращает задержку перед повторным запросом, указанную сервером, в секундах.

    Поддерживает:
    - Retry-After: задержка в секундах;
    - Retry-After: HTTP-дата, до которой нужно ждать;
    - X-RateLimit-Reset: Unix timestamp в секундах или миллисекундах.
    """
    headers = err.response.headers

    raw = headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(raw)

            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)

            return max(0.0, (retry_at - datetime.now(tz=UTC)).total_seconds())
        except (ValueError, TypeError):
            pass

    reset = headers.get("X-RateLimit-Reset")

    if reset:
        try:
            timestamp = float(reset)
        except ValueError:
            return None

        # Некоторые провайдеры передают timestamp в миллисекундах.
        if timestamp > 1e12:
            timestamp /= 1000

        return max(0.0, timestamp - time.time())
    return None


async def post_with_retries(
    client: AsyncClient, url: str, *, json: dict | None = None, data: dict | None = None, headers: dict | None = None
) -> dict[str, Any]:
    """
    POST с retry для stateless LLM-эндпоинтов (повтор запроса не создаёт сайд-эффектов).

    Временные ошибки: 429/5xx + транспортные. Задержка - от сервера (Retry-After,
    X-RateLimit-Reset), иначе свой backoff с full jitter. Неповторимые ошибки и
    исчерпанные попытки уходят вызывающему.

    Не использовать для мутирующих API: POST без idempotency-key при ретрае дублирует эффекты.
    """
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(url, json=json, data=data, headers=headers)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except HTTPStatusError as err:
            if err.response.status_code not in RETRYABLE_STATUSES or attempt == RETRIES:
                raise
            delay = _parse_retry_after(err)
            if delay is None:
                delay = random.uniform(0, BASE_DELAY * 2**attempt)  # nosec B311
            else:
                delay += random.uniform(0, JITTER)  # nosec B311
        except (ConnectionError, TimeoutException):
            if attempt == RETRIES:
                raise
            delay = random.uniform(0, BASE_DELAY * 2**attempt)  # nosec B311

        delay = min(max(delay, MIN_DELAY), MAX_DELAY)

        await asyncio.sleep(delay)

    raise AssertionError("unreachable")
