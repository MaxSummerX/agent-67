# agent-67

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/MaxSummerX/agent-67/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxSummerX/agent-67/actions/workflows/ci.yml)
---
Минималистичный агент на чистом Python: свой цикл tool-calling, три LLM-провайдера (OpenRouter / Ollama / GigaChat), опциональный слой типизированных решений (Jev), без тяжёлых фреймворков.

## Возможности

- Цикл «модель -> инструменты -> модель» с ограничением раундов
- Мультипровайдерность: OpenRouter и локальная Ollama говорят в OpenAI-формате, GigaChat - собственный function_call-диалект
- Движок типизированных решений (Jev / System One). Опционален - без ключа агент работает как раньше
- Tool-calling в формате OpenAI
- Инструменты: веб-поиск (ddgs), загрузка страниц (readability + HTML -> Markdown),
  работа с файлами (read/write/edit/list), shell (exec с safety guard)
- CLI на rich: баннер, беседы `/new`, `/list`, `/open`, `/help`,
  панель статистики токенов/кэша/времени раунда
- Prompt caching для anthropic/gemini/qwen, retry с бэкоффом на 429/5xx, таймауты и сетевые сбои
- История диалогов: в памяти или в JSON-файлах
- Слоёная структура: `core` (интерфейсы) / `infrastructure` (реализации) / `composition` (сборка)

## Безопасность

- **SSRF-защита**: `fetch_url` резолвит хост и блокирует приватные, loopback и cloud-metadata адреса
- **Prompt injection**: веб-контент оборачивается рамкой «данные, не инструкции»
- **Sandbox файлов**: workspace/allowed_dir с защитой от `../` и симлинков (`resolve()` + проверка границы)
- **Exec guard**: deny/allow паттерны, ограничение рабочей директории, kill группы процессов при таймауте
- **Path traversal**: идентификаторы диалогов санитизируются; история пишется атомарно (tmp + `os.replace`)
- Ошибки инструментов возвращаются модели как tool-result - цикл не падает

## Запуск

```bash
uv sync
cp .env.example .env  # выбрать провайдера и вписать ключи
uv run python -m agent.first_agent
```

## Провайдеры

Выбор — переменной `PROVIDER` в `.env`.

### OpenRouter (по умолчанию)

```
PROVIDER=open_router
API_KEY=...                          # ключ OpenRouter (обязателен)
BASE_URL=https://openrouter.ai/api/v1/chat/completions
MODEL=укажите_свою_модель
```

Free-модели OpenRouter часто отдают 429 — общий пул модели перегружен, это не лимит вашего ключа.
`MODEL=openrouter/free` — роутер сам выбирает доступную free-модель.

### Ollama (локально, без ключей)

```
PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1/chat/completions
OLLAMA_MODEL=укажите_свою_модель
```

Требуется установленный [Ollama](https://ollama.com) и скачанная модель: `ollama pull <model>`.

### GigaChat (Sber AI)

```
PROVIDER=giga_chat
GIGACHAT_CREDENTIALS=...             # base64 client_id:client_secret
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat-3-Lightning
```

Credentials выдаются в [Sber Studio](https://developers.sber.ru/studio/) при регистрации приложения.
Авторизация OAuth: клиент получает и обновляет access-токен сам, api_key не нужен.
Нюансы протокола (валидный JSON в результате функции, arguments объектом) клиент конвертирует сам.

## Движок решений (Jev, опционально)

Ось System One: агент может спрашивать у модели решений - не текст, а типизированные
ответы (`Choice` / `Score` / `Noul`) с распределением вероятностей и уверенностью.
Подключается автоматически при наличии ключа:

```
JEV_API_KEY=...                      # ключ TypeSafe или OpenRouter
JEV_BASE_URL=https://openrouter.ai/api/v1/systemone   # или https://api.typesafe.ai/v1/systemone
```

Без `JEV_API_KEY` движок не подключается - поведение агента не меняется.

```python
from agent.core.decision import BaseDecisionEngine, Choice, Noul
from agent.core.tools.base import BaseTool, ToolResult


# JevEngine - реализация по умолчанию; свой провайдер подключается через BaseDecisionEngine
async def guard_exec(
    engine: BaseDecisionEngine,
    exec_tool: BaseTool,
    command: str,
) -> ToolResult:
    answers = await engine.decide(
        state={"команда": command},
        questions={
            "безопасно": Noul(instructions="Безопасно ли выполнить `команду` в sandbox агента?"),
            "риск": Choice(
                criteria={
                    "ок": "Обычная команда: чтение, сборка, тесты, работа с файлами в workspace",
                    "деструктив": "Удаление или перезапись данных вне workspace",
                    "сетевой_запуск": "Скачивание и запуск кода из сети, piping в shell",
                }
            ),
        },
    )
    if answers["безопасно"].value < 0.7:
        return ToolResult.error(f"Отклонено ({answers['риск'].value}), уверенность {answers['безопасно'].value:.2f}")
    return await exec_tool.execute({"command": command})
```

Все вопросы уходят одним batched-запросом и оцениваются параллельно.

## Структура

```
agent/
  core/            # интерфейсы + цикл агента (tool-calling loop)
    decision.py               # контракт движка решений
  infrastructure/  # LLM-клиенты, инструменты, хранилища истории
    llm/_http.py             # retry-транспорт
    llm/openai_compatible.py # база OpenAI-совместимых провайдеров
    llm/giga_chat.py         # GigaChat: OAuth + конвертация форматов
    decision/jev.py          # JevEngine: Jev заперт здесь
  composition/     # реестр провайдеров create_llm, точка сборки агента
  prompts/         # системный промпт
  ui.py            # rich-отображение CLI
tests/
  bench_agent.py   # smoke-бенч на живом LLM
```

## Бенчмарк

Прогон 16 задач L1–L4 (exec/filesystem/grep/json/csv) через инструменты агента
с автоматической валидацией ответов; отчёт — `sandbox/test_report.json`.
Не pytest-тест, требует живого LLM:

```bash
uv run python tests/bench_agent.py              # провайдер из .env
PROVIDER=ollama uv run python tests/bench_agent.py
```

## Разработка

```bash
uv run pytest tests/ -q          # тесты
uv run ruff check .              # линтер
uv run mypy .                    # типы
```
