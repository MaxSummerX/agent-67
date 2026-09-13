# agent-67

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/MaxSummerX/agentSDK/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxSummerX/agentSDK/actions/workflows/ci.yml)

---
Минималистичный агент на чистом Python: свой цикл tool-calling, LLM через OpenRouter, без тяжёлых фреймворков.

## Возможности

- Цикл «модель -> инструменты -> модель» с ограничением раундов
- Tool-calling в формате OpenAI (совместимо с OpenRouter)
- Инструменты: веб-поиск (ddgs), загрузка страниц (urllib)
- История диалогов: в памяти или в JSON-файлах
- Retry с экспоненциальным бэкоффом на 429/5xx и сетевые сбои
- Слоёная структура: `core` (интерфейсы) / `infrastructure` (реализации) / `composition` (сборка)

## Безопасность

- **SSRF-защита**: `fetch_url` резолвит хост и блокирует приватные, loopback и cloud-metadata адреса
- **Prompt injection**: веб-контент оборачивается рамкой «данные, не инструкции»
- **Path traversal**: идентификаторы диалогов санитизируются; история пишется атомарно (tmp + `os.replace`)
- Ошибки инструментов возвращаются модели как tool-result - цикл не падает

## Запуск

```bash
uv sync
cp .env.example .env  # вписать API_KEY
uv run python -m agent.first_agent
```

## Конфигурация

`.env` в корне проекта:

```
API_KEY=...                          # ключ OpenRouter (обязателен)
BASE_URL=https://openrouter.ai/api/v1/chat/completions
MODEL=inclusionai/ling-3.0-flash-sante:free
```

Free-модели OpenRouter часто отдают 429 — общий пул модели перегружен, это не лимит вашего ключа.
`MODEL=openrouter/free` — роутер сам выбирает доступную free-модель.

## Структура

```
agent/
  core/            # интерфейсы + цикл агента (tool-calling loop)
  infrastructure/  # OpenRouter-клиент, инструменты, хранилища истории
  composition/     # точка сборки агента
  prompts/         # системный промпт
tests/             # тесты
```

## Разработка

```bash
uv run pytest tests/ -q          # тесты
uv run ruff check .              # линтер
uv run mypy .                    # типы
```
