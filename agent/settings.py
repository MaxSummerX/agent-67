"""Конфигурация окружения: загрузка .env и настройки провайдера."""

import os
from pathlib import Path


def load_env(path: Path) -> None:
    """
    Читает KEY=VALUE из файла в os.environ, не перезаписывая уже заданные переменные.

    Пустые значения игнорируются - переменная считается незаданной.
    """
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.strip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip("\"'")
                if value:
                    os.environ.setdefault(key.strip(), value)


load_env(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.environ.get("API_KEY", "")
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS", "")
BASE_URL = os.environ.get("BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-3-Lightning")
GIGACHAT_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
JEV_API_KEY = os.environ.get("JEV_API_KEY", "")
JEV_BASE_URL = os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone")
PROVIDER = os.environ.get("PROVIDER", "open_router")
