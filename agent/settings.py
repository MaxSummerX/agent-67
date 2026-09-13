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

API_KEY = os.environ["API_KEY"]  # обязателен, падает сразу при отсутствии
BASE_URL = os.environ.get("BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")
