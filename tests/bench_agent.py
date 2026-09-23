"""
Бенчмарк агента на core-стеке: задачи через инструменты + автоматическая валидация.

Не pytest-тест — требует живого LLM. Запуск:
    PROVIDER=openrouter uv run python tests/bench_agent.py
    PROVIDER=ollama   uv run python tests/bench_agent.py
    PROVIDER=giga uv run python tests/bench_agent.py
"""

import asyncio
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2

from agent.composition import create_agent
from agent.core import Agent


SANDBOX = Path("sandbox").resolve()


class AgentBench:
    """Прогоняет задачи через агента и валидирует ответы."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.conversation_id = "bench"
        self.results: list[dict[str, Any]] = []

    async def run_test(self, name: str, prompt: str, validator: Callable[[str], bool]) -> dict[str, Any]:
        """Одна задача: отправить промпт агенту, проверить ответ валидатором."""
        print(f"\n{'=' * 60}\n🧪 {name}\n{'=' * 60}")

        start = time.perf_counter()
        try:
            answer, usage = await self.agent.run(prompt, self.conversation_id)
            elapsed = time.perf_counter() - start
            passed = validator(answer)
            result: dict[str, Any] = {
                "name": name,
                "passed": passed,
                "time": round(elapsed, 2),
                "tokens_in": usage.prompt_tokens,
                "tokens_out": usage.completion_tokens,
                "response": answer[:500],
            }
            print(f"{'✅ PASSED' if passed else '❌ FAILED'} (за {result['time']}с)")
        except Exception as e:  # noqa: BLE001
            result = {
                "name": name,
                "passed": False,
                "time": round(time.perf_counter() - start, 2),
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"❌ ERROR: {e}")

        self.results.append(result)
        return result

    def print_summary(self) -> None:
        """Итоги и JSON-отчёт в sandbox/test_report.json."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        avg = sum(r["time"] for r in self.results) / total if total else 0

        print(f"\n{'=' * 60}\n📊 ИТОГИ\n{'=' * 60}")
        print(f"Пройдено: {passed}/{total} ({passed / total * 100:.0f}%)")
        print(f"Среднее время: {avg:.1f}с")

        SANDBOX.mkdir(exist_ok=True)
        report = SANDBOX / "test_report.json"
        report.write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📁 Отчёт: {report}")


# --- Валидаторы ---


def _has(response: str, *keywords: str) -> bool:
    lower = response.lower()
    return any(kw.lower() in lower for kw in keywords)


def _refused(response: str) -> bool:
    """Ответ-отказ: агент задачу не выполнил, даже если упомянул ключевые слова."""
    return _has(response, "нет доступа", "не могу получить", "не могу выполнить", "нет такой возможности")


def _ok(response: str, *keywords: str) -> bool:
    """Ключевое слово найдено И это не отказ."""
    return _has(response, *keywords) and not _refused(response)


def validate_system_info(r: str) -> bool:
    return _ok(r, "cpu", "память", "ядер", "GHz", "GiB")


def validate_python_files(r: str) -> bool:
    return ".py" in r or _has(r, "python")


def validate_dir_listing(r: str) -> bool:
    return any(ind in r for ind in ("📁", "📄", ".py")) or _has(r, "директор", "файл", "folder", "file")


def validate_created(r: str) -> bool:
    return _has(r, "успешно", "создан", "записан")


def validate_hello(r: str) -> bool:
    return "hello from test" in r.lower()


def validate_grep(r: str) -> bool:
    return _has(r, "найден", "файл")


def validate_script_run(r: str) -> bool:
    return any(c.isdigit() for c in r) or "202" in r


def validate_json_format(r: str) -> bool:
    return bool(re.search(r'\{[^{}]*"[^"]+"\s*:\s*[^{}]*\}', r))


def validate_csv_format(r: str) -> bool:
    lines = [ln for ln in r.split("\n") if "," in ln and not ln.strip().startswith("#")]
    return len(lines) >= 2


def validate_file_edit(r: str) -> bool:
    return _ok(r, "успешно", "изменён", "отредактирован", "теперь", "добавлен")


def validate_log_parse(r: str) -> bool:
    return _has(r, "error", "warning", "info", "лог", "запис") and _has(r, "найден", "обнаружен", "запис", "количеств")


def validate_tree_structure(r: str) -> bool:
    return any(ind in r for ind in ("├──", "└──", "│")) or _has(r, "директори")


def validate_directory_stats(r: str) -> bool:
    has_numbers = any(c.isdigit() for c in r)
    return has_numbers and _has(r, "файл", "директори", "строк")


def validate_json_parse(r: str) -> bool:
    has_json = bool(re.search(r'\{[^{}]*"[^"]+"\s*:\s*[^{}]*\}', r))
    has_table = bool(re.search(r"^\|.*\d.*\|$", r, flags=re.M))
    has_list = bool(re.search(r"^[-*]\s+.+\d+", r, flags=re.M))
    return has_json or has_table or has_list


TESTS: list[dict[str, Any]] = [
    {
        "name": "L1: информация о CPU",
        "prompt": "Покажи информацию о CPU (например, через lscpu)",
        "validator": validate_system_info,
    },
    {
        "name": "L1: найти Python файлы",
        "prompt": "Найди все Python файлы в текущей директории",
        "validator": validate_python_files,
    },
    {
        "name": "L1: содержимое директории",
        "prompt": "Покажи содержимое текущей директории",
        "validator": validate_dir_listing,
    },
    {
        "name": "L2: создать файл",
        "prompt": "Создай файл hello.txt с текстом 'Hello from test!'",
        "validator": validate_created,
    },
    {"name": "L2: прочитать файл", "prompt": "Прочитай содержимое файла hello.txt", "validator": validate_hello},
    {
        "name": "L2: grep по файлам",
        "prompt": "Найди в файлах текущей директории (рекурсивно) слово 'Hello' через grep. Покажи файлы и строки с совпадениями.",
        "validator": validate_grep,
    },
    {
        "name": "L3: скрипт с JSON-выводом",
        "prompt": (
            "Создай Python скрипт system_info.py, который выводит информацию о системе "
            "(hostname, user, os, python_version) в формате JSON. Затем выполни его и покажи вывод."
        ),
        "validator": validate_json_format,
    },
    {
        "name": "L2: создать скрипт с датой",
        "prompt": "Создай Python скрипт date.py, который выводит текущую дату",
        "validator": lambda r: validate_created(r) or "дата" in r.lower(),
    },
    {
        "name": "L3: скрипт с CSV-выводом",
        "prompt": (
            "Создай Python скрипт disk_usage.py, который выводит использование диска в формате CSV "
            "(filesystem, size, used, available, percent, mount). Затем выполни его и покажи вывод."
        ),
        "validator": validate_csv_format,
    },
    {
        "name": "L3: редактирование файла",
        "prompt": "Измени файл hello.txt, добавив строку 'Second line' в конец",
        "validator": validate_file_edit,
    },
    {
        "name": "L3: выполнить скрипт",
        "prompt": "Выполни скрипт date.py через python",
        "validator": validate_script_run,
    },
    {
        "name": "L3: подсчёт строк кода",
        "prompt": "Подсчитай количество строк во всех .py файлах текущей директории",
        "validator": validate_directory_stats,
    },
    {
        "name": "L4: дерево директорий",
        "prompt": "Построй дерево директорий текущей папки с количеством файлов в каждой поддиректории. Используй find или tree.",
        "validator": validate_tree_structure,
    },
    {
        "name": "L4: статистика по директории",
        "prompt": "Проанализируй текущую директорию: сколько .py файлов, сколько всего строк кода в них.",
        "validator": validate_directory_stats,
    },
    {
        "name": "L4: парсинг вывода команды",
        "prompt": "Выполни команду 'ls -la --time-style=+%s' и извлеки имена файлов и их размеры в структурированном виде.",
        "validator": validate_json_parse,
    },
    {
        "name": "L4: анализ лог-файла",
        "prompt": (
            "Создай тестовый лог-файл test.log с записями уровней INFO, WARNING, ERROR, "
            "затем проанализируй его и покажи количество записей каждого уровня."
        ),
        "validator": validate_log_parse,
    },
]


async def main() -> None:
    """Собирает агента под провайдера из PROVIDER и прогоняет все задачи."""
    provider = os.getenv("PROVIDER", "openrouter").lower()

    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True)

    timeout = httpx2.Timeout(300)
    async with httpx2.AsyncClient(timeout=timeout) as http_client:
        base_provider = {"openrouter": "open_router", "giga": "giga_chat"}.get(provider, provider)
        agent = create_agent(http_client, persist=False, workspace=str(SANDBOX), provider=base_provider)
        print(f"🌐 {base_provider}, sandbox: {SANDBOX}")

        bench = AgentBench(agent)
        for test in TESTS:
            await bench.run_test(**test)
            await asyncio.sleep(5)  # free-тир OpenRouter: 20 запросов/мин

        bench.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
