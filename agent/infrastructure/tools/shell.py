"""Инструмент для выполнения shell-команд.

Этот модуль предоставляет возможность агенту выполнять команды в оболочке (shell)
с механизмами безопасности для предотвращения опасных операций.
"""

import asyncio
import os
import re
import signal
from pathlib import Path
from typing import Any

from agent.core.tools.base import BaseTool, ToolResult


KILL_WAIT_SECONDS = 5.0  # сколько ждать завершения процесса после kill()
MAX_OUTPUT_CHARS = 10_000  # лимит вывода команды для контекста модели

# Хвост к сообщениям guard'а: объясняет модели, что ретраи бесполезны
_GUARD_NOTE = (
    "\n\nЭто жёсткая граница политики, а не временная ошибка. НЕ пытайтесь обойти "
    "(символические ссылки, base64, альтернативные инструменты, смена working_dir). "
    "Если ресурс действительно нужен — сообщите, что он недоступен при текущих "
    "ограничениях, и спросите, как поступить."
)


class ExecTool(BaseTool):
    """Инструмент для выполнения shell-команд.

    Позволяет агенту выполнять команды в системной оболочке с таймаутом,
    ограничениями безопасности и возможностью указания рабочей директории.

    Guard ниже — страховка от случайных катастрофических ошибок модели
    (rm -rf, shutdown и т.п.), НЕ защита от намеренного обхода: shell-синтаксис
    (кавычки, подстановки, пайпы) regex'ом не парсится. От атак закрывает
    белый список allow_patterns, права ОС или контейнер.

    Поддерживаемые механизмы безопасности:
    - deny_patterns: список регулярных выражений для блокировки опасных команд
    - allow_patterns: белый список разрешённых шаблонов команд
    - restrict_to_workspace: запрет доступа вне рабочей директории
    """

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = True,
        path_append: str = "",
    ):
        """Инициализирует инструмент выполнения команд.

        Args:
            timeout: Максимальное время выполнения команды в секундах.
            working_dir: Рабочая директория по умолчанию для команд.
            deny_patterns: Список регулярных выражений для блокировки опасных команд.
                По умолчанию блокирует rm -rf, format, dd, shutdown и др.
            allow_patterns: Белый список разрешённых шаблонов команд. Если указан,
                выполняются только команды, соответствующие одному из шаблонов.
            restrict_to_workspace: Блокировать команды, пытающиеся выйти за пределы
                рабочей директории (обнаруживает ../ и абсолютные пути).
            path_append: Строка для добавления к переменной среды PATH.
        """
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"(?:^|[;&|]\s*)format(?!=)\b",  # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Выполнить shell-команду и вернуть её вывод. Использовать с осторожностью."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell-команда для выполнения",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Необязательная рабочая директория для команды",
                },
            },
            "required": ["command"],
        }

    async def execute(self, args: dict) -> ToolResult:
        command = args["command"]
        cwd = args.get("working_dir") or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return ToolResult.error(guard_error)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=True,  # чтобы killpg убивал и дочерние процессы
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except TimeoutError:
                await self._kill_process_tree(process)
                return ToolResult.error(f"Ошибка: время выполнения команды истекло ({self.timeout} сек)")
            except asyncio.CancelledError:
                # нас отменили (shutdown агента и т.п.) — не оставляем процесс висеть
                await self._kill_process_tree(process)
                raise

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nКод завершения: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(нет вывода)"

            # Обрезать очень длинный вывод
            if len(result) > MAX_OUTPUT_CHARS:
                result = result[:MAX_OUTPUT_CHARS] + f"\n... (обрезано, ещё {len(result) - MAX_OUTPUT_CHARS} символов)"

            return ToolResult(result)

        except Exception as e:
            return ToolResult.error(f"Ошибка выполнения команды: {e}")

    @staticmethod
    async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
        """Убивает процесс вместе с дочерними (группа процессов) и дожидается завершения."""
        try:
            os.killpg(process.pid, signal.SIGKILL)  # start_new_session=True => pid == pgid
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()  # fallback: хотя бы сам процесс
        try:
            await asyncio.wait_for(process.wait(), timeout=KILL_WAIT_SECONDS)
        except TimeoutError:
            pass

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Проверяет команду на соответствие ограничениям безопасности.

        Args:
            command: Проверяемая команда.
            cwd: Рабочая директория для проверки путей.

        Returns:
            str | None: Сообщение об ошибке, если команда заблокирована,
                иначе None.
        """

        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Ошибка: команда заблокирована защитой (обнаружен опасный шаблон)" + _GUARD_NOTE

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Ошибка: команда заблокирована защитой (нет в белом списке)" + _GUARD_NOTE

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Ошибка: команда заблокирована защитой (обнаружен выход за пределы пути)" + _GUARD_NOTE

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                p = Path(raw.strip()).resolve()
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Ошибка: команда заблокирована защитой (путь вне рабочей директории)" + _GUARD_NOTE

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        """Извлекает абсолютные пути из командной строки.

        Args:
            command: Командная строка для поиска путей.

        Returns:
            list[str]: Список найденных абсолютных путей Windows (C:\\...) и POSIX (/...).
        """
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)  # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>])(/[^\s\"'>]+)", command)  # POSIX: /absolute only
        return win_paths + posix_paths
