import difflib
from pathlib import Path
from typing import Any

from agent.core.tools.base import BaseTool, ToolResult


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
) -> Path:
    """
    Разрешает путь с учётом рабочей директории и ограничений доступа.

    Args:
        path: Исходный путь (относительный или абсолютный).
        workspace: Рабочая директория для относительных путей.
        allowed_dir: Разрешённая директория для проверки доступа.

    Returns:
        Path: Разрешённый абсолютный путь.

    Raises:
        PermissionError: Если путь находится вне разрешённой директории.
        ValueError: Если путь относительный, а workspace и allowed_dir не заданы.
    """
    pathway = Path(path)
    if not pathway.is_absolute():
        base = workspace or allowed_dir
        if base is None:
            raise ValueError("Относительный путь требует workspace или allowed_dir")
        pathway = base / pathway
    resolved = pathway.resolve()
    if allowed_dir:
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            raise PermissionError(f"Путь {path} находится вне разрешённой директории {allowed_dir}") from None
    return resolved


class ReadFileTool(BaseTool):
    """
    Инструмент для чтения содержимого файла.

    Позволяет агенту читать текстовые файлы из файловой системы
    с учётом ограничений доступа.
    """

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None) -> None:
        """Инициализирует инструмент чтения файла.

        Args:
            workspace: Рабочая директория для относительных путей.
            allowed_dir: Разрешённая директория для ограничения доступа.
        """
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Прочитать содержимое файла по указанному пути."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу для чтения.",
                }
            },
            "required": ["path"],
        }

    async def execute(self, args: dict) -> ToolResult:
        try:
            path = args["path"]
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return ToolResult.error(f"Ошибка: файл не найден: {path}")
            if not file_path.is_file():
                return ToolResult.error(f"Ошибка: это не файл: {path}")

            content = file_path.read_text(encoding="utf-8")
            return ToolResult(content)
        except PermissionError as e:
            return ToolResult.error(f"Ошибка: {e}")
        except Exception as e:
            return ToolResult.error(f"Ошибка чтения файла: {e}")


class WriteFileTool(BaseTool):
    """
    Инструмент для записи содержимого в файл.

    Позволяет агенту создавать и перезаписывать текстовые файлы.
    Родительские директории создаются автоматически при необходимости.
    """

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None) -> None:
        """Инициализирует инструмент записи файла.

        Args:
            workspace: Рабочая директория для относительных путей.
            allowed_dir: Разрешённая директория для ограничения доступа.
        """
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Записать содержимое в файл по указанному пути. Родительские директории создаются при необходимости."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Путь к файлу для записи",
                },
                "content": {
                    "type": "string",
                    "description": "Содержимое для записи",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, args: dict) -> ToolResult:
        try:
            path, content = args["path"], args["content"]
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(f"Успешно записано {len(content)} символов в {file_path}")
        except PermissionError as e:
            return ToolResult.error(f"Ошибка: {e}")
        except Exception as e:
            return ToolResult.error(f"Ошибка записи файла: {e}")


class EditFileTool(BaseTool):
    """
    Инструмент для редактирования файла путем замены текста.

    Позволяет агенту вносить точечные изменения в файлы,
    заменяя одну строку на другую. Старый текст должен
    присутствовать в файле точно в таком виде.
    """

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None) -> None:
        """Инициализирует инструмент редактирования файла.

        Args:
            workspace: Рабочая директория для относительных путей.
            allowed_dir: Разрешённая директория для ограничения доступа.
        """
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Отредактировать файл, заменив old_text на new_text. old_text должен точно присутствовать в файле."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к редактируемому файлу"},
                "old_text": {
                    "type": "string",
                    "description": "Точный текст для поиска и замены",
                },
                "new_text": {
                    "type": "string",
                    "description": "Текст для замены",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, args: dict) -> ToolResult:
        try:
            path, old_text, new_text = args["path"], args["old_text"], args["new_text"]
            file_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not file_path.exists():
                return ToolResult.error(f"Ошибка: файл не найден: {path}")

            content = file_path.read_text(encoding="utf-8")

            if old_text not in content:
                return ToolResult.error(self._not_found_message(old_text, content, path))

            count = content.count(old_text)
            if count > 1:
                return ToolResult.error(
                    f"Ошибка: old_text встречается {count} раз. Уточните контекст, чтобы замена была однозначной."
                )

            new_content = content.replace(old_text, new_text, 1)
            file_path.write_text(new_content, encoding="utf-8")

            return ToolResult(f"Файл успешно отредактирован: {file_path}")
        except PermissionError as e:
            return ToolResult.error(f"Ошибка: {e}")
        except Exception as e:
            return ToolResult.error(f"Ошибка редактирования файла: {e}")

    @staticmethod
    def _not_found_message(old_text: str, content: str, path: str) -> str:
        """Формирует сообщение об ошибке, если искомый текст не найден.

        Пытается найти наиболее похожий фрагмент текста и показывает diff,
        чтобы помочь понять, в чём различие.

        Args:
            old_text: Искомый текст, который не был найден.
            content: Содержимое файла.
            path: Путь к файлу для сообщения об ошибке.

        Returns:
            str: Сообщение об ошибке с возможным diff похожего текста.
        """
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio > 0.5:
            diff = "\n".join(
                difflib.unified_diff(
                    old_lines,
                    lines[best_start : best_start + window],
                    fromfile="old_text (передан)",
                    tofile=f"{path} (в файле, строка {best_start + 1})",
                    lineterm="",
                )
            )
            return f"Ошибка: old_text не найден в {path}.\nНаиболее похожий фрагмент ({best_ratio:.0%} совпадения) на строке {best_start + 1}:\n{diff}"
        return f"Ошибка: old_text не найден в {path}. Похожий текст не найден, проверьте содержимое файла."


class ListDirTool(BaseTool):
    """
    Инструмент для просмотра содержимого директории.

    Позволяет агенту получать список файлов и поддиректорий
    в указанной директории. Директории помечаются значком 📁,
    файлы — значком 📄.
    """

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        """Инициализирует инструмент просмотра директории.

        Args:
            workspace: Рабочая директория для относительных путей.
            allowed_dir: Разрешённая директория для ограничения доступа.
        """
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "Показать содержимое директории."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Путь к директории"}},
            "required": ["path"],
        }

    async def execute(self, args: dict) -> ToolResult:
        try:
            path = args["path"]
            dir_path = _resolve_path(path, self._workspace, self._allowed_dir)
            if not dir_path.exists():
                return ToolResult.error(f"Ошибка: директория не найдена: {path}")
            if not dir_path.is_dir():
                return ToolResult.error(f"Ошибка: это не директория: {path}")

            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")

            if not items:
                return ToolResult(f"Директория {path} пуста")

            return ToolResult("\n".join(items))
        except PermissionError as e:
            return ToolResult.error(f"Ошибка: {e}")
        except Exception as e:
            return ToolResult.error(f"Ошибка чтения директории: {e}")
