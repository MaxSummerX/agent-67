import asyncio
import json
import os
import re
from pathlib import Path

from agent.core import BaseConversationStore


_ID_SANITIZE = re.compile(r"[^a-zA-Z0-9_-]")


class JSONConversationStore(BaseConversationStore):
    """История диалогов в JSONL-файлах: по одному файлу на беседу в history, append-only."""

    def __init__(self, directory: Path = Path("history"), max_history: int = 100) -> None:
        self.directory = directory
        self.max_history = max_history
        self.directory.mkdir(exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        safe_id = _ID_SANITIZE.sub("_", conversation_id)
        return self.directory / f"{safe_id}.jsonl"

    async def load(self, conversation_id: str) -> list[dict]:
        """
        Возвращает последние max_history сообщений; битая последняя строка пропускается.

        Чтение с диска - через to_thread, чтобы не блокировать event loop.
        """
        path = self._path(conversation_id)

        def _load() -> list[dict]:
            if not path.exists():
                return []
            lines = path.read_text().splitlines()
            if not lines:
                return []
            try:
                return [json.loads(line) for line in lines[-self.max_history :]]
            except json.JSONDecodeError:
                return [json.loads(line) for line in lines[-self.max_history - 1 : -1]]

        return await asyncio.to_thread(_load)

    async def save(self, conversation_id: str, data: list[dict]) -> None:
        """Атомарно перезаписывает файл; IO через to_thread."""
        path = self._path(conversation_id)

        def _write() -> None:
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(json.dumps(msgs, ensure_ascii=False) + "\n" for msgs in data))
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

    async def append(self, conversation_id: str, data: list[dict]) -> None:
        """Дописывает сообщения строками в конец файла."""
        path = self._path(conversation_id)
        payload = "".join(json.dumps(msgs, ensure_ascii=False) + "\n" for msgs in data)

        def _append() -> None:
            with path.open("a", encoding="utf-8") as file:
                file.write(payload)
                file.flush()

        await asyncio.to_thread(_append)
