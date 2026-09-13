import asyncio
import json
import os
import re
from pathlib import Path

from agent.core import BaseConversationStore


_ID_SANITIZE = re.compile(r"[^a-zA-Z0-9_-]")


class JSONConversationStore(BaseConversationStore):
    """История диалогов в JSON-файлах: по одному файлу на беседу в history."""

    def __init__(self, directory: Path = Path("history")) -> None:
        self.directory = directory
        self.directory.mkdir(exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        safe_id = _ID_SANITIZE.sub("_", conversation_id)
        return self.directory / f"{safe_id}.json"

    async def load(self, conversation_id: str) -> list[dict]:
        """Читает файл беседы; диск-IO через to_thread, чтобы не блокировать event loop."""
        path = self._path(conversation_id)
        if not path.exists():
            return []

        def _load() -> list[dict]:
            loaded: list[dict] = json.loads(path.read_text())
            return loaded

        return await asyncio.to_thread(_load)

    async def save(self, conversation_id: str, data: list[dict]) -> None:
        """Атомарно перезаписывает файл; IO через to_thread."""
        path = self._path(conversation_id)

        def _write() -> None:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            os.replace(tmp, path)

        await asyncio.to_thread(_write)
