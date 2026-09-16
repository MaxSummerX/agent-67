from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


class ToolResult(str):
    """Строково-совместимый вывод инструмента со структурированным статусом."""

    is_error: bool

    def __new__(cls, content: str, *, is_error: bool = False) -> ToolResult:
        obj = super().__new__(cls, content)
        obj.is_error = is_error
        return obj

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content, is_error=True)


class BaseTool(ABC):
    """Инструмент агента: имя, описание, JSON-схема и исполнение."""

    _TYPE_MAPPING: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя инструмента, на которое ссылается LLM."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Краткое описание для модели: что и когда использовать."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Схема параметров инструмента в формате JSON Schema."""
        ...

    @abstractmethod
    async def execute(self, args: dict) -> ToolResult:
        """Выполняет инструмент с распакованными аргументами, возвращает текст-результат."""
        ...

    def schema(self) -> dict:
        """JSON-схема tool-calling (OpenAI-формат)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": deepcopy(self.parameters)},
        }

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Валидирует параметры согласно схеме инструмента."""
        schema = self.parameters
        if schema.get("type", "object") != "object":
            raise ValueError(f"Схема должна иметь тип object, получено {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, value: Any, schema: dict[str, Any], path: str) -> list[str]:
        """Рекурсивно валидирует значение согласно схеме JSON Schema."""
        schema_type = schema.get("type", "object")
        label = path or "parameter"

        if schema_type in self._TYPE_MAPPING and not isinstance(value, self._TYPE_MAPPING[schema_type]):
            return [f"{label} должен быть {schema_type}"]

        if schema_type in ("integer", "number") and isinstance(value, bool):
            return [f"{label} должен быть {schema_type}"]

        errors = []

        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} должен быть одним из {schema['enum']}")
        if schema_type in ("integer", "number"):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{label} должен быть >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{label} должен быть <= {schema['maximum']}")
        if schema_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{label} должно быть как минимум {schema['minLength']} chars")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{label} должно быть как максимум {schema['maxLength']} chars")
        if schema_type == "object":
            properties = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"отсутствует обязательное {path + '.' + key if path else key}")
            for key, val in value.items():
                if key in properties:
                    errors.extend(self._validate(val, properties[key], path + "." + key if path else key))
        if schema_type == "array" and "items" in schema:
            for i, item in enumerate(value):
                errors.extend(self._validate(item, schema["items"], f"{path}[{i}]" if path else f"[{i}]"))
        return errors
