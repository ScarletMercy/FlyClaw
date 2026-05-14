"""Lightweight tool definition.

Tools are plain async functions. ToolDef wraps them with the metadata needed
for OpenAI tool calling (name, description, parameter schema).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

logger = logging.getLogger("myclaw.agent.tooldef")

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable
    _valid_params: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        if not self._valid_params:
            sig = inspect.signature(self.fn)
            self.__dict__['_valid_params'] = frozenset(
                p for p in sig.parameters if p not in ("self", "cls")
            )

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, args: dict) -> str:
        import inspect as _inspect
        filtered = {k: v for k, v in args.items() if k in self._valid_params}
        result = self.fn(**filtered)
        if _inspect.isawaitable(result):
            result = await result
        return str(result) if result is not None else ""

    @classmethod
    def from_function(cls, fn: Callable, name: str | None = None) -> ToolDef:
        tool_name = name or fn.__name__
        description = _extract_description(fn)
        parameters = _extract_parameters(fn)
        return cls(name=tool_name, description=description, parameters=parameters, fn=fn)

    @classmethod
    def from_schema(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable,
    ) -> ToolDef:
        return cls(name=name, description=description, parameters=parameters, fn=fn)


def _extract_description(fn: Callable) -> str:
    doc = fn.__doc__
    if not doc:
        return ""
    lines = doc.strip().splitlines()
    desc_parts: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Args:") or stripped.startswith("Returns:") or stripped.startswith("Example"):
            break
        if stripped == "" and desc_parts:
            break
        if stripped:
            desc_parts.append(stripped)
    return " ".join(desc_parts) if desc_parts else lines[0].strip()


def _extract_parameters(fn: Callable) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        prop: dict[str, Any] = {}

        py_type = hints.get(param_name, param.annotation)
        if py_type is inspect.Parameter.empty:
            prop["type"] = "string"
        else:
            json_type = _resolve_type(py_type)
            prop.update(json_type)

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            if isinstance(param.default, bool):
                prop["default"] = param.default
            elif isinstance(param.default, (int, float, str)):
                prop["default"] = param.default

        properties[param_name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _resolve_type(py_type: Any) -> dict[str, Any]:
    origin = getattr(py_type, "__origin__", None)

    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if origin is set:
        return {"type": "array", "uniqueItems": True}

    if origin is object or (hasattr(py_type, "__args__") and py_type.__args__):
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0])
        if len(non_none) > 1:
            return _resolve_type(non_none[0])

    if py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}

    return {"type": "string"}
