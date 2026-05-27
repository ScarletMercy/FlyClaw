"""Lightweight tool definition.

Tools are plain async functions. ToolDef wraps them with the metadata needed
for OpenAI tool calling (name, description, parameter schema).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_type_hints

logger = logging.getLogger("flyclaw.agent.tooldef")

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


def _parse_args_doc(fn: Callable) -> dict[str, str]:
    doc = fn.__doc__
    if not doc:
        return {}
    parts = doc.split("Args:")
    if len(parts) < 2:
        return {}
    tail = parts[1]
    for stop in ("Returns:", "Raises:", "Example"):
        tail = tail.split(stop)[0]
    sig = inspect.signature(fn)
    valid_names = frozenset(p for p in sig.parameters if p not in ("self", "cls"))
    result: dict[str, str] = {}
    current_name = ""
    current_desc: list[str] = []
    for raw_line in tail.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        colon_pos = stripped.find(":")
        candidate = stripped[:colon_pos].replace("_", "").replace("-", "") if colon_pos > 0 else ""
        if candidate and candidate.isalnum() and stripped[:colon_pos].strip() in valid_names:
            if current_name:
                result[current_name] = " ".join(current_desc).strip()
            current_name = stripped[:colon_pos].strip()
            current_desc = [stripped[colon_pos + 1:].strip()]
        elif current_name:
            current_desc.append(stripped)
    if current_name:
        result[current_name] = " ".join(current_desc).strip()
    return result


def _extract_parameters(fn: Callable) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    arg_docs = _parse_args_doc(fn)

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

        if param_name in arg_docs:
            prop["description"] = arg_docs[param_name]

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
        args = getattr(py_type, "__args__", ())
        if args:
            return {"type": "array", "items": _resolve_type(args[0])}
        return {"type": "array"}
    if origin is dict:
        args = getattr(py_type, "__args__", ())
        if args and len(args) >= 2:
            return {"type": "object", "additionalProperties": _resolve_type(args[1])}
        return {"type": "object"}
    if origin is set:
        args = getattr(py_type, "__args__", ())
        result = {"type": "array", "uniqueItems": True}
        if args:
            result["items"] = _resolve_type(args[0])
        return result

    if origin is Literal:
        args = getattr(py_type, "__args__", ())
        values = [a for a in args if a is not type(None)]
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        return {"type": "string", "enum": [str(v) for v in values]}

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
