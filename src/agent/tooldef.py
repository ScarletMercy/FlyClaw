"""Lightweight tool definition.

Tools are plain async functions. ToolDef wraps them with the metadata needed
for OpenAI tool calling (name, description, parameter schema).
"""

from __future__ import annotations

import enum
import inspect
import logging
import types as _bt
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union, get_type_hints

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
            self.__dict__["_valid_params"] = frozenset(p for p in sig.parameters if p not in ("self", "cls"))

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
        filtered = {k: v for k, v in args.items() if k in self._valid_params}
        result = self.fn(**filtered)
        if inspect.isawaitable(result):
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
        if not stripped:
            # 空行 = 段落边界，保留分隔符避免多段落合并为一行
            if desc_parts and desc_parts[-1] != "\n":
                desc_parts.append("\n")
        else:
            desc_parts.append(stripped)
    # 合并：同段落用空格，段落间用换行
    raw = " ".join(desc_parts)
    return raw.replace(" \n ", "\n\n").replace(" \n", "\n\n").replace("\n ", "\n\n").strip() if desc_parts else ""


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
            current_desc = [stripped[colon_pos + 1 :].strip()]
        elif current_name:
            current_desc.append(stripped)
    if current_name:
        result[current_name] = " ".join(current_desc).strip()
    return result


def _typing_eval_namespace() -> dict[str, Any]:
    """Namespace with common typing + builtin names for eval'ing string annotations."""
    import typing as _typing

    ns: dict[str, Any] = {}
    for name in dir(_typing):
        if not name.startswith("_"):
            ns[name] = getattr(_typing, name)
    for name in (str, int, float, bool, list, dict, set, tuple, bytes, type, type(None)):
        ns[name.__name__] = name
    # functools cached_property etc. are rarely in annotations, skip for perf
    return ns


def _resolve_string_hints(fn: Callable) -> dict[str, Any]:
    """Fallback: manually resolve string annotations when get_type_hints() fails."""
    hints: dict[str, Any] = {}
    # Restrict __builtins__ to prevent eval('__import__("os")...') etc.
    # Must come LAST to override __builtins__ from fn.__globals__.
    globalns: dict[str, Any] = {
        **getattr(fn, "__globals__", {}),
        **_typing_eval_namespace(),
        "__builtins__": {},
    }
    for pname, param in inspect.signature(fn).parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            continue
        if isinstance(ann, str):
            try:
                hints[pname] = eval(ann, globalns)
            except Exception:
                pass  # will default to {"type": "string"} in _resolve_type
    return hints


def _extract_parameters(fn: Callable) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        logger.warning("get_type_hints failed for %s, attempting manual resolution", getattr(fn, "__name__", fn))
        hints = _resolve_string_hints(fn)

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
            if isinstance(param.default, (bool, int, float, str)):
                prop["default"] = param.default
            elif param.default is None or isinstance(param.default, (list, dict)):
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
    args = getattr(py_type, "__args__", ())

    if origin is list:
        if args:
            return {"type": "array", "items": _resolve_type(args[0])}
        return {"type": "array"}
    if origin is dict:
        if args and len(args) >= 2:
            return {"type": "object", "additionalProperties": _resolve_type(args[1])}
        return {"type": "object"}
    if origin is set:
        result = {"type": "array", "uniqueItems": True}
        if args:
            result["items"] = _resolve_type(args[0])
        return result

    if origin is Literal:
        values = [a for a in args if a is not type(None)]
        if not values:
            return {"type": "string"}
        if all(isinstance(v, bool) for v in values):
            return {"type": "boolean", "enum": values}
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        return {"type": "string", "enum": [str(v) for v in values]}

    # Handle Union types: typing.Union and types.UnionType (X | Y in 3.10+)
    is_union = origin is Union or (hasattr(_bt, "UnionType") and isinstance(py_type, _bt.UnionType))
    if is_union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0])
        if non_none:
            return {"anyOf": [_resolve_type(t) for t in non_none]}
        return {"type": "string"}

    # Handle Enum types — extract values for JSON Schema enum constraint
    if isinstance(py_type, type) and issubclass(py_type, enum.Enum):
        values = [e.value for e in py_type]
        if issubclass(py_type, str):
            return {"type": "string", "enum": values}
        if issubclass(py_type, int):
            return {"type": "integer", "enum": values}
        return {"type": "string", "enum": values}

    if py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}

    return {"type": "string"}
