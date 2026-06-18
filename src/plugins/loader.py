from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from src.agent.tooldef import ToolDef
from src.version import __version__
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("flyclaw.plugins")


class PluginManifest(BaseModel):
    id: str
    name: str = ""
    version: str = __version__
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    hooks: dict[str, str] = Field(default_factory=dict)


class HookResult(BaseModel):
    block: bool = False
    modified_args: Optional[dict] = None
    require_approval: bool = False
    approval_title: str = ""
    approval_description: str = ""


class PluginRecord(BaseModel):
    manifest: PluginManifest
    dir_path: str
    tools: list[Any] = Field(default_factory=list, exclude=True)
    hooks: dict[str, list[Callable]] = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _load_module(module_path: Path, module_name: str, allowed_dir: Optional[Path] = None):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        logger.error("Cannot load module: %s", module_path)
        return None
    if allowed_dir is not None:
        resolved = module_path.resolve()
        try:
            resolved.relative_to(allowed_dir.resolve())
        except ValueError:
            logger.error("Path traversal attempt: %s outside %s", module_path, allowed_dir)
            return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.error("Failed to execute module: %s", module_path, exc_info=True)
        return None
    sys.modules[module_name] = module
    return module


def _extract_tools(module) -> list[Any]:
    tools: list[Any] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, ToolDef):
            tools.append(attr)
    get_tools_fn = getattr(module, "get_tools", None)
    if callable(get_tools_fn):
        try:
            extra = get_tools_fn()
            if isinstance(extra, list):
                tools.extend(extra)
        except Exception:
            pass
    return tools


def _resolve_hook_ref(ref: str, module):
    if ":" in ref:
        attr_name = ref.split(":", 1)[1]
    else:
        attr_name = ref
    return getattr(module, attr_name, None)


def load_plugin(plugin_dir: Path) -> Optional[PluginRecord]:
    manifest_path = plugin_dir / "plugin.json"
    if not manifest_path.exists():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PluginManifest(**data)
    except Exception as e:
        logger.error("Failed to parse plugin manifest %s: %s", manifest_path, e)
        return None

    tools: list[Any] = []
    hooks: dict[str, list[Callable]] = {}

    for tools_file in manifest.tools:
        tools_path = plugin_dir / tools_file
        if not tools_path.exists():
            logger.warning(
                "Plugin %s: tools file not found: %s (expected at: %s)",
                manifest.id,
                tools_file,
                tools_path,
            )
            continue
        module = _load_module(tools_path, f"plugin_{manifest.id}_{tools_path.stem}", allowed_dir=plugin_dir)
        if module:
            extracted = _extract_tools(module)
            tools.extend(extracted)
            logger.info(
                "Plugin %s: loaded %d tools from %s",
                manifest.id,
                len(extracted),
                tools_file,
            )

    for hook_name, hook_ref in manifest.hooks.items():
        parts = hook_ref.split(":", 1)
        if len(parts) != 2:
            logger.warning("Plugin %s: invalid hook ref: %s", manifest.id, hook_ref)
            continue
        file_name, func_name = parts
        hook_path = plugin_dir / file_name
        if not hook_path.exists():
            logger.warning("Plugin %s: hook file not found: %s", manifest.id, file_name)
            continue
        module = _load_module(hook_path, f"plugin_{manifest.id}_{hook_path.stem}", allowed_dir=plugin_dir)
        if module:
            func = getattr(module, func_name, None)
            if func and callable(func):
                hooks.setdefault(hook_name, []).append(func)
                logger.info("Plugin %s: loaded hook %s", manifest.id, hook_name)

    return PluginRecord(
        manifest=manifest,
        dir_path=str(plugin_dir),
        tools=tools,
        hooks=hooks,
    )


def discover_plugins(extra_dirs: list[str] | None = None) -> list[PluginRecord]:
    records: list[PluginRecord] = []
    seen_ids: set[str] = set()

    scan_dirs: list[Path] = []
    bundled = Path(__file__).parent.parent.parent / "plugins"
    if bundled.exists():
        scan_dirs.append(bundled)

    for extra in extra_dirs or []:
        p = Path(extra).expanduser().resolve()
        if p.exists():
            scan_dirs.append(p)

    for base_dir in scan_dirs:
        for child in sorted(base_dir.iterdir()):
            if child.name.startswith((".", "_")):
                continue
            if not child.is_dir():
                continue
            if not (child / "plugin.json").exists():
                continue
            record = load_plugin(child)
            if record and record.manifest.id not in seen_ids:
                seen_ids.add(record.manifest.id)
                records.append(record)

    logger.info("Discovered %d plugins", len(records))
    return records
