from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field


@dataclass
class A2uiBuilder:
    _lines: list[str] = field(default_factory=list)

    def add_text(self, text: str, usage_hint: str = "body", surface_id: str = "main") -> str:
        comp_id = f"text.{uuid.uuid4().hex[:6]}"
        root_id = f"root.{uuid.uuid4().hex[:6]}"
        self._lines.append(json.dumps({
            "surfaceUpdate": {
                "surfaceId": surface_id,
                "components": [
                    {"id": root_id, "component": {"Column": {"children": {"explicitList": [comp_id]}}}},
                    {"id": comp_id, "component": {"Text": {"text": {"literalString": text}, "usageHint": usage_hint}}},
                ],
            }
        }, ensure_ascii=False))
        self._lines.append(json.dumps({
            "beginRendering": {"surfaceId": surface_id, "root": root_id}
        }, ensure_ascii=False))
        return comp_id

    def add_column(self, child_ids: list[str], surface_id: str = "main") -> str:
        col_id = f"col.{uuid.uuid4().hex[:6]}"
        self._lines.append(json.dumps({
            "surfaceUpdate": {
                "surfaceId": surface_id,
                "components": [
                    {"id": col_id, "component": {"Column": {"children": {"explicitList": child_ids}}}},
                ],
            }
        }, ensure_ascii=False))
        return col_id

    def add_markdown(self, text: str, surface_id: str = "main") -> str:
        comp_id = f"md.{uuid.uuid4().hex[:6]}"
        root_id = f"root.{uuid.uuid4().hex[:6]}"
        self._lines.append(json.dumps({
            "surfaceUpdate": {
                "surfaceId": surface_id,
                "components": [
                    {"id": root_id, "component": {"Column": {"children": {"explicitList": [comp_id]}}}},
                    {"id": comp_id, "component": {"Markdown": {"text": {"literalString": text}}}},
                ],
            }
        }, ensure_ascii=False))
        self._lines.append(json.dumps({
            "beginRendering": {"surfaceId": surface_id, "root": root_id}
        }, ensure_ascii=False))
        return comp_id

    def to_jsonl(self) -> list[str]:
        return list(self._lines)

    def to_jsonl_string(self) -> str:
        return "\n".join(self._lines)
