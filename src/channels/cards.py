from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger("myclaw.cards")

_TEMPLATE_COLORS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "orange": "orange",
    "purple": "purple",
    "indigo": "indigo",
    "wathet": "wathet",
    "turquoise": "turquoise",
    "yellow": "yellow",
    "grey": "grey",
}


class InteractiveCardBuilder:
    def __init__(self):
        self._header: Optional[dict] = None
        self._elements: list[dict] = []
        self._note: Optional[str] = None

    def header(self, title: str, template: str = "blue") -> "InteractiveCardBuilder":
        color = _TEMPLATE_COLORS.get(template, "blue")
        self._header = {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        }
        return self

    def add_markdown(self, content: str) -> "InteractiveCardBuilder":
        self._elements.append({"tag": "markdown", "content": content})
        return self

    def add_button(
        self,
        text: str,
        value: str,
        style: str = "primary",
        url: str = "",
    ) -> "InteractiveCardBuilder":
        btn_type_map = {"primary": "primary", "danger": "danger", "default": "default"}
        btn = {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": btn_type_map.get(style, "default"),
            "value": value,
        }
        if url:
            btn["url"] = url
        self._elements.append({"tag": "action", "actions": [btn]})
        return self

    def add_action_row(self, buttons: list[dict]) -> "InteractiveCardBuilder":
        actions = []
        for b in buttons:
            btn = {
                "tag": "button",
                "text": {"tag": "plain_text", "content": b.get("text", "")},
                "type": b.get("type", "default"),
                "value": b.get("value", ""),
            }
            if b.get("url"):
                btn["url"] = b["url"]
            actions.append(btn)
        self._elements.append({"tag": "action", "actions": actions})
        return self

    def add_divider(self) -> "InteractiveCardBuilder":
        self._elements.append({"tag": "hr"})
        return self

    def add_note(self, text: str) -> "InteractiveCardBuilder":
        self._note = text
        return self

    def build(self) -> dict:
        card: dict[str, Any] = {
            "schema": "2.0",
            "config": {"streaming_mode": False},
        }
        if self._header:
            card["header"] = self._header
        elements = list(self._elements)
        if self._note:
            elements.append({"tag": "hr"})
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"<font color='grey'>{self._note}</font>",
                }
            )
        card["body"] = {"elements": elements}
        return card

    def build_content(self) -> str:
        return json.dumps(self.build(), ensure_ascii=False)


class CardCallbackRegistry:
    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, action_id: str, handler: Callable) -> None:
        self._handlers[action_id] = handler

    def resolve(self, action_id: str) -> Optional[Callable]:
        return self._handlers.get(action_id)

    def list_callbacks(self) -> list[str]:
        return list(self._handlers.keys())


_registry: Optional[CardCallbackRegistry] = None


def get_card_callback_registry() -> CardCallbackRegistry:
    global _registry
    if _registry is None:
        _registry = CardCallbackRegistry()
    return _registry
