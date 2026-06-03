"""Accessibility tree snapshot with element refs for browser automation.

Uses CDP Accessibility.getFullAXTree to get the accessibility tree,
then formats it into human-readable text with @eN element refs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("flyclaw.browser.snapshot")

_ref_counter: int = 0
_node_refs: dict[str, str] = {}  # node_id -> @eN


def _reset_refs():
    global _ref_counter, _node_refs
    _ref_counter = 0
    _node_refs = {}


def _get_ref(node_id: str) -> str:
    global _ref_counter
    if node_id in _node_refs:
        return _node_refs[node_id]
    _ref_counter += 1
    ref = f"@e{_ref_counter}"
    _node_refs[node_id] = ref
    return ref


_INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "searchbox",
    "combobox",
    "checkbox",
    "radio",
    "slider",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "switch",
    "option",
    "treeitem",
    "gridcell",
}

_SKIP_ROLES = {"generic", "none", "presentation", "StaticText"}


def build_page_info(url: str, title: str, tree_text: str, element_count: int) -> str:
    header = f"Page: {title}\nURL: {url}\n"
    if element_count > 0:
        header += f"Elements: {element_count}\n"
    header += "---\n"
    return header + tree_text


async def get_snapshot(page, compact: bool = True) -> dict:
    """Take an accessibility tree snapshot via CDP.

    Returns dict with keys: url, title, snapshot, element_count
    """
    _reset_refs()

    url = page.url
    title = await page.title()

    cdp = None
    try:
        cdp = await page.context.new_cdp_session(page)
        result = await cdp.send("Accessibility.getFullAXTree")
    except Exception as e:
        logger.warning("CDP accessibility snapshot failed: %s", e)
        return {
            "url": url,
            "title": title,
            "snapshot": "(snapshot unavailable)",
            "element_count": 0,
        }
    finally:
        if cdp:
            try:
                await cdp.detach()
            except Exception:
                logger.debug("CDP detach failed", exc_info=True)

    nodes = result.get("nodes", [])
    if not nodes:
        return {
            "url": url,
            "title": title,
            "snapshot": "(empty page)",
            "element_count": 0,
        }

    tree_text = _format_cdp_tree(nodes, compact)

    if len(tree_text) > 20000:
        tree_text = tree_text[:20000] + "\n... (truncated)"

    return {
        "url": url,
        "title": title,
        "snapshot": tree_text,
        "element_count": _ref_counter,
    }


def _format_cdp_tree(nodes: list[dict], compact: bool) -> str:
    """Format CDP AX tree nodes into text with refs."""
    # Build lookup by backendDOMNodeId for hierarchy
    by_id: dict[str, dict] = {}
    for n in nodes:
        nid = n.get("backendDOMNodeId")
        if nid is not None:
            by_id[str(nid)] = n

    lines: list[str] = []
    for node in nodes:
        role_obj = node.get("role", {})
        role = role_obj.get("value", "") if isinstance(role_obj, dict) else str(role_obj)
        name_obj = node.get("name", {})
        name = name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj)
        value_obj = node.get("value", {})
        value = value_obj.get("value", "") if isinstance(value_obj, dict) else str(value_obj)

        if not role or (compact and role in _SKIP_ROLES):
            continue

        node_id = node.get("nodeId", str(id(node)))
        interactive = role in _INTERACTIVE_ROLES or role in ("heading", "textbox", "searchbox", "combobox")
        ref = _get_ref(node_id) if interactive else ""

        desc = ""
        if ref:
            desc += f"{ref} "
        desc += role

        if name:
            desc += f' "{name[:200]}"'
        if value and value != name:
            desc += f" [value: {value[:100]}]"

        # State info
        states = []
        props = node.get("properties", [])
        for p in props:
            pname = p.get("name", "")
            if pname == "disabled" and p.get("value", {}).get("value"):
                states.append("disabled")
            elif pname == "focused" and p.get("value", {}).get("value"):
                states.append("focused")
            elif pname == "checked":
                v = p.get("value", {}).get("value")
                if v and v != "false":
                    states.append("checked")
            elif pname == "expanded":
                v = p.get("value", {}).get("value")
                if v is not None:
                    states.append("expanded" if v else "collapsed")
            elif pname == "selected" and p.get("value", {}).get("value"):
                states.append("selected")

        if states:
            desc += f" [{', '.join(states)}]"

        lines.append(desc)

    return "\n".join(lines)


def get_interactive_nodes(nodes: list[dict]) -> list[dict]:
    """Extract interactive nodes in order (for _ref_to_locator matching)."""
    result = []
    for node in nodes:
        role_obj = node.get("role", {})
        role = role_obj.get("value", "") if isinstance(role_obj, dict) else str(role_obj)
        if role in _INTERACTIVE_ROLES or role in ("heading", "textbox", "searchbox", "combobox"):
            result.append(node)
    return result
