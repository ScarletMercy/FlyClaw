"""Tests for src/tools/browser/snapshot.py — accessibility tree parsing, element refs, page info."""

import pytest

from src.tools.browser.snapshot import (
    build_page_info,
    _format_cdp_tree,
    _reset_refs,
    _get_ref,
    get_interactive_nodes,
    _INTERACTIVE_ROLES,
    _SKIP_ROLES,
)


@pytest.fixture(autouse=True)
def reset_refs():
    _reset_refs()
    yield
    _reset_refs()


# ── _get_ref ───────────────────────────────────────────────


class TestGetRef:
    def test_first_ref_is_e1(self):
        assert _get_ref("node_1") == "@e1"

    def test_second_ref_is_e2(self):
        _get_ref("node_1")
        assert _get_ref("node_2") == "@e2"

    def test_same_node_returns_same_ref(self):
        r1 = _get_ref("node_1")
        r2 = _get_ref("node_1")
        assert r1 == r2 == "@e1"

    def test_different_nodes_increment(self):
        refs = [_get_ref(f"node_{i}") for i in range(5)]
        assert refs == ["@e1", "@e2", "@e3", "@e4", "@e5"]


# ── build_page_info ────────────────────────────────────────


class TestBuildPageInfo:
    def test_basic(self):
        result = build_page_info("https://example.com", "Example", "tree text", 3)
        assert "Page: Example" in result
        assert "URL: https://example.com" in result
        assert "Elements: 3" in result
        assert "tree text" in result

    def test_zero_elements(self):
        result = build_page_info("https://x.com", "X", "(empty)", 0)
        assert "Elements" not in result

    def test_separator(self):
        result = build_page_info("u", "t", "body", 1)
        assert "---" in result


# ── _format_cdp_tree ───────────────────────────────────────


def _make_node(node_id="n1", role="button", name="Click me", value="", props=None):
    """Build a CDP AX tree node dict."""
    node = {
        "nodeId": node_id,
        "role": {"type": "role", "value": role},
        "name": {"type": "computedString", "value": name},
    }
    if value:
        node["value"] = {"type": "string", "value": value}
    if props:
        node["properties"] = props
    return node


class TestFormatCdpTree:
    def test_interactive_element_gets_ref(self):
        nodes = [_make_node(role="button", name="Submit")]
        result = _format_cdp_tree(nodes, compact=True)
        assert "@e1" in result
        assert "button" in result
        assert "Submit" in result

    def test_link_gets_ref(self):
        nodes = [_make_node(role="link", name="Home")]
        result = _format_cdp_tree(nodes, compact=True)
        assert "@e1" in result
        assert "link" in result

    def test_textbox_gets_ref(self):
        nodes = [_make_node(role="textbox", name="Search")]
        result = _format_cdp_tree(nodes, compact=True)
        assert "@e1" in result

    def test_generic_skipped_in_compact(self):
        nodes = [_make_node(role="generic", name="")]
        result = _format_cdp_tree(nodes, compact=True)
        assert result == ""

    def test_generic_shown_in_full(self):
        nodes = [_make_node(role="generic", name="")]
        result = _format_cdp_tree(nodes, compact=False)
        assert "generic" in result

    def test_none_role_skipped(self):
        nodes = [_make_node(role="", name="")]
        # Override role to empty
        nodes[0]["role"] = {"type": "role", "value": ""}
        result = _format_cdp_tree(nodes, compact=True)
        assert result == ""

    def test_disabled_state(self):
        nodes = [
            _make_node(
                role="button",
                name="Submit",
                props=[{"name": "disabled", "value": {"type": "boolean", "value": True}}],
            )
        ]
        result = _format_cdp_tree(nodes, compact=True)
        assert "disabled" in result

    def test_focused_state(self):
        nodes = [
            _make_node(
                role="textbox",
                name="Input",
                props=[{"name": "focused", "value": {"type": "boolean", "value": True}}],
            )
        ]
        result = _format_cdp_tree(nodes, compact=True)
        assert "focused" in result

    def test_checked_state(self):
        nodes = [
            _make_node(
                role="checkbox",
                name="Agree",
                props=[{"name": "checked", "value": {"type": "boolean", "value": True}}],
            )
        ]
        result = _format_cdp_tree(nodes, compact=True)
        assert "checked" in result

    def test_expanded_state(self):
        nodes = [
            _make_node(
                role="button",
                name="Menu",
                props=[{"name": "expanded", "value": {"type": "boolean", "value": True}}],
            )
        ]
        result = _format_cdp_tree(nodes, compact=True)
        assert "expanded" in result

    def test_collapsed_state(self):
        nodes = [
            _make_node(
                role="button",
                name="Menu",
                props=[{"name": "expanded", "value": {"type": "boolean", "value": False}}],
            )
        ]
        result = _format_cdp_tree(nodes, compact=True)
        assert "collapsed" in result

    def test_value_shown(self):
        nodes = [_make_node(role="textbox", name="Name", value="John")]
        result = _format_cdp_tree(nodes, compact=True)
        assert "John" in result

    def test_name_truncated_at_200(self):
        long_name = "x" * 300
        nodes = [_make_node(role="button", name=long_name)]
        result = _format_cdp_tree(nodes, compact=True)
        assert len(result) < len(long_name)

    def test_multiple_interactive_elements(self):
        nodes = [
            _make_node(node_id="n1", role="link", name="Home"),
            _make_node(node_id="n2", role="button", name="Submit"),
            _make_node(node_id="n3", role="textbox", name="Search"),
        ]
        result = _format_cdp_tree(nodes, compact=True)
        assert "@e1" in result
        assert "@e2" in result
        assert "@e3" in result


# ── get_interactive_nodes ──────────────────────────────────


class TestGetInteractiveNodes:
    def test_filters_interactive(self):
        nodes = [
            _make_node(node_id="n1", role="button", name="Click"),
            _make_node(node_id="n2", role="generic", name=""),
            _make_node(node_id="n3", role="link", name="Home"),
        ]
        result = get_interactive_nodes(nodes)
        assert len(result) == 2
        roles = [n["role"]["value"] for n in result]
        assert "button" in roles
        assert "link" in roles

    def test_empty_list(self):
        assert get_interactive_nodes([]) == []

    def test_heading_included(self):
        nodes = [_make_node(role="heading", name="Title")]
        result = get_interactive_nodes(nodes)
        assert len(result) == 1


# ── _ref_to_locator (from tools.py) ────────────────────────


class TestRefToLocator:
    def test_invalid_ref_no_at(self):
        from src.tools.browser.tools import _ref_to_locator

        result = asyncio_run(_ref_to_locator(None, "e1"))
        assert result is None

    def test_invalid_ref_no_e(self):
        from src.tools.browser.tools import _ref_to_locator

        result = asyncio_run(_ref_to_locator(None, "@x1"))
        assert result is None

    def test_invalid_ref_non_numeric(self):
        from src.tools.browser.tools import _ref_to_locator

        result = asyncio_run(_ref_to_locator(None, "@eabc"))
        assert result is None


def asyncio_run(coro):
    """Run an async coroutine synchronously for testing."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # If already in an async context, create a new loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
