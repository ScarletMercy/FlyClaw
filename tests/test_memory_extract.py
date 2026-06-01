"""Tests for auto_extract_memory and _extract_matched_clause."""

import pytest

from src.tools.memory_tools import auto_extract_memory, _extract_matched_clause


# ---------------------------------------------------------------------------
# _extract_matched_clause unit tests
# ---------------------------------------------------------------------------


class TestExtractMatchedClause:
    """Verify _extract_matched_clause extracts the correct clause."""

    def test_empty_text(self):
        assert _extract_matched_clause("", 0, 0) == ""

    def test_match_at_beginning(self):
        # "我叫张三" — match "我叫" at 0-2
        result = _extract_matched_clause("我叫张三", 0, 2)
        assert "我叫张三" in result
        assert result == "我叫张三"

    def test_match_at_end(self):
        # "你好，我叫张三" — match "我叫" at 3-5
        result = _extract_matched_clause("你好，我叫张三", 3, 5)
        assert result == "我叫张三"

    def test_match_in_middle(self):
        text = "帮我写代码，记住我喜欢Python，谢谢"
        ms = text.index("记住")
        me = ms + len("记住")
        result = _extract_matched_clause(text, ms, me)
        assert "我喜欢Python" in result
        assert "帮我写代码" not in result
        assert "谢谢" not in result

    def test_multiline_text(self):
        text = "第一行\n帮我写代码，记住我喜欢Python\n第三行"
        result = _extract_matched_clause(text, text.index("记住"), text.index("记住") + 2)
        assert "我喜欢Python" in result
        assert "第一行" not in result
        assert "第三行" not in result

    def test_url_not_broken_by_period(self):
        # Email with dots — the period in "example.com" must not be treated as separator
        text = "我的邮箱是test@example.com，帮我发邮件"
        result = _extract_matched_clause(text, text.index("邮箱"), text.index("邮箱") + 2)
        assert "test@example.com" in result

    def test_phone_number(self):
        text = "顺便说一下，我的手机号是13800138000，记得存一下"
        result = _extract_matched_clause(text, text.index("手机号"), text.index("手机号") + 3)
        assert "13800138000" in result
        assert "顺便" not in result
        assert "记得存" not in result

    def test_no_separators_full_text(self):
        # No separators at all — returns entire text
        text = "我叫张三今天天气不错"
        result = _extract_matched_clause(text, 0, 2)
        assert result == text

    def test_result_is_substring_of_input(self):
        """The extracted clause must always be a substring of the original text."""
        cases = [
            ("帮我写个排序算法，记住我喜欢用Python", 7, 9),
            ("你好，对了以后请用中文回复我", 4, 6),
            ("我的邮箱是 test@example.com，帮我发个邮件", 0, 2),
        ]
        for text, ms, me in cases:
            result = _extract_matched_clause(text, ms, me)
            assert result in text, f"'{result}' not a substring of '{text}'"


# ---------------------------------------------------------------------------
# auto_extract_memory integration tests
# ---------------------------------------------------------------------------


class TestAutoExtractMemory:
    """Verify auto_extract_memory returns (content, category) correctly."""

    def test_empty_input(self):
        assert auto_extract_memory("", "reply") is None

    def test_short_input(self):
        assert auto_extract_memory("ok", "reply") is None

    def test_no_match(self):
        assert auto_extract_memory("今天天气怎么样？", "晴天") is None

    def test_preference_extraction(self):
        text = "帮我写个排序算法，记住我喜欢用Python"
        result = auto_extract_memory(text, "")
        assert result is not None
        content, category = result
        assert category == "preference"
        assert "我喜欢用Python" in content
        assert "帮我写个排序算法" not in content

    def test_identity_extraction(self):
        text = "你好，我叫韩飞摩，很高兴认识你"
        result = auto_extract_memory(text, "")
        assert result is not None
        content, category = result
        assert category == "identity"
        assert "我叫韩飞摩" in content
        assert "你好" not in content
        assert "很高兴认识你" not in content

    def test_contact_email(self):
        text = "我的邮箱是test@example.com，帮我发个邮件"
        result = auto_extract_memory(text, "")
        assert result is not None
        content, category = result
        assert category == "contact"
        assert "test@example.com" in content

    def test_contact_phone(self):
        text = "顺便说一下，我的手机号是13800138000，记得存一下"
        result = auto_extract_memory(text, "")
        assert result is not None
        content, category = result
        assert category == "contact"
        assert "13800138000" in content

    def test_project_extraction(self):
        text = "我的项目用了FastAPI和Vue，帮我看看路由配置"
        result = auto_extract_memory(text, "")
        assert result is not None
        content, category = result
        assert category == "project"
        assert "FastAPI" in content

    def test_service_extraction(self):
        text = "API地址是https://api.example.com/v2，帮我测试一下"
        result = auto_extract_memory(text, "")
        assert result is not None
        content, category = result
        assert category == "service"
        assert "api.example.com" in content

    def test_result_is_substring_of_input(self):
        """Extracted content must be a substring of the original input — no fabrication."""
        cases = [
            "帮我写个排序算法，记住我喜欢用Python",
            "你好，对了以后请用中文回复我",
            "我的邮箱是test@example.com，帮我发个邮件",
            "顺便说一下，我的手机号是13800138000，记得存一下",
            "你好，别用Java",
        ]
        for text in cases:
            result = auto_extract_memory(text, "")
            if result:
                content, _ = result
                assert content in text, f"'{content}' not a substring of '{text}'"

    def test_return_type(self):
        result = auto_extract_memory("我喜欢Python", "")
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_never_returns_empty_content(self):
        """即使提取结果为空字符串，也不应返回 ('', category)。"""
        # 构造一个极端场景：模式匹配了，但子句切分后为空
        # _extract_matched_clause 对 ",记住," 的匹配结果在 rstrip("，,") 后可能为空
        # auto_extract_memory 应返回 None 而非空字符串
        from src.tools.memory_tools import _extract_matched_clause

        # 直测 _extract_matched_clause 的边界
        assert _extract_matched_clause("", 0, 0) == ""
        # auto_extract_memory 绝不返回空 content
        import re

        for cat, pat in [
            ("preference", re.compile(r"记住")),
        ]:
            m = pat.search("，，，记住，，，")
            if m:
                clause = _extract_matched_clause("，，，记住，，，", m.start(), m.end())
                # 无论 clause 是否为空，auto_extract_memory 不应返回空 content
                result = auto_extract_memory("，，，记住，，，", "")
                if result:
                    content, _ = result
                    assert content, "auto_extract_memory returned empty content!"
