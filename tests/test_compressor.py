"""ContextCompressor 单元测试 —— 聚焦 list content 的摘要适配。"""

from src.compressor.compressor import _content_to_text


class TestContentToText:
    def test_str_passthrough(self):
        assert _content_to_text("hello") == "hello"

    def test_empty_str(self):
        assert _content_to_text("") == ""

    def test_list_extracts_text_block(self):
        content = [
            {"type": "text", "text": "看这张图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
        ]
        assert _content_to_text(content) == "看这张图"

    def test_list_skips_all_media_blocks(self):
        # 全媒体 block(无 text)→ 空串,base64 一字节都不进摘要
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        ]
        assert _content_to_text(content) == ""

    def test_list_multiple_text_blocks_joined(self):
        content = [
            {"type": "text", "text": "Screenshot saved to: /x.png"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "text", "text": "尾随文本"},
        ]
        assert _content_to_text(content) == "Screenshot saved to: /x.png\n尾随文本"

    def test_empty_list(self):
        assert _content_to_text([]) == ""

    def test_text_block_missing_text_key_skipped(self):
        # 防御:type=text 但无 text 键 → 跳过,不 KeyError
        assert _content_to_text([{"type": "text"}, {"type": "text", "text": "ok"}]) == "ok"

    def test_non_str_non_list_fallback(self):
        # 兜底:理论不会出现,但别崩
        assert _content_to_text(None) == ""  # type: ignore


class TestFormatTurnsWithMediaList:
    """_format_turns 对固化 list content 的处理 —— 锁住 R1 不回归。

    固化后带图消息是 list(含 base64 image_url block)。摘要输入必须提取 text、
    不能把 base64 repr 喂给摘要 LLM。
    """

    def _cc(self):
        from src.compressor.compressor import ContextCompressor
        from src.config import CompressionConfig

        return ContextCompressor(CompressionConfig())

    def test_user_image_list_no_base64_in_output(self):
        cc = self._cc()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这张图"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"}},
                ],
            }
        ]
        out = cc._format_turns(msgs)
        assert "看这张图" in out
        assert "base64" not in out
        assert "data:" not in out
        assert "image_url" not in out

    def test_tool_screenshot_list_extracts_text_only(self):
        cc = self._cc()
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": [
                    {"type": "text", "text": "Screenshot saved to: /tmp/x.png"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                ],
            }
        ]
        out = cc._format_turns(msgs)
        assert "Screenshot saved to" in out
        assert "base64" not in out

    def test_str_content_unchanged(self):
        # 回归保护:str content 行为不变(旧机制标记 str 不会变)
        cc = self._cc()
        msgs = [{"role": "user", "content": "看这张 [image_url: /a.png]"}]
        out = cc._format_turns(msgs)
        assert "看这张 [image_url: /a.png]" in out


class TestCompactWithMediaList:
    """_compact(静态截断 / LLM 失败降级路径)对固化 list content 的处理 —— 锁住 M1。

    compression.enabled=False 或 LLM 摘要失败时走 _compact,它和 _format_turns 是
    平行的摘要路径,同样不能让 base64 泄漏进 summary。
    """

    def _cc(self):
        from src.compressor.compressor import ContextCompressor
        from src.config import CompressionConfig

        return ContextCompressor(CompressionConfig(enabled=False))

    def test_compact_user_image_list_no_base64(self):
        cc = self._cc()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这张图"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"}},
                ],
            }
        ]
        # 撑足够 tail,让带图消息滚进 pruned(被摘要的部分)
        msgs += [{"role": "assistant", "content": f"reply {i} " * 50} for i in range(8)]
        msgs += [{"role": "user", "content": "final question"}]
        result = cc._compact(msgs, context_window_tokens=5)
        joined = "".join(m.get("content", "") for m in result if isinstance(m.get("content"), str))
        assert "base64" not in joined, f"base64 泄漏进静态摘要: {joined[:200]}"
        assert "data:" not in joined
        assert "看这张图" in joined  # text part 被正确提取
