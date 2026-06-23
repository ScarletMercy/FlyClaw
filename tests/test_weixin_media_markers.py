"""微信渠道媒体标记注入（对齐 QQ）的单元测试。"""

from src.channels.weixin import _attach_media_markers


class TestAttachMediaMarkers:
    def test_image_appended_text_preserved(self):
        assert _attach_media_markers("看这张", ["/a/b.png"], ["image/jpeg"]) == ('看这张\n[image_url: "/a/b.png"]')

    def test_video_marker(self):
        assert _attach_media_markers("v", ["/a/b.mp4"], ["video/mp4"]) == 'v\n[video_url: "/a/b.mp4"]'

    def test_audio_skipped(self):
        # 多模态不支持音频块 → 跳过，text 不变
        assert _attach_media_markers("hi", ["/a/b.silk"], ["audio/silk"]) == "hi"

    def test_mixed_skips_audio_keeps_order(self):
        out = _attach_media_markers("x", ["/a.png", "/b.silk", "/c.mp4"], ["image/jpeg", "audio/silk", "video/mp4"])
        assert out == 'x\n[image_url: "/a.png"]\n[video_url: "/c.mp4"]'

    def test_empty_text_gets_default_prompt(self):
        assert _attach_media_markers("", ["/a.png"], ["image/png"]) == ('请查看以下图片/视频\n[image_url: "/a.png"]')

    def test_whitespace_text_gets_default_prompt(self):
        assert _attach_media_markers("   ", ["/a.png"], ["image/png"]) == ('请查看以下图片/视频\n[image_url: "/a.png"]')

    def test_path_with_bracket_is_quoted(self):
        # 路径含 ]（如下载重命名 报告[1].png）时用引号包裹，避免被 inject 的正则截断
        assert _attach_media_markers("看", ["/a/报告[1].png"], ["image/png"]) == ('看\n[image_url: "/a/报告[1].png"]')

    def test_no_media_unchanged(self):
        assert _attach_media_markers("hello", [], []) == "hello"

    def test_no_media_empty_unchanged(self):
        assert _attach_media_markers("", [], []) == ""
