"""build_media_marker 单元测试 —— qq/weixin 渠道共用的媒体标记格式(单一 owner)。

格式契约: [{kind}_url: "{path}"],kind ∈ {image, video}。path 永远双引号包裹
(含 ] 等特殊字符时防 inject 正则截断)。消费者是 LLM(从 text 提取路径调 describe_media),
无代码解析端,故格式需稳定。提取此函数前格式在 qq.py(4 处)/weixin.py(2 处)复制。
"""

import pytest

from src.channels.base import build_media_marker


class TestBuildMediaMarker:
    def test_image(self):
        assert build_media_marker("image", "/a/b.png") == '[image_url: "/a/b.png"]'

    def test_video(self):
        assert build_media_marker("video", "/a/b.mp4") == '[video_url: "/a/b.mp4"]'

    def test_url_path(self):
        # qq 渠道传的是远程 URL
        assert build_media_marker("video", "https://x/v.mp4") == '[video_url: "https://x/v.mp4"]'

    def test_path_with_bracket_is_quoted(self):
        # 含 ] 的路径(如 报告[1].png)原样包在引号里,防 inject 正则截断
        assert build_media_marker("image", "/a/报告[1].png") == '[image_url: "/a/报告[1].png"]'

    def test_unsupported_kind_raises(self):
        # audio 走别的路径(qq 转录/weixin 跳过),不该构 marker
        with pytest.raises(ValueError):
            build_media_marker("audio", "/a.mp3")
