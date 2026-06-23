"""_build_media_text characterization 测试 —— qq 渠道 video/image marker 文本构造。

锁住原 qq.py:883-902 的行为(检查项 9 把 4 个重复 append 循环 dedup、提示文本设置与
marker 追加分离后,行为必须不变)。关键顺序依赖: video 块先跑(可能设 text),image 块后跑
(读已被 video 块修改的 text 判断是否设提示)。
"""

from src.channels.qq import _build_media_text


class TestBuildMediaText:
    def test_video_only_empty_text_sets_prompt(self):
        # 纯视频 + text 空 → 设"请描述这个视频" + video marker
        assert _build_media_text("", [], ["v1.mp4"], True) == '请描述这个视频\n[video_url: "v1.mp4"]'

    def test_video_only_existing_text_kept(self):
        # 纯视频 + text 非空 → 保留 text + video marker(不设提示)
        assert _build_media_text("看这个", [], ["v1.mp4"], True) == '看这个\n[video_url: "v1.mp4"]'

    def test_image_only_empty_text_sets_prompt(self):
        # 纯图(image+url 不设 has_media) + text 空 → 设"请描述这张图片" + image marker
        assert _build_media_text("", ["i1.png"], [], False) == '请描述这张图片\n[image_url: "i1.png"]'

    def test_multiple_images_empty_text_sets_count_prompt(self):
        # 多图 + text 空 → "请描述这 N 张图片" + markers
        out = _build_media_text("", ["i1.png", "i2.png"], [], False)
        assert out == '请描述这 2 张图片\n[image_url: "i1.png"]\n[image_url: "i2.png"]'

    def test_image_only_existing_text_kept(self):
        # 纯图 + text 非空 → 保留 text + image marker(image 块走 elif)
        assert _build_media_text("看", ["i1.png"], [], False) == '看\n[image_url: "i1.png"]'

    def test_video_and_image_video_prompt_used(self):
        # video+image + text 空 → video 块设"请描述这个视频"+video marker;image 块 text 已非空只追加 marker
        out = _build_media_text("", ["i1.png"], ["v1.mp4"], True)
        assert out == '请描述这个视频\n[video_url: "v1.mp4"]\n[image_url: "i1.png"]'

    def test_media_no_url_empty_text_gets_generic(self):
        # 无 video/image url 但 has_media(如纯音频/文件) + text 空 → "[收到媒体消息]"
        assert _build_media_text("", [], [], True) == "[收到媒体消息]"

    def test_no_media_empty_text_unchanged(self):
        assert _build_media_text("", [], [], False) == ""

    def test_no_media_text_preserved(self):
        assert _build_media_text("hi", [], [], False) == "hi"

    def test_text_with_image_appends_marker(self):
        # 已有 text + 图 → 保留 + image marker
        assert _build_media_text("嗨", ["i1.png"], [], False) == '嗨\n[image_url: "i1.png"]'
