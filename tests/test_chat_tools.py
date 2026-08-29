"""Tests for src/tools/chat_tools.py — media type inference, context management."""

from src.tools.chat_tools import (
    _infer_media_type,
    _get_context,
    set_current_chat_context,
    _current_channel_type,
    _current_chat_id,
)


# ── _infer_media_type ──────────────────────────────────────


class TestInferMediaType:
    def test_png_image(self):
        assert _infer_media_type("photo.png") == "image"

    def test_jpg_image(self):
        assert _infer_media_type("photo.jpg") == "image"

    def test_jpeg_image(self):
        assert _infer_media_type("photo.jpeg") == "image"

    def test_gif_image(self):
        assert _infer_media_type("anim.gif") == "image"

    def test_webp_image(self):
        assert _infer_media_type("photo.webp") == "image"

    def test_mp3_audio(self):
        assert _infer_media_type("song.mp3") == "audio"

    def test_wav_audio(self):
        assert _infer_media_type("voice.wav") == "audio"

    def test_silk_audio(self):
        assert _infer_media_type("voice.silk") == "audio"

    def test_txt_file(self):
        assert _infer_media_type("doc.txt") == "file"

    def test_py_file(self):
        assert _infer_media_type("script.py") == "file"

    def test_no_extension(self):
        assert _infer_media_type("README") == "file"

    def test_url_with_image(self):
        assert _infer_media_type("https://example.com/photo.png") == "image"

    def test_url_with_audio(self):
        assert _infer_media_type("https://example.com/song.mp3") == "audio"

    def test_url_with_query_string(self):
        assert _infer_media_type("https://cdn.example.com/img.png?v=123&x=456") == "image"

    def test_case_insensitive(self):
        assert _infer_media_type("Photo.PNG") == "image"

    def test_url_with_file(self):
        assert _infer_media_type("https://example.com/data.json") == "file"


# ── context management ─────────────────────────────────────


class TestContextManagement:
    def test_set_and_get(self):
        set_current_chat_context("qq", "chat123")
        channel, chat_id = _get_context()
        assert channel == "qq"
        assert chat_id == "chat123"
        # Reset
        set_current_chat_context("", "")

    def test_default_empty(self):
        _current_channel_type.set("")
        _current_chat_id.set("")
        channel, chat_id = _get_context()
        assert channel == ""
        assert chat_id == ""
