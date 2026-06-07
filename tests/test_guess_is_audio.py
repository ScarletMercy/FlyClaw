"""Tests for _guess_is_audio early rejection in media understanding."""

from src.tools.media_understanding_tools import _guess_is_audio


class TestGuessIsAudio:
    # --- data: URLs ---

    def test_data_url_audio_mime(self):
        assert _guess_is_audio("data:audio/mpeg;base64,AAAA") is True

    def test_data_url_wav_mime(self):
        assert _guess_is_audio("data:audio/wav;base64,AAAA") is True

    def test_data_url_image_mime(self):
        assert _guess_is_audio("data:image/png;base64,AAAA") is False

    def test_data_url_malformed(self):
        assert _guess_is_audio("data:") is False

    # --- Local files ---

    def test_local_mp3(self):
        assert _guess_is_audio("/path/to/song.mp3") is True

    def test_local_wav(self):
        assert _guess_is_audio("recording.wav") is True

    def test_local_silk(self):
        assert _guess_is_audio("voice.silk") is True

    def test_local_txt(self):
        assert _guess_is_audio("document.txt") is False

    def test_local_no_extension(self):
        assert _guess_is_audio("README") is False

    def test_local_png(self):
        assert _guess_is_audio("photo.png") is False

    # --- Remote URLs ---

    def test_remote_mp3(self):
        assert _guess_is_audio("https://example.com/audio.mp3") is True

    def test_remote_ogg_with_query(self):
        assert _guess_is_audio("https://cdn.example.com/track.ogg?v=1") is True

    def test_remote_jpg(self):
        assert _guess_is_audio("https://example.com/photo.jpg") is False

    def test_remote_no_extension(self):
        assert _guess_is_audio("https://example.com/api/audio") is False

    # --- Windows drive letter paths ---

    def test_windows_path_mp3(self):
        assert _guess_is_audio("C:\\Users\\test\\song.mp3") is True
