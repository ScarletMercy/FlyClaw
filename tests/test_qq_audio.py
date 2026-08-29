"""Tests for src/channels/qq_audio.py — audio format detection, SILK detection, WAV conversion."""

import struct
import wave


from src.channels.qq_audio import guess_audio_ext, looks_like_silk, raw_pcm_to_wav


# ── guess_audio_ext ────────────────────────────────────────


class TestGuessAudioExt:
    def test_silk_v3_header(self):
        data = b"#!SILK_V3" + b"\x00" * 100
        assert guess_audio_ext(data) == ".silk"

    def test_silk_short_header(self):
        data = b"#!SILK" + b"\x00" * 100
        assert guess_audio_ext(data) == ".silk"

    def test_silk_02_prefix(self):
        data = b"\x02!" + b"\x00" * 100
        assert guess_audio_ext(data) == ".silk"

    def test_wav_header(self):
        data = b"RIFF" + struct.pack("<I", 100) + b"WAVE" + b"\x00" * 100
        assert guess_audio_ext(data) == ".wav"

    def test_flac_header(self):
        data = b"fLaC" + b"\x00" * 100
        assert guess_audio_ext(data) == ".flac"

    def test_mp3_header_fffb(self):
        data = b"\xff\xfb" + b"\x00" * 100
        assert guess_audio_ext(data) == ".mp3"

    def test_mp3_header_fff3(self):
        data = b"\xff\xf3" + b"\x00" * 100
        assert guess_audio_ext(data) == ".mp3"

    def test_ogg_header(self):
        data = b"\x4f\x67\x67\x53" + b"\x00" * 100
        assert guess_audio_ext(data) == ".ogg"

    def test_amr_default(self):
        data = b"\x00" * 20
        assert guess_audio_ext(data) == ".amr"

    def test_amr_header_1(self):
        data = b"\x00\x00\x00\x20" + b"\x00" * 100
        assert guess_audio_ext(data) == ".amr"

    def test_amr_header_2(self):
        data = b"\x00\x00\x00\x1c" + b"\x00" * 100
        assert guess_audio_ext(data) == ".amr"


# ── looks_like_silk ────────────────────────────────────────


class TestLooksLikeSilk:
    def test_silk_v3(self):
        assert looks_like_silk(b"#!SILK_V3" + b"\x00" * 50) is True

    def test_silk_short(self):
        assert looks_like_silk(b"#!SILK" + b"\x00" * 50) is True

    def test_silk_02(self):
        assert looks_like_silk(b"\x02!" + b"\x00" * 50) is True

    def test_wav_not_silk(self):
        assert looks_like_silk(b"RIFF" + b"\x00" * 100) is False

    def test_mp3_not_silk(self):
        assert looks_like_silk(b"\xff\xfb" + b"\x00" * 100) is False

    def test_empty(self):
        assert looks_like_silk(b"") is False

    def test_short_data(self):
        assert looks_like_silk(b"\x02") is False


# ── raw_pcm_to_wav ─────────────────────────────────────────


class TestRawPcmToWav:
    def test_converts_pcm_to_wav(self):
        # Generate 1 second of silence at 16kHz, 16-bit, mono
        pcm_data = b"\x00\x00" * 16000
        result = raw_pcm_to_wav(pcm_data)
        assert result is not None
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_empty_data(self):
        result = raw_pcm_to_wav(b"")
        # Should still produce a valid WAV (with 0 frames)
        assert result is not None
        assert result[:4] == b"RIFF"

    def test_custom_params(self):
        pcm_data = b"\x00\x00" * 8000
        result = raw_pcm_to_wav(pcm_data, sample_rate=8000, sample_width=2, channels=1)
        assert result is not None
        # Verify the WAV header
        with wave.open(__import__("io").BytesIO(result), "rb") as wf:
            assert wf.getframerate() == 8000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
