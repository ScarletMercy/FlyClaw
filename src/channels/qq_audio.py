"""QQ voice message audio processing — SILK/AMR decoding and STT transcription.

Ported from hermes-agent gateway/platforms/qqbot/adapter.py audio pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger("flyclaw.channels.qq_audio")

_VOICE_EXTENSIONS = frozenset(
    {".silk", ".amr", ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".speex", ".flac"}
)


def guess_audio_ext(data: bytes) -> str:
    if data[:9] == b"#!SILK_V3" or data[:5] == b"#!SILK":
        return ".silk"
    if data[:2] == b"\x02!":
        return ".silk"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return ".mp3"
    if data[:4] == b"\x30\x26\xb2\x75" or data[:4] == b"\x4f\x67\x67\x53":
        return ".ogg"
    if data[:4] in {b"\x00\x00\x00\x20", b"\x00\x00\x00\x1c"}:
        return ".amr"
    return ".amr"


def looks_like_silk(data: bytes) -> bool:
    return data[:4] == b"#!SILK" or data[:2] == b"\x02!" or data[:9] == b"#!SILK_V3"


async def convert_to_wav(audio_data: bytes, ext: str = "") -> Optional[bytes]:
    if not audio_data or len(audio_data) < 10:
        return None

    if not ext:
        ext = guess_audio_ext(audio_data)

    if ext == ".wav":
        if audio_data[:4] == b"RIFF":
            return audio_data
        return raw_pcm_to_wav(audio_data)

    src_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_data)
            src_path = tmp.name
        wav_path = src_path.rsplit(".", 1)[0] + ".wav"

        is_silk = ext == ".silk" or looks_like_silk(audio_data)
        if is_silk:
            result = await _silk_to_wav(src_path, wav_path)
            if result:
                return Path(wav_path).read_bytes()

        result = await _ffmpeg_to_wav(src_path, wav_path)
        if result:
            return Path(wav_path).read_bytes()

        return raw_pcm_to_wav(audio_data)
    except Exception as e:
        logger.warning("Audio conversion failed: %s", e)
        return None
    finally:
        for p in (src_path, wav_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


async def _silk_to_wav(src_path: str, wav_path: str) -> Optional[str]:
    try:
        import pilk
    except ImportError:
        return None

    try:
        pilk.silk_to_wav(src_path, wav_path, rate=16000)
        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 44:
            return wav_path
    except Exception:
        pass

    import shutil
    silk_path = src_path.rsplit(".", 1)[0] + ".silk"
    try:
        shutil.copy2(src_path, silk_path)
        pilk.silk_to_wav(silk_path, wav_path, rate=16000)
        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 44:
            return wav_path
    except Exception:
        pass
    finally:
        try:
            os.unlink(silk_path)
        except OSError:
            pass

    return None


async def _ffmpeg_to_wav(src_path: str, wav_path: str) -> Optional[str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", src_path,
            "-ar", "16000", "-ac", "1", wav_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.wait(), timeout=30)
        if proc.returncode != 0:
            return None
    except (asyncio.TimeoutError, FileNotFoundError):
        return None

    if not Path(wav_path).exists() or Path(wav_path).stat().st_size <= 44:
        return None
    return wav_path


def raw_pcm_to_wav(data: bytes, sample_rate: int = 16000, sample_width: int = 2, channels: int = 1) -> Optional[bytes]:
    try:
        buf = __import__("io").BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(data)
        return buf.getvalue()
    except Exception:
        return None
