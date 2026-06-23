from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("flyclaw.tools.media_understanding")


def _get_runner():
    from src._container import get_container

    return get_container().media_understanding_runner


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    if not data_url.startswith("data:"):
        raise ValueError("Invalid data URL: must start with 'data:'")
    if "," not in data_url:
        raise ValueError("Invalid data URL: missing comma separator")
    header, b64 = data_url.split(",", 1)
    if ":" not in header:
        raise ValueError("Invalid data URL: missing MIME type")
    mime_part = header.split(":", 1)[1].split(";")[0]
    return base64.b64decode(b64), mime_part


def _strip_mime_params(content_type: str) -> str:
    return content_type.split(";")[0].strip() if content_type else ""


async def _resolve_media_input(source: str, default_mime: str) -> tuple[bytes, str]:
    """Resolve a media source (data URL, local file path, or HTTP URL) to (bytes, mime_type)."""
    if source.startswith("data:"):
        return _decode_data_url(source)

    # Local file path (no scheme, or Windows drive like D:\...)
    parsed = urlparse(source)
    if not parsed.scheme or (len(parsed.scheme) == 1 and parsed.scheme.isalpha()):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {source}")
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        return data, mime or default_mime

    # Remote URL — use DNS-pinned safe_fetch for SSRF protection
    from src.security.url_safety import safe_fetch

    resp = await safe_fetch(source, timeout=30.0)
    if resp.status_code != 200:
        raise ValueError(f"Failed to download: HTTP {resp.status_code}")
    mime_type = _strip_mime_params(resp.headers.get("content-type", default_mime))
    return resp.content, mime_type


_AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".wma", ".opus", ".silk", ".amr", ".speex"}
)


def _guess_is_audio(source: str) -> bool:
    """Quick check whether a source looks like audio without downloading it."""
    # data: URL — check MIME part
    if source.startswith("data:"):
        try:
            header = source.split(",", 1)[0]
            mime = header.split(":", 1)[1].split(";")[0]
            return mime.startswith("audio/")
        except (IndexError, ValueError):
            return False

    # Local file — check extension
    parsed = urlparse(source)
    if not parsed.scheme or (len(parsed.scheme) == 1 and parsed.scheme.isalpha()):
        return Path(source).suffix.lower() in _AUDIO_EXTENSIONS

    # Remote URL — check extension from path
    return Path(parsed.path).suffix.lower() in _AUDIO_EXTENSIONS


async def describe_media(media_url: str) -> str:
    """Describe/analyze an image or video. Provide a URL, a data: base64 URL, or a local file path.

    视觉来源跟随**当前激活模型**:激活模型多模态→用它看图(覆盖 media_understanding);
    激活模型文本→用 tools.media_understanding 配的独立视觉模型;两者都不满足→视觉未启用。
    自动切/手动切/reload 切换都实时跟随(每次调用读激活模型)。video 也优先跟随激活多模态模型
    (ChatClient.chat 透传 video_url),后端不支持视频时降级独立视觉模型;audio 不支持(早返)。

    Args:
        media_url: URL of the image or video, a data:...;base64,... data URL, or a local file path.
    """
    try:
        # Early rejection: skip downloading audio files entirely
        if _guess_is_audio(media_url):
            return "[error] Audio files are not supported by describe_media."

        # 视觉是否启用是纯内存判断(_active_vision_info / _get_runner 都不触网),
        # 必须在 _resolve_media_input 之前判断:否则视觉关闭时会把整张图/整段视频
        # 下完(走 safe_fetch,无字节上限)再返回 [视觉功能未启用] —— 群聊频繁媒体
        # 重复大流量浪费,且无上限下载是轻量 DoS 面。
        from src._container import get_container

        container = get_container()
        # enable 是视觉唯一总开关:关 → 直接未启用(哪怕主模型多模态),不下载、不触网。
        # 统一前 model.multimodal=true 能绕过 enabled 独立开门——这行是堵那个漏。
        if not container.config.tools.media_understanding.enabled:
            return "[视觉功能未启用]"
        active_client, active_multimodal, model_name = _active_vision_info(container)
        runner = _get_runner()
        if not active_multimodal and runner is None:
            return "[视觉功能未启用]"

        data, mime_type = await _resolve_media_input(media_url, "image/png")

        from src.media_understanding.types import MediaCapability
        from src.media_understanding.runner import MediaUnderstandingRunner

        capability = MediaUnderstandingRunner.guess_capability_from_mime(mime_type) or MediaCapability.IMAGE
        if capability == MediaCapability.AUDIO:
            return "[error] Audio files are not supported by describe_media."

        # 激活模型多模态 + 图像 → 优先用激活模型看图;瞬时失败(429/网络/API error)有 runner 则降级(与 video 对称)。
        # 注意:multimodal flag 是用户手填布尔(能力声明),不是可用性 SLA——429/网络错不是能力缺失,
        # 此时该用已配好的 runner 兜底,而非直接 [error]。无 runner 则保留原 [error] 语义。
        if active_multimodal and capability == MediaCapability.IMAGE:
            try:
                return await _describe_with_client(active_client, model_name, data, mime_type)
            except Exception as e:
                if runner is None:
                    return f"[error] {e}"  # 无降级目标,保留原错误语义
                logger.warning("激活模型 %s 处理图像失败,降级独立视觉模型: %s", model_name, e)
                # 落到下面的 runner 路径降级(capability==IMAGE,会跳过 video 分支)

        # 激活模型多模态 + 视频 → 也优先用激活 client(ChatClient.chat 透传 video_url block)。
        # 但 multimodal flag 只保证图像、不保证视频;后端若拒收 video_url 则降级到独立视觉模型。
        if active_multimodal and capability == MediaCapability.VIDEO:
            try:
                return await _describe_with_client(
                    active_client, model_name, data, mime_type, capability=MediaCapability.VIDEO
                )
            # 宽 catch 有意为之:multimodal flag 只保证图像、不保证视频,video 是尽力而为。
            # 这里走 active_client.chat(单 ChatClient,非 chain.chat)→ 失败不触发 FallbackChain
            # cooldown、不 fallback(video 失败不该冷却/拖垮正常工作的主文本模型);网络/认证错也
            # 降级到 runner(独立端点,可能可用)。warning 记完整异常可追溯。
            except Exception as e:
                logger.warning("激活模型 %s 处理视频失败,降级独立视觉模型: %s", model_name, e)
                # 落到下面的 runner 路径降级

        # 否则(文本激活看图/video, 或 video 激活降级)→ 独立视觉模型; 没配则视觉未启用
        if runner is None:
            return "[视觉功能未启用]"
        result = await runner.understand(data, capability, mime_type=mime_type)
        if result.error:
            return f"[error] {result.error}"
        label = "video" if capability == MediaCapability.VIDEO else "image"
        return f"[{label} description] ({result.model})\n{result.text}"
    except Exception as e:
        logger.error("describe_media error: %s", e)
        return f"[error] {e}"


def _active_vision_info(container):
    """拿当前激活模型的 (client, is_multimodal, model_name)。

    FallbackChain: 激活 = _all[_active_idx],multimodal = _multimodal_flags[_active_idx];
    单 ChatClient: 激活 = 本身,multimodal = config.model.multimodal。
    """
    from src.agent.client import FallbackChain

    client = container.agent_loop._client
    if isinstance(client, FallbackChain):
        active = client.active
        return active, client.active_multimodal, active.model
    return client, getattr(client, "multimodal", False), client.model


async def _describe_with_client(client, model_name: str, data: bytes, mime_type: str, *, capability=None) -> str:
    """用指定(激活)模型 client 看图/看视频。

    capability=VIDEO 时构造 video_url block(ChatClient.chat 透传给 OpenAI 兼容后端);
    否则构造 image_url block。
    """
    from src.media_understanding.types import MediaCapability

    data_url = f"data:{mime_type or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"
    if capability == MediaCapability.VIDEO:
        content_blocks = [
            {"type": "video_url", "video_url": {"url": data_url}},
            {"type": "text", "text": "Describe this video in detail. What is happening?"},
        ]
        label = "video"
    else:
        content_blocks = [
            {"type": "text", "text": "Describe this image in detail. If it contains text, transcribe the text."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        label = "image"
    messages = [{"role": "user", "content": content_blocks}]
    response = await client.chat(messages, tools=None)
    text = getattr(response, "content", "") or ""
    return f"[{label} description] ({model_name})\n{text}"


def get_tools() -> list:
    from src.agent.tooldef import ToolDef

    # 始终注册 describe_media:视觉开关(跟随激活模型)由 describe_media 内部判断,
    # 不靠注册——这样工具集稳定,KV cache 不被模型切换破坏。
    return [ToolDef.from_function(describe_media)]
