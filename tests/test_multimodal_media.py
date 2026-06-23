"""Tests for multimodal media — describe_media 视觉跟随激活模型。

设计:describe_media 始终注册;视觉来源跟随当前激活模型(经 FallbackChain._active_idx)。
激活多模态→用它看图;激活文本+runner→runner;都没有→视觉未启用(非 error)。
切换(自动/手动/reload)实时跟随。
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.agent.client import ChatResponse, FallbackChain


def _make_chain(primary_multimodal: bool, fallback_multimodal: bool, active_idx: int) -> FallbackChain:
    """构造真实 FallbackChain(isinstance 检查通过),设激活 idx + multimodal 标志。"""
    primary = MagicMock()
    primary.chat = AsyncMock(return_value=ChatResponse(content="primary看图", tool_calls=[]))
    primary.model = "primary-model"
    fb = MagicMock()
    fb.chat = AsyncMock(return_value=ChatResponse(content="fallback看图", tool_calls=[]))
    fb.model = "fallback-model"
    chain = FallbackChain(primary, [fb], multimodal_flags=[primary_multimodal, fallback_multimodal])
    chain._active_idx = active_idx
    return chain


class TestModelConfigMultimodal:
    def test_default_false(self):
        from src.config import ModelConfig

        assert ModelConfig().multimodal is False

    def test_set_true(self):
        from src.config import ModelConfig

        assert ModelConfig(multimodal=True).multimodal is True

    def test_fallback_multimodal_default_false(self):
        from src.config import ModelFallback

        assert ModelFallback(provider="openai", name="x").multimodal is False

    def test_fallback_multimodal_set_true(self):
        from src.config import ModelFallback

        assert ModelFallback(provider="openai", name="x", multimodal=True).multimodal is True


class TestGetToolsAlwaysRegisters:
    def test_always_registers_regardless_of_vision_state(self, monkeypatch):
        # get_tools 始终注册 describe_media(视觉开关由内部判断,不靠注册 → KV cache 稳定)
        import src._container as cont

        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=False),
                tools=SimpleNamespace(
                    media_understanding=SimpleNamespace(enabled=False, name="", image=SimpleNamespace(name=""))
                ),
            )
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)
        from src.tools.media_understanding_tools import get_tools

        assert len(get_tools()) == 1


class TestDescribeMediaVisionFollowsActive:
    """视觉来源跟随当前激活模型(覆盖用户的 6 种场景里的切换部分)。"""

    def _setup(self, monkeypatch, chain, runner):
        import src.tools.media_understanding_tools as mut
        import src._container as cont

        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=False, name="primary-model"),
                tools=SimpleNamespace(media_understanding=SimpleNamespace(enabled=True)),
            ),
            agent_loop=SimpleNamespace(_client=chain),
            media_understanding_runner=runner,
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)

        async def fake_resolve(source, default_mime):
            return b"\x89PNG_faked", "image/png"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve)
        return mut

    def test_active_primary_multimodal_uses_primary(self, monkeypatch):
        chain = _make_chain(True, False, active_idx=0)
        mut = self._setup(monkeypatch, chain, runner=None)
        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert "primary看图" in result
        chain._all[0].chat.assert_awaited_once()

    def test_enabled_false_kills_vision_even_if_model_multimodal(self, monkeypatch):
        # 统一到 enable:enabled=false → 视觉全关,哪怕主模型多模态。
        # 堵 model.multimodal=true 独立开门绕过 enable 的漏(修复前此 case 返回 "primary看图")。
        import src.tools.media_understanding_tools as mut
        import src._container as cont

        chain = _make_chain(True, False, active_idx=0)  # 主模型多模态
        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=True, name="primary-model"),
                tools=SimpleNamespace(media_understanding=SimpleNamespace(enabled=False)),
            ),
            agent_loop=SimpleNamespace(_client=chain),
            media_understanding_runner=None,
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)

        resolve_spy = AsyncMock(return_value=(b"\x89PNG", "image/png"))
        monkeypatch.setattr(mut, "_resolve_media_input", resolve_spy)

        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert result == "[视觉功能未启用]"
        chain._all[0].chat.assert_not_called()  # enabled=false 不该用主模型看图
        resolve_spy.assert_not_awaited()  # 也不该下载

    def test_active_fallback_multimodal_uses_fallback(self, monkeypatch):
        # 场景④⑤: 切到多模态 fallback → 用它(覆盖 primary/runner)
        chain = _make_chain(False, True, active_idx=1)
        mut = self._setup(monkeypatch, chain, runner=None)
        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert "fallback看图" in result
        chain._all[0].chat.assert_not_called()  # primary 没被调
        chain._all[1].chat.assert_awaited_once()

    def test_active_text_fallback_with_runner_uses_runner(self, monkeypatch):
        # 场景⑥配了: 切到文本 fallback + runner 可用 → runner(保留)
        chain = _make_chain(True, False, active_idx=1)
        fake_result = MagicMock(error=None, text="Qwen看图", model="Qwen-VL")
        runner = MagicMock()
        runner.understand = AsyncMock(return_value=fake_result)
        mut = self._setup(monkeypatch, chain, runner=runner)
        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert "Qwen看图" in result
        chain._all[1].chat.assert_not_called()  # 文本 fallback 不调

    def test_active_text_fallback_no_runner_vision_off(self, monkeypatch):
        # 场景⑥没配: 切到文本 fallback + 无 runner → 视觉未启用(非 error)
        chain = _make_chain(True, False, active_idx=1)
        mut = self._setup(monkeypatch, chain, runner=None)
        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert result == "[视觉功能未启用]"
        chain._all[1].chat.assert_not_called()

    def test_vision_off_returns_before_download(self, monkeypatch):
        # finding 4: 视觉未启用(文本激活+无runner) → 下载媒体前早返,_resolve_media_input 不应被 await。
        # 修复前会先把整张图/视频下完再返回 [视觉功能未启用](群聊浪费 + 无上限下载 DoS 面)。
        import src.tools.media_understanding_tools as mut
        import src._container as cont

        chain = _make_chain(False, False, active_idx=0)  # 文本激活,非多模态
        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=False, name="primary-model"),
                tools=SimpleNamespace(media_understanding=SimpleNamespace(enabled=True)),
            ),
            agent_loop=SimpleNamespace(_client=chain),
            media_understanding_runner=None,
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)

        resolve_spy = AsyncMock(return_value=(b"\x89PNG", "image/png"))
        monkeypatch.setattr(mut, "_resolve_media_input", resolve_spy)

        result = asyncio.run(mut.describe_media("http://x/big.png"))
        assert result == "[视觉功能未启用]"
        resolve_spy.assert_not_awaited()  # 关键:视觉未启用时根本不该下载

    def test_video_multimodal_active_uses_active_client(self, monkeypatch):
        # finding 3: video + 激活多模态 + 无 runner → 走激活 client(透传 video_url),
        # 不再静默返"视觉未启用"。ChatClient.chat 是透传,能发 video_url block。
        chain = _make_chain(True, False, active_idx=0)
        mut = self._setup(monkeypatch, chain, runner=None)

        async def fake_resolve_video(source, default_mime):
            return b"\x00\x00\x00\x20ftyp", "video/mp4"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve_video)

        result = asyncio.run(mut.describe_media("http://x/v.mp4"))
        assert "primary看图" in result  # 走激活 client 成功,不再静默未启用
        assert "video" in result  # [video description] 标记
        chain._all[0].chat.assert_awaited_once()

    def test_single_chat_client_text_uses_runner(self, monkeypatch):
        # 单 ChatClient(无 fallback,_active_vision_info else 分支)+ 文本 → runner
        import src.tools.media_understanding_tools as mut
        import src._container as cont

        fake_main = MagicMock()  # 非 FallbackChain
        fake_main.multimodal = False  # 文本模型(显式,防 MagicMock truthy 自动属性干扰 _active_vision_info)
        fake_result = MagicMock(error=None, text="Qwen看图", model="Qwen-VL")
        runner = MagicMock()
        runner.understand = AsyncMock(return_value=fake_result)
        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=False, name="LongCat"),
                tools=SimpleNamespace(media_understanding=SimpleNamespace(enabled=True)),
            ),
            agent_loop=SimpleNamespace(_client=fake_main),
            media_understanding_runner=runner,
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)

        async def fake_resolve(source, default_mime):
            return b"\x89PNG", "image/png"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve)

        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert "Qwen看图" in result
        fake_main.chat.assert_not_called()  # 文本模型不调主 client
        runner.understand.assert_awaited_once()

    def test_single_chat_client_multimodal_follows_client_not_config(self, monkeypatch):
        # finding #2: 单 ChatClient + client.multimodal=True 但静态 config.model.multimodal=False
        # (模拟 dashboard 切到多模态模型后 config 未更新) → _active_vision_info 应读 client.multimodal 走激活 client
        import src.tools.media_understanding_tools as mut
        import src._container as cont

        fake_main = MagicMock()
        fake_main.multimodal = True  # 新 client 多模态(切换后)
        fake_main.model = "new-mm-model"
        fake_main.chat = AsyncMock(return_value=ChatResponse(content="新模型看图", tool_calls=[]))
        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=False, name="old-text-model"),  # 旧 config 未更新
                tools=SimpleNamespace(media_understanding=SimpleNamespace(enabled=True)),
            ),
            agent_loop=SimpleNamespace(_client=fake_main),
            media_understanding_runner=None,
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)

        async def fake_resolve(source, default_mime):
            return b"\x89PNG", "image/png"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve)

        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert "新模型看图" in result  # 走激活 client(读 client.multimodal=True, 非旧 config False)
        fake_main.chat.assert_awaited_once()

    def test_audio_early_return(self, monkeypatch):
        # audio 早返,不走 runner/client
        chain = _make_chain(True, False, active_idx=0)
        mut = self._setup(monkeypatch, chain, runner=None)
        result = asyncio.run(mut.describe_media("http://x/a.mp3"))
        assert "Audio" in result  # [error] Audio files are not supported...
        chain._all[0].chat.assert_not_called()

    def test_video_multimodal_active_client_fails_falls_back_to_runner(self, monkeypatch):
        # finding 3 降级: video + 激活多模态 + 激活client不支持视频(抛错) + 有runner → 降级 runner。
        # multimodal flag 只保证图像不保证视频,后端拒收 video_url 时必须能降级。
        chain = _make_chain(True, False, active_idx=0)
        chain._all[0].chat = AsyncMock(side_effect=Exception("model does not support video input"))
        fake_result = MagicMock(error=None, text="runner看视频", model="Qwen-VL")
        runner = MagicMock()
        runner.understand = AsyncMock(return_value=fake_result)
        mut = self._setup(monkeypatch, chain, runner=runner)

        async def fake_resolve_video(source, default_mime):
            return b"\x00\x00\x00\x20ftyp", "video/mp4"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve_video)

        result = asyncio.run(mut.describe_media("http://x/v.mp4"))
        assert "runner看视频" in result  # 降级到 runner
        chain._all[0].chat.assert_awaited_once()  # 先试了激活 client
        runner.understand.assert_awaited_once()  # 失败后降级 runner

    def test_video_multimodal_active_client_fails_no_runner_vision_off(self, monkeypatch):
        # finding 3 降级到底: video + 激活多模态 + client抛错 + 无runner → 未启用(试过 client 才返)
        chain = _make_chain(True, False, active_idx=0)
        chain._all[0].chat = AsyncMock(side_effect=Exception("model does not support video input"))
        mut = self._setup(monkeypatch, chain, runner=None)

        async def fake_resolve_video(source, default_mime):
            return b"\x00\x00\x00\x20ftyp", "video/mp4"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve_video)

        result = asyncio.run(mut.describe_media("http://x/v.mp4"))
        assert result == "[视觉功能未启用]"  # 降级到底
        chain._all[0].chat.assert_awaited_once()  # 试过激活 client

    def test_image_multimodal_active_client_fails_falls_back_to_runner(self, monkeypatch):
        # image + 激活多模态 + 激活client抛错(429/网络/API error) + 有runner → 降级 runner(与 video 对称)。
        # 修复前:image 路径无 try/except,失败直接返 [error],够不到 runner。
        chain = _make_chain(True, False, active_idx=0)
        chain._all[0].chat = AsyncMock(side_effect=Exception("rate limit"))
        fake_result = MagicMock(error=None, text="runner看图", model="Qwen-VL")
        runner = MagicMock()
        runner.understand = AsyncMock(return_value=fake_result)
        mut = self._setup(monkeypatch, chain, runner=runner)  # _setup 的 fake_resolve 返 image/png

        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert "runner看图" in result  # 降级到 runner
        chain._all[0].chat.assert_awaited_once()  # 先试了激活 client
        runner.understand.assert_awaited_once()  # 失败后降级 runner

    def test_image_multimodal_active_client_fails_no_runner_returns_error(self, monkeypatch):
        # image + 激活多模态 + client抛错 + 无runner → [error](保留原错误语义,不假装"视觉未启用")。
        chain = _make_chain(True, False, active_idx=0)
        chain._all[0].chat = AsyncMock(side_effect=Exception("rate limit"))
        mut = self._setup(monkeypatch, chain, runner=None)

        result = asyncio.run(mut.describe_media("http://x/a.png"))
        assert result.startswith("[error]")  # 无降级目标 → 原错误语义
        chain._all[0].chat.assert_awaited_once()

    def test_single_chat_client_multimodal_video_uses_active_client(self, monkeypatch):
        # 单 ChatClient(非 FallbackChain,_active_vision_info else 分支)+ 多模态 + video → 走激活 client。
        # 补 reviewer 指出的覆盖缺口:单 client 路径的 video 分支此前未测。
        import src.tools.media_understanding_tools as mut
        import src._container as cont

        fake_main = MagicMock()
        fake_main.multimodal = True  # 单 client 多模态(显式,与同类测试一致,不靠 MagicMock truthy 自动属性)
        fake_main.chat = AsyncMock(return_value=ChatResponse(content="主模型看视频", tool_calls=[]))
        fake_main.model = "main-model"
        container = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(multimodal=True, name="main-model"),
                tools=SimpleNamespace(media_understanding=SimpleNamespace(enabled=True)),
            ),
            agent_loop=SimpleNamespace(_client=fake_main),
            media_understanding_runner=None,
        )
        monkeypatch.setattr(cont, "get_container", lambda: container)

        async def fake_resolve_video(source, default_mime):
            return b"\x00\x00\x00\x20ftyp", "video/mp4"

        monkeypatch.setattr(mut, "_resolve_media_input", fake_resolve_video)

        result = asyncio.run(mut.describe_media("http://x/v.mp4"))
        assert "主模型看视频" in result
        assert "video" in result
        fake_main.chat.assert_awaited_once()

    def test_describe_with_client_video_sends_video_url_block(self):
        # _describe_with_client(VIDEO) 必须构造 video_url block(防回归:image/video block 构造被交换)。
        from src.media_understanding.types import MediaCapability
        import src.tools.media_understanding_tools as mut

        captured = {}

        async def fake_chat(messages, tools=None, **extra):
            captured["messages"] = messages
            return ChatResponse(content="ok", tool_calls=[])

        fake_client = MagicMock()
        fake_client.chat = fake_chat

        result = asyncio.run(
            mut._describe_with_client(fake_client, "m", b"vidbytes", "video/mp4", capability=MediaCapability.VIDEO)
        )
        content = captured["messages"][0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[1]["type"] == "text"
        assert "video" in result
