"""Tests for `flyclaw model add` 修复:聚焦加一个回退模型,不再跑完整 setup 向导。"""

from types import SimpleNamespace
from unittest.mock import MagicMock


def test_prompt_fallback_model_collects_all_fields_including_multimodal(monkeypatch):
    # _prompt_fallback_model 应收齐 provider/name/base_url/api_key/context_window/multimodal
    import src.cli as cli
    from src.config import ModelFallback

    monkeypatch.setattr("src.setup._ask_choice", lambda p, c, default="": "custom")
    answers = {
        "  模型名称": "GLM-4.6V-Flash",
        "  接口地址": "https://open.bigmodel.cn/api/paas/v4",
        "  API 密钥": "key123",
    }
    monkeypatch.setattr("src.setup._ask", lambda p, default="": answers.get(p, default))
    monkeypatch.setattr("src.setup._ask_int", lambda p, default: 128000)
    monkeypatch.setattr("src.setup._ask_yn", lambda p, default=True: True)

    fb = cli._prompt_fallback_model()
    assert isinstance(fb, ModelFallback)
    assert fb.provider == "openai"  # custom 分支 → openai
    assert fb.name == "GLM-4.6V-Flash"
    assert fb.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert fb.api_key == "key123"
    assert fb.context_window == 128000
    assert fb.multimodal is True


def test_cmd_model_add_appends_fallback_and_does_not_run_wizard(monkeypatch):
    # 修复前 model add 调 run_wizard(跑完整向导);修复后应聚焦加一个 fallback + 存盘
    import src.cli as cli
    from src.config import AppConfig, ModelConfig, ModelFallback

    config = AppConfig(model=ModelConfig(provider="openai", name="primary", api_key="k"))
    assert config.model.fallbacks == []
    monkeypatch.setattr(cli, "_load_config", lambda: config)

    save_spy = MagicMock()
    monkeypatch.setattr("src.config.save_config", save_spy)
    wizard_spy = MagicMock()
    monkeypatch.setattr("src.setup.run_wizard", wizard_spy)
    new_fb = ModelFallback(provider="openai", name="GLM-4.6V-Flash", multimodal=True)
    monkeypatch.setattr(cli, "_prompt_fallback_model", lambda: new_fb)

    rc = cli.cmd_model(SimpleNamespace(model_command="add"))
    assert rc == 0
    wizard_spy.assert_not_called()  # 关键:不再跑完整向导
    assert config.model.fallbacks == [new_fb]  # append 进 fallbacks
    save_spy.assert_called_once_with(config)  # 存盘


def _switch_config():
    """primary A + fallbacks [B, C],字段各异便于验证交换到字段级。"""
    from src.config import AppConfig, ModelConfig, ModelFallback

    return AppConfig(
        model=ModelConfig(
            provider="openai",
            name="A",
            api_key="ka",
            base_url="http://a",
            context_window=1000,
            multimodal=False,
            fallbacks=[
                ModelFallback(
                    provider="openai", name="B", api_key="kb", base_url="http://b", context_window=2000, multimodal=True
                ),
                ModelFallback(
                    provider="openai",
                    name="C",
                    api_key="kc",
                    base_url="http://c",
                    context_window=3000,
                    multimodal=False,
                ),
            ],
        )
    )


def test_cmd_model_switch_swaps_primary_with_fallback(monkeypatch):
    # switch 1: B 升主,A 降入 fb[0],C 不动;无重复,总数 3。
    # 修复前:B 升主但留 fb[0](=B)→ primary B, fb [B, C](重复)→ fb[0].name != "A" 失败。
    import src.cli as cli

    config = _switch_config()
    monkeypatch.setattr(cli, "_load_config", lambda: config)
    save_spy = MagicMock()
    monkeypatch.setattr("src.config.save_config", save_spy)

    rc = cli.cmd_model(SimpleNamespace(model_command="switch", id=1))
    assert rc == 0
    assert config.model.name == "B"
    assert config.model.multimodal is True  # B 的字段带上主
    assert [fb.name for fb in config.model.fallbacks] == ["A", "C"]  # A 降入,C 不动,无 B 重复
    assert config.model.fallbacks[0].multimodal is False  # A 的字段降下去
    save_spy.assert_called_once()


def test_cmd_model_switch_zero_is_noop(monkeypatch):
    # switch 0 = 已是主模型 → no-op,不改 config、不写盘。
    import src.cli as cli

    config = _switch_config()
    monkeypatch.setattr(cli, "_load_config", lambda: config)
    save_spy = MagicMock()
    monkeypatch.setattr("src.config.save_config", save_spy)

    rc = cli.cmd_model(SimpleNamespace(model_command="switch", id=0))
    assert rc == 0
    assert config.model.name == "A"  # 没变
    assert [fb.name for fb in config.model.fallbacks] == ["B", "C"]
    save_spy.assert_not_called()  # no-op 不写盘


def test_cmd_model_switch_roundtrip_restores(monkeypatch):
    # switch 1(A→B)再 switch 1(B→A)→ 恢复原始排列。
    import src.cli as cli

    config = _switch_config()
    monkeypatch.setattr(cli, "_load_config", lambda: config)
    monkeypatch.setattr("src.config.save_config", MagicMock())

    cli.cmd_model(SimpleNamespace(model_command="switch", id=1))
    cli.cmd_model(SimpleNamespace(model_command="switch", id=1))
    assert config.model.name == "A"
    assert [fb.name for fb in config.model.fallbacks] == ["B", "C"]
