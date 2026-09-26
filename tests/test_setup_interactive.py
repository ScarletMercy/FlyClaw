"""交互式 questionary 分支的测试。

TTY 下 _ask* 走 questionary 控件（select/confirm/text，对齐 chcode）；非 TTY
（pytest 环境本身）自动回退 input() 文字交互，回退路径由 test_setup_wizard_*.py
覆盖。这里用假 questionary 模块替换 _interactive_ui() 返回值，驱动交互分支并
校验控件参数与退出语义。
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest

import src.setup as setup


def _fake_questionary(answers: list, calls: list):
    """构造假 questionary 模块：控件依次消费 answers，调用参数记入 calls。

    answers 中的异常实例会被 raise（用于模拟 Ctrl+C / Ctrl+D）。
    """

    def _control(kind):
        def _make(message, **kwargs):
            calls.append((kind, message, kwargs))

            class _Q:
                def unsafe_ask(self):
                    item = answers.pop(0)
                    if isinstance(item, BaseException):
                        raise item
                    return item

            return _Q()

        return _make

    return types.SimpleNamespace(
        select=_control("select"),
        confirm=_control("confirm"),
        text=_control("text"),
        Style=lambda pairs: ("fake-style", pairs),
    )


def _enable_fake_ui(monkeypatch, answers, calls):
    fake = _fake_questionary(answers, calls)
    monkeypatch.setattr(setup, "_interactive_ui", lambda: fake)


class TestInteractiveControls:
    def test_select_strips_prompt_and_passes_default(self, monkeypatch):
        calls: list = []
        _enable_fake_ui(monkeypatch, ["重输"], calls)
        assert setup._ask_choice("  操作", ["重试", "重输", "放弃"], default="重试") == "重输"
        kind, message, kwargs = calls[0]
        assert kind == "select"
        assert message == "操作"  # 前导缩进已剥离，交给 questionary 排版
        assert kwargs["choices"] == ["重试", "重输", "放弃"]
        assert kwargs["default"] == "重试"
        # noinherit 去高亮背景（Windows 终端渲染问题，对齐 chcode）
        assert kwargs["style"] == ("fake-style", [("highlighted", "noinherit"), ("selected", "noinherit")])

    def test_select_default_not_in_choices_passes_none(self, monkeypatch):
        calls: list = []
        _enable_fake_ui(monkeypatch, ["a"], calls)
        setup._ask_choice("选", ["a", "b"])  # default="" 不在选项里
        assert calls[0][2]["default"] is None

    def test_confirm_maps_bool(self, monkeypatch):
        calls: list = []
        _enable_fake_ui(monkeypatch, [False], calls)
        assert setup._ask_yn("  启用？", default=True) is False
        kind, message, kwargs = calls[0]
        assert kind == "confirm"
        assert message == "启用？"
        assert kwargs["default"] is True

    def test_text_empty_answer_falls_back_to_default(self, monkeypatch):
        calls: list = []
        _enable_fake_ui(monkeypatch, [""], calls)
        assert setup._ask("  模型名称", default="gpt-4o") == "gpt-4o"
        assert calls[0][0] == "text"
        assert calls[0][2]["default"] == "gpt-4o"

    def test_text_answer_stripped(self, monkeypatch):
        _enable_fake_ui(monkeypatch, ["  sk-xxx  "], [])
        assert setup._ask("  API 密钥") == "sk-xxx"

    def test_required_reasks_on_empty_without_default(self, monkeypatch, capsys):
        calls: list = []
        _enable_fake_ui(monkeypatch, ["", "val2"], calls)
        assert setup._ask_required("  新 API 密钥") == "val2"
        assert len(calls) == 2
        assert "不能为空" in capsys.readouterr().out

    def test_required_empty_answer_uses_default(self, monkeypatch):
        calls: list = []
        _enable_fake_ui(monkeypatch, [""], calls)
        assert setup._ask_required("  模型名称", default="m1") == "m1"
        assert calls[0][2]["default"] == "m1"


class TestTerminalFallback:
    """questionary 附着终端失败（如 mintty 伪终端非 winpty）→ 降级文字交互。"""

    def test_attach_failure_degrades_to_text_input(self, monkeypatch, capsys):
        monkeypatch.setattr(setup, "_interactive_disabled", False)

        def _boom(*args, **kwargs):
            raise RuntimeError("no console buffer")

        fake = types.SimpleNamespace(select=_boom, confirm=_boom, text=_boom, Style=lambda p: None)
        monkeypatch.setattr(setup, "_interactive_ui", lambda: fake)
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "重试")
        assert setup._ask_choice("  操作", ["重试", "重输", "放弃"], default="重试") == "重试"
        out = capsys.readouterr().out
        assert "交互式控件不可用" in out
        assert "请从以下选项" not in out  # 文字分支首次输入即命中，未重问

    def test_attach_failure_disables_interactive_globally(self, monkeypatch):
        monkeypatch.setattr(setup, "_interactive_disabled", False)
        setup._q_failed(RuntimeError("no console buffer"))
        assert setup._interactive_disabled is True
        assert setup._interactive_ui() is None  # 后续提问直接走文字分支


class TestInteractiveInterrupt:
    def test_ctrl_c_exits_wizard_without_save(self, monkeypatch):
        _enable_fake_ui(monkeypatch, [KeyboardInterrupt()], [])
        with pytest.raises(SystemExit) as exc_info:
            setup._ask_choice("  操作", ["重试", "重输", "放弃"])
        assert exc_info.value.code == 0  # 与 _ask 的 Ctrl+C 语义一致

    def test_eof_exits_wizard(self, monkeypatch):
        _enable_fake_ui(monkeypatch, [EOFError()], [])
        with pytest.raises(SystemExit):
            setup._ask("  模型名称")

    def test_ctrl_c_during_construction_exits_not_degrades(self, monkeypatch):
        """控件构造期到达的 Ctrl+C 同样干净退出，且不误触发降级。"""

        def _boom(*args, **kwargs):
            raise KeyboardInterrupt()

        fake = types.SimpleNamespace(select=_boom, confirm=_boom, text=_boom, Style=lambda p: None)
        monkeypatch.setattr(setup, "_interactive_ui", lambda: fake)
        monkeypatch.setattr(setup, "_interactive_disabled", False)
        with pytest.raises(SystemExit) as exc_info:
            setup._ask_choice("  操作", ["重试", "重输", "放弃"])
        assert exc_info.value.code == 0
        assert setup._interactive_disabled is False  # Ctrl+C 是退出意图，不是终端故障

    def test_ctrl_c_during_style_construction_exits_not_degrades(self, monkeypatch):
        """Style 构造也须在 _q_ask 守卫内：Ctrl+C 干净退出而非裸抛 KeyboardInterrupt。"""

        fake = types.SimpleNamespace(
            select=lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应到达")),
            confirm=None,
            text=None,
            Style=lambda pairs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        monkeypatch.setattr(setup, "_interactive_ui", lambda: fake)
        monkeypatch.setattr(setup, "_interactive_disabled", False)
        with pytest.raises(SystemExit) as exc_info:
            setup._ask_choice("  操作", ["重试", "重输", "放弃"])
        assert exc_info.value.code == 0
        assert setup._interactive_disabled is False


class TestInteractiveDetection:
    def test_non_tty_returns_none(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: True))
        assert setup._interactive_ui() is None

    def test_tty_without_questionary_returns_none(self, monkeypatch):
        monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(isatty=lambda: True))
        real_import = builtins.__import__

        def _no_questionary(name, *args, **kwargs):
            if name == "questionary":
                raise ImportError("no questionary")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_questionary)
        assert setup._interactive_ui() is None

    def test_ctrl_c_during_import_exits_wizard(self, monkeypatch):
        """Ctrl+C 打断 questionary 导入：干净退出而非裸抛 traceback。"""
        tty = types.SimpleNamespace(isatty=lambda: True, write=lambda *_: None, flush=lambda: None)
        monkeypatch.setattr(sys, "stdin", tty)
        monkeypatch.setattr(sys, "stdout", tty)
        real_import = builtins.__import__

        def _ki_questionary(name, *args, **kwargs):
            if name == "questionary":
                raise KeyboardInterrupt()
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _ki_questionary)
        with pytest.raises(SystemExit) as exc_info:
            setup._interactive_ui()
        assert exc_info.value.code == 0
