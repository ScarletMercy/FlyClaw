"""多实例支持：通过实例编号隔离完整 home 目录。

用法:
    flyclaw          → 默认实例 (~/.flyclaw/)
    flyclaw 2        → 实例 2  (~/.flyclaw2/)
    flyclaw setup 2  → 配置实例 2
    flyclaw doctor 2 → 诊断实例 2

命名约定:
    默认实例: ~/.flyclaw/        (config.yaml, data/, skills/, …)
    实例 N:   ~/.flyclawN/       (config.yaml, data/, skills/, …)
"""

from __future__ import annotations

import sys
from pathlib import Path

_instance_number: int | None = None


def home_dir(n: int | None = None) -> Path:
    """返回实例的 home 目录。"""
    if n is None:
        n = _instance_number
    if n is None:
        return Path.home() / ".flyclaw"
    return Path.home() / f".flyclaw{n}"


def parse_instance_from_argv() -> int | None:
    """从 sys.argv 尾部剥离实例编号。

    在 argparse 运行之前调用。如果最后一个参数是纯数字 ≥ 1，
    将其从 sys.argv 中移除并返回。

    已知子命令中接受尾部数字参数的路径会跳过剥离，
    避免 ``flyclaw model switch 2`` 中的 "2" 被误判为实例号。
    """
    # 已知子命令路径：最后一个词是子命令名，其接受一个数字位置参数
    # 例如 ("model", "switch") 表示 `flyclaw model switch <id>`
    # 新增带数字参数的子命令时必须在此登记，否则尾部数字会被误判为实例号
    _NUMERIC_SUBCOMMANDS: set[tuple[str, ...]] = {
        ("model", "switch"),
    }

    if len(sys.argv) >= 2:
        last = sys.argv[-1]
        if last.isdigit() and not last.startswith("-"):
            # 检查是否匹配已知带数字参数的子命令路径
            prefix = tuple(sys.argv[1:-1])  # 去掉 argv[0] 和最后一个
            if prefix in _NUMERIC_SUBCOMMANDS:
                return None
            val = int(last)
            if val >= 1:
                sys.argv.pop()
                return val
    return None


def set_instance(n: int | None) -> None:
    """设置当前实例编号（进程启动时调用一次）。"""
    global _instance_number
    _instance_number = n


def get_instance() -> int | None:
    """返回当前实例编号，None 表示默认实例。"""
    return _instance_number


def config_path(n: int | None = None) -> Path:
    """返回配置文件路径。"""
    return home_dir(n) / "config.yaml"


def data_dir(n: int | None = None) -> Path:
    """返回数据目录路径。"""
    return home_dir(n) / "data"


def skills_dir(n: int | None = None) -> Path:
    """返回技能目录路径。"""
    return home_dir(n) / "skills"


def temp_dir(n: int | None = None) -> Path:
    """返回临时文件目录路径。"""
    return home_dir(n) / "temp"


def service_name(n: int | None = None) -> str:
    """返回守护进程服务名。"""
    if n is None:
        n = _instance_number
    if n is None:
        return "flyclaw"
    return f"flyclaw-{n}"


def instance_label(n: int | None = None) -> str:
    """返回实例标签，默认实例为空字符串，实例 N 为 '-N'。"""
    if n is None:
        n = _instance_number
    if n is None:
        return ""
    return f"-{n}"
