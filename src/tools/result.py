"""工具结果信封——工具层词汇表：定义工具执行结果的形状。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolResult:
    """工具结果信封：content 给模型看，status 给程序消费（审计/熔断/门控）。

    content 的 [error]/[blocked] 前缀措辞保留——delegate 回退等存量消费点
    仍按前缀嗅探。__str__ 回落 content，str() 压扁点拿到内容而非 repr。
    """

    content: str = ""
    status: Literal["success", "error"] = "success"

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content=content, status="error")

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    def __str__(self) -> str:
        return self.content
