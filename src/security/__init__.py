import unicodedata as _ud

from src.security.audit import run_security_audit
from src.security.credential_patterns import CREDENTIAL_PATTERNS
from src.security.redact import redact


def normalize_unicode(text: str) -> str:
    """NFD 分解后去除所有 Mn 组合标记，用于安全扫描前的文本归一化。

    攻击者可在关键词字符上叠加 combining marks（如 U+0300 重音符）使文本视觉不变
    但 regex 匹配失败。NFC 归一化不够（预组字符仍非 ASCII），需要 NFD 分解后剥离 Mn。
    不可见字符（Cf 类）不受影响。
    """
    return "".join(c for c in _ud.normalize("NFD", text) if _ud.category(c) != "Mn")


__all__ = ["CREDENTIAL_PATTERNS", "normalize_unicode", "run_security_audit", "redact"]
