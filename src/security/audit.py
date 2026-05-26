from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("flyclaw.security")


def run_security_audit(config) -> dict[str, Any]:
    results = {"passed": 0, "warnings": 0, "info": 0, "issues": []}

    def _check(name: str, severity: str, message: str):
        if severity == "PASS":
            results["passed"] += 1
            logger.info("[安全] 通过 %s", name)
        elif severity == "WARN":
            results["warnings"] += 1
            logger.warning("[安全] 警告 %s: %s", name, message)
            results["issues"].append({"name": name, "severity": "WARN", "message": message})
        else:
            results["info"] += 1
            logger.info("[安全] 信息 %s", name)

    host = config.gateway.host
    token = config.gateway.auth_token
    if host in ("0.0.0.0", "", "::", "[::]", "0:0:0:0:0:0:0:0") and not token:
        _check("gateway-auth", "WARN", f"网关暴露在 {host} 但未设置认证令牌")
    elif token:
        _check("gateway-auth", "PASS", "")

    if config.tools.exec.approval_mode == "off":
        _check("exec-approval", "INFO", "审批模式已关闭")
    else:
        _check("exec-approval", "PASS", "")

    data_dir = Path.home() / ".flyclaw" / "data"
    if data_dir.exists():
        _check("data-dir", "PASS", "")
    else:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            _check("data-dir", "PASS", "已创建 ~/.flyclaw/data/")
        except Exception:
            _check("data-dir", "WARN", "无法创建 ~/.flyclaw/data/ 目录")

    _check_secrets(config, results, _check)

    if getattr(config, "auth", None) and config.auth.enabled:
        _check("rbac-enabled", "PASS", "")
    else:
        _check("rbac-enabled", "INFO", "认证/RBAC 已禁用")

    logger.info("[安全] 审计完成: %d 项通过, %d 项警告", results["passed"], results["warnings"])
    return results


def _check_secrets(config, results, _check):
    """Scan config.yaml for potential hardcoded secrets."""
    import re

    config_path = Path("config.yaml")
    if not config_path.exists():
        return

    try:
        content = config_path.read_text(encoding="utf-8")
    except Exception:
        return

    # Patterns for potential secrets
    # sk- prefix (API keys), or long alphanumeric strings near sensitive keywords
    secret_patterns = [
        r'(?:api_key|apikey|secret|token|password)\s*[:=]\s*["\']?(sk-[a-zA-Z0-9]+)',
        r'(?:api_key|apikey|secret|token|password)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})',
    ]

    found = False
    for pattern in secret_patterns:
        # Skip lines with ${} env var references
        for line_no, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if "${" in stripped:
                continue
            if re.search(pattern, stripped, re.IGNORECASE):
                if not found:
                    _check("secrets", "WARN", f"config.yaml 中发现疑似明文密钥 (第 {line_no} 行)")
                    found = True
                break
        if found:
            break

    if not found:
        _check("secrets", "PASS", "")
