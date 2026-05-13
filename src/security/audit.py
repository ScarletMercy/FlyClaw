from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("myclaw.security")


def run_security_audit(config) -> dict[str, Any]:
    """Run startup security audit on configuration.

    Checks:
    1. Gateway auth - warns if gateway exposed (0.0.0.0) without auth_token
    2. Feishu whitelist - warns if dm_policy=open with empty allow_from
    3. Exec safety - info if approval_mode=off
    4. Data directory - ensures data/ exists or can be created
    5. Secret leakage - scans config.yaml for hardcoded secrets

    Returns:
        Dict with counts of passed, warnings, info, and list of issues
    """
    results = {"passed": 0, "warnings": 0, "info": 0, "issues": []}

    def _check(name: str, severity: str, message: str):
        if severity == "PASS":
            results["passed"] += 1
            logger.info("[security] PASS %s", name)
        elif severity == "WARN":
            results["warnings"] += 1
            logger.warning("[security] WARN %s: %s", name, message)
            results["issues"].append({"name": name, "severity": "WARN", "message": message})
        else:
            results["info"] += 1
            logger.info("[security] INFO %s", name)

    # Check 1: Gateway auth
    host = config.gateway.host
    token = config.gateway.auth_token
    if host in ("0.0.0.0", "", "::", "[::]", "0:0:0:0:0:0:0:0") and not token:
        _check("gateway-auth", "WARN", f"gateway exposed on {host} without auth_token")
    elif token:
        _check("gateway-auth", "PASS", "")

    # Check 2: Feishu whitelist
    if config.channels.feishu.enabled:
        if config.channels.feishu.dm_policy == "open" and not config.channels.feishu.allow_from:
            _check("feishu-dm", "WARN", "dm_policy=open with empty allow_from")
        else:
            _check("feishu-dm", "PASS", "")

    # Check 3: Exec approval
    if config.tools.exec.approval_mode == "off":
        _check("exec-approval", "INFO", "approval_mode=off")
    else:
        _check("exec-approval", "PASS", "")

    # Check 4: Data directory
    data_dir = Path("data")
    if data_dir.exists():
        _check("data-dir", "PASS", "")
    else:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            _check("data-dir", "PASS", "created data/")
        except Exception:
            _check("data-dir", "WARN", "cannot create data/ directory")

    # Check 5: Secret leakage in config.yaml
    _check_secrets(config, results, _check)

    # Check 6: RBAC / Auth
    if getattr(config, "auth", None) and config.auth.enabled:
        _check("rbac-enabled", "PASS", "")
        if not getattr(config, "owner_id", ""):
            _check("rbac-owner", "WARN", "owner_id not set — no user will have owner role automatically")
        else:
            _check("rbac-owner", "PASS", "")
    else:
        _check("rbac-enabled", "INFO", "auth/RBAC disabled")

    logger.info("[security] Audit complete: %d passed, %d warnings", results["passed"], results["warnings"])
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
                    _check("secrets", "WARN", f"potential hardcoded secrets in config.yaml (line {line_no})")
                    found = True
                break
        if found:
            break

    if not found:
        _check("secrets", "PASS", "")
