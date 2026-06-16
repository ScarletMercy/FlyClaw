"""Gateway local-binding + auto token.

- host hard-locked to loopback (code constant, not config)
- auth token auto-generated when none is configured, persisted to a STANDALONE
  file (not config.yaml — writing config.yaml would clobber ${ENV} placeholders
  via the full-dump save_config). The token is injected back into the in-memory
  config so every auth-check site reading config.gateway.auth_token sees it.
- idempotent: a persisted token is reused, never regenerated.
"""

from src.config import AppConfig
from src.gateway import GATEWAY_HOST, ensure_gateway_token


def test_gateway_host_is_hardcoded_loopback():
    """host is a code constant, never a user-settable value."""
    assert GATEWAY_HOST == "127.0.0.1"


def _config_with_token(token: str) -> AppConfig:
    cfg = AppConfig()
    cfg.gateway.auth_token = token
    return cfg


def test_generate_persists_to_standalone_file_not_config_yaml(tmp_path):
    """空 token → 生成并写独立文件;config.yaml 不被 save_config 全量改(占位符零风险)。"""
    cfg = _config_with_token("")
    token_file = tmp_path / "gateway_token"

    token = ensure_gateway_token(cfg, token_file=token_file)

    assert token and len(token) >= 32, "必须生成强 token"
    assert cfg.gateway.auth_token == token, "必须注入回内存(下游认证点读 config 字段)"
    assert token_file.exists(), "必须持久化到独立文件"
    assert token_file.read_text(encoding="utf-8").strip() == token


def test_persisted_token_is_reused_not_regenerated(tmp_path):
    """独立文件已存在 → 复用,不重新生成(重启后 token 不变)。"""
    cfg = _config_with_token("")
    token_file = tmp_path / "gateway_token"
    token_file.write_text("previously-generated-token", encoding="utf-8")

    token = ensure_gateway_token(cfg, token_file=token_file)

    assert token == "previously-generated-token"
    assert cfg.gateway.auth_token == "previously-generated-token"


def test_explicit_config_token_wins_and_file_not_touched(tmp_path):
    """config 手动设了 token → 优先用,不创建独立文件(向后兼容手动配置)。"""
    cfg = _config_with_token("user-set-secret")
    token_file = tmp_path / "gateway_token"

    token = ensure_gateway_token(cfg, token_file=token_file)

    assert token == "user-set-secret"
    assert not token_file.exists(), "config 已有 token 时不应碰独立文件"
