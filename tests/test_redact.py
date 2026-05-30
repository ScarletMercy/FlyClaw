import re

from src.security.credential_patterns import CREDENTIAL_PATTERNS
from src.security.redact import _PREFIX_PATTERNS, redact


class TestRedactCredentialPatterns:
    def test_prefix_patterns_from_shared_module(self):
        assert _PREFIX_PATTERNS == [cp.pattern for cp in CREDENTIAL_PATTERNS]

    def test_all_prefix_patterns_compile(self):
        for p in _PREFIX_PATTERNS:
            re.compile(p)

    def test_redact_openai_key(self):
        assert redact("key=sk-abc1234567890") == "key=***"

    def test_redact_openai_key_with_underscore(self):
        assert redact("key=sk-abc_def123456") == "key=***"

    def test_redact_short_key_10chars(self):
        assert redact("key=sk-abcdefghij") == "key=***"

    def test_redact_github_pat(self):
        result = redact("token=ghp_abcdefghijklmnopqrstuvwx")
        assert "ghp_abcdefghijklmnopqrstuvwx" not in result

    def test_redact_aws_key(self):
        result = redact("AWS AKIAIOSFODNN7EXAMPLE here")
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_redact_private_key_block(self):
        result = redact("-----BEGIN RSA PRIVATE KEY-----\nblah\n-----END RSA PRIVATE KEY-----")
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result
        assert "[REDACTED PRIVATE KEY]" in result
