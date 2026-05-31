import re

import pytest

from src.security.credential_patterns import CREDENTIAL_PATTERNS
from src.security.redact import _PREFIX_PATTERNS, redact


def _min_sample(pattern: str) -> str:
    """Build the shortest string matching *pattern*.

    Handles the regex subset used in credential_patterns.py:
    literal chars, ``\\. ``, ``[charset]{N,}`` / ``[charset]{N}``.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "[":
            j = pattern.index("]", i)
            cls = pattern[i + 1 : j]
            i = j + 1
            # Parse quantifier
            if i < len(pattern) and pattern[i] == "{":
                k = pattern.index("}", i)
                spec = pattern[i + 1 : k]
                i = k + 1
                n = int(spec.rstrip(","))
                if spec.endswith(","):
                    n += 1  # minimum + 1 for safety
            else:
                n = 1
            # Pick a char that belongs to the class
            if "a-z" in cls:
                ch = "a"
            elif "A-Z" in cls:
                ch = "A"
            elif "0-9" in cls:
                ch = "0"
            elif cls:  # literal chars like "baprs"
                ch = cls[0]
            else:
                ch = "x"
            out.append(ch * n)
        elif pattern[i] == "\\":
            i += 1
            out.append(pattern[i])
            i += 1
        else:
            out.append(pattern[i])
            i += 1
    return "".join(out)


class TestRedactCredentialPatterns:
    def test_prefix_patterns_from_shared_module(self):
        assert _PREFIX_PATTERNS == [cp.pattern for cp in CREDENTIAL_PATTERNS]

    def test_all_prefix_patterns_compile(self):
        for p in _PREFIX_PATTERNS:
            re.compile(p)

    @pytest.mark.parametrize(
        "cp",
        CREDENTIAL_PATTERNS,
        ids=lambda c: c.name,
    )
    def test_each_credential_pattern_redacted(self, cp):
        """Every CREDENTIAL_PATTERNS entry must be caught by redact()."""
        sample = _min_sample(cp.pattern)
        result = redact(f"key={sample}")
        assert sample not in result, (
            f"{cp.name}: pattern {cp.pattern!r} sample {sample!r} not redacted"
        )

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
