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
        assert sample not in result, f"{cp.name}: pattern {cp.pattern!r} sample {sample!r} not redacted"

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


class TestMarkdownFormatPreservation:
    """Verify that redact() never breaks markdown syntax."""

    BT = "`"  # backtick
    FENCE = "`" * 3

    # --- URL query params ---

    def test_markdown_link_with_sensitive_query_paren_preserved(self):
        """Closing ) of markdown link must survive redaction."""
        result = redact("[doc](https://example.com?token=abc123)")
        assert result.endswith(")"), f"Expected ) at end, got: {result!r}"
        assert "token=***" in result

    def test_markdown_link_with_non_sensitive_query_unchanged(self):
        result = redact("[doc](https://example.com?page=1&size=10)")
        assert result == "[doc](https://example.com?page=1&size=10)"

    def test_inline_code_with_sensitive_url_backticks_preserved(self):
        """Backticks wrapping inline code with a URL must survive."""
        inp = f"{self.BT}https://example.com?api_key=abc123{self.BT}"
        result = redact(inp)
        assert result.startswith(self.BT), f"Missing opening backtick: {result!r}"
        assert result.endswith(self.BT), f"Missing closing backtick: {result!r}"
        # Inline code content is protected — should be unchanged
        assert "api_key=abc123" in result

    def test_shell_command_with_sensitive_url_paren_preserved(self):
        """$(curl url?token=abc) — the closing ) must survive."""
        result = redact("$(curl https://example.com?token=abc123)")
        assert result.endswith(")"), f"Expected ) at end, got: {result!r}"

    # --- ENV assignments ---

    def test_env_unquoted_with_trailing_paren(self):
        """API_KEY=secret) — ) after value must survive."""
        result = redact("Set API_KEY=mysecretvalue123) to continue")
        assert ")" in result, f") was consumed: {result!r}"

    def test_env_in_inline_code_backticks_preserved(self):
        """`export TOKEN=abcdef123456` — backticks must survive, content unchanged."""
        inp = f"{self.BT}export TOKEN=abcdef123456{self.BT} is required"
        result = redact(inp)
        assert result.count(self.BT) == 2, f"Backtick count changed: {result!r}"
        # Inline code content is protected
        assert "TOKEN=abcdef123456" in result

    def test_env_quoted_value_redacted(self):
        """API_KEY="sk-testvalue1234567890" — quoted value should still be redacted."""
        result = redact('API_KEY="sk-testvalue1234567890"')
        assert "sk-testvalue1234567890" not in result

    # --- Auth header ---

    def test_auth_header_with_trailing_paren(self):
        result = redact("Authorization: Bearer sk-abcdefghij1234567890)")
        assert ")" in result, f") was consumed: {result!r}"

    # --- Code blocks ---

    def test_code_block_with_api_key_content_preserved(self):
        """Code block content should not be modified."""
        inp = f'Usage:\n{self.FENCE}python\napi_key="sk-abcdefghij1234567890"\n{self.FENCE}'
        result = redact(inp)
        assert "sk-abcdefghij1234567890" in result, f"Code block content was modified: {result!r}"
        assert inp == result  # completely unchanged

    def test_code_block_with_env_var_content_preserved(self):
        inp = f"Config:\n{self.FENCE}bash\nexport API_KEY=my_secret_value_12345\n{self.FENCE}"
        result = redact(inp)
        assert "my_secret_value_12345" in result

    # --- Mixed content ---

    def test_code_block_preserved_outside_text_redacted(self):
        """Code block untouched, text outside still redacted."""
        inp = f"Key:\n{self.FENCE}\nsk-abcdefghij1234567890\n{self.FENCE}\nLink: [api](https://x.com?token=abc123)"
        result = redact(inp)
        # Code block preserved
        assert "sk-abcdefghij1234567890" in result
        # Outside text redacted
        assert "token=***" in result
        assert "abc123" not in result.split("token=")[1]
        # Markdown link intact
        assert result.count(")") >= 1

    # --- Safety ---

    def test_no_null_bytes_in_output(self):
        """No \\x00 should ever appear in redact output."""
        inp = (
            f'{self.FENCE}python\napi_key="sk-abcdefghij1234567890"\n{self.FENCE}\n'
            "[doc](https://x.com?token=abc123)\n"
            f"{self.BT}export TOKEN=secret{self.BT}"
        )
        result = redact(inp)
        assert "\0" not in result

    def test_no_secrets_text_unchanged(self):
        """Text with no secrets should pass through unchanged."""
        inp = "Hello world\n[link](https://example.com/page)\n`code here`"
        assert redact(inp) == inp
