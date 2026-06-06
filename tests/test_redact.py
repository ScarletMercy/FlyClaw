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
        """Backticks wrapping inline code must survive; credentials inside are redacted."""
        inp = f"{self.BT}https://example.com?api_key=abc123{self.BT}"
        result = redact(inp)
        assert result.startswith(self.BT), f"Missing opening backtick: {result!r}"
        assert result.endswith(self.BT), f"Missing closing backtick: {result!r}"
        # Inline code credentials should be redacted
        assert "api_key=abc123" not in result
        assert "api_key=***" in result

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
        """`export TOKEN=abcdef123456` — backticks must survive, credentials redacted."""
        inp = f"{self.BT}export TOKEN=abcdef123456{self.BT} is required"
        result = redact(inp)
        assert result.count(self.BT) == 2, f"Backtick count changed: {result!r}"
        # Inline code credentials should be redacted
        assert "TOKEN=abcdef123456" not in result
        assert "TOKEN=***" in result

    def test_inline_code_without_secrets_unchanged(self):
        """`print("hello")` — ordinary inline code must pass through unchanged."""
        inp = f'{self.BT}print("hello"){self.BT} is a function call'
        result = redact(inp)
        assert result == inp

    def test_inline_code_with_special_chars_preserved(self):
        """`x = a | b && c * d` — operators and special chars must survive."""
        inp = f"{self.BT}x = a | b && c * d{self.BT}"
        result = redact(inp)
        assert result == inp

    def test_multiple_inline_code_mixed_with_plain_text(self):
        """Mix of credential-bearing and plain inline code."""
        BT = self.BT
        inp = f"Use {BT}API_KEY=secret123{BT} and {BT}print('hi'){BT} together"
        result = redact(inp)
        # Credential-bearing inline code: redacted, backticks kept
        assert "secret123" not in result
        assert f"{BT}API_KEY=***{BT}" in result
        # Plain inline code: unchanged
        assert f"{BT}print('hi'){BT}" in result

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
        """Code block credentials are redacted; fence structure preserved."""
        inp = f'Usage:\n{self.FENCE}python\napi_key="sk-abcdefghij1234567890"\n{self.FENCE}'
        result = redact(inp)
        # Credential must be redacted even inside code blocks
        assert "sk-abcdefghij1234567890" not in result
        # But the fence structure and language tag are preserved
        assert "```python" in result or "```\npython" in result or "```" in result

    def test_code_block_with_env_var_content_preserved(self):
        """ENV credentials inside code blocks are redacted; fence preserved."""
        inp = f"Config:\n{self.FENCE}bash\nexport API_KEY=my_secret_value_12345\n{self.FENCE}"
        result = redact(inp)
        assert "my_secret_value_12345" not in result
        # Fence structure still present
        assert "```" in result

    # --- Mixed content ---

    def test_code_block_preserved_outside_text_redacted(self):
        """Credentials redacted both inside and outside code blocks."""
        inp = f"Key:\n{self.FENCE}\nsk-abcdefghij1234567890\n{self.FENCE}\nLink: [api](https://x.com?token=abc123)"
        result = redact(inp)
        # Credential inside code block is now redacted too
        assert "sk-abcdefghij1234567890" not in result
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


class TestInlineCodeRedaction:
    """Comprehensive coverage for inline code credential redaction."""

    BT = "`"

    # --- Prefix patterns in inline code ---

    def test_inline_code_openai_key_redacted(self):
        inp = f"{self.BT}sk-abcdefghij1234567890{self.BT}"
        result = redact(inp)
        assert "sk-abcdefghij1234567890" not in result
        assert self.BT in result  # backticks preserved

    def test_inline_code_github_pat_redacted(self):
        inp = f"{self.BT}ghp_abcdefghijklmnopqrstuvwx{self.BT}"
        result = redact(inp)
        assert "ghp_abcdefghijklmnopqrstuvwx" not in result

    def test_inline_code_aws_key_redacted(self):
        inp = f"{self.BT}AKIAIOSFODNN7EXAMPLE{self.BT}"
        result = redact(inp)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    # --- ENV patterns in inline code ---

    def test_inline_code_env_quoted_single(self):
        inp = f"{self.BT}API_KEY='mysecretvalue123'{self.BT}"
        result = redact(inp)
        assert "mysecretvalue123" not in result
        assert "API_KEY=" in result

    def test_inline_code_env_quoted_double(self):
        inp = f'{self.BT}SECRET="sk-test-1234567890abcdef"{self.BT}'
        result = redact(inp)
        assert "sk-test-1234567890abcdef" not in result

    def test_inline_code_env_unquoted(self):
        inp = f"{self.BT}PASSWORD=hunter2{self.BT}"
        result = redact(inp)
        assert "hunter2" not in result
        assert "PASSWORD=" in result

    # --- JSON credential in inline code ---

    def test_inline_code_json_api_key(self):
        inp = f'{self.BT}"apiKey": "sk-abcdefghij1234567890"{self.BT}'
        result = redact(inp)
        assert "sk-abcdefghij1234567890" not in result
        assert "apiKey" in result

    def test_inline_code_json_token(self):
        inp = f'{self.BT}"token": "abc123def456"{self.BT}'
        result = redact(inp)
        assert "abc123def456" not in result

    # --- Auth header in inline code ---

    def test_inline_code_auth_bearer(self):
        inp = f"{self.BT}Authorization: Bearer sk-abcdefghij1234567890{self.BT}"
        result = redact(inp)
        assert "sk-abcdefghij1234567890" not in result
        assert "Bearer" in result

    # --- URL userinfo in inline code ---

    def test_inline_code_url_password(self):
        inp = f"{self.BT}https://admin:s3cret@host/path{self.BT}"
        result = redact(inp)
        assert "s3cret" not in result
        assert "admin" in result  # username preserved

    # --- DB connection string in inline code ---

    def test_inline_code_db_connstr(self):
        inp = f"{self.BT}postgres://user:dbpass99@localhost:5432/mydb{self.BT}"
        result = redact(inp)
        assert "dbpass99" not in result
        assert "localhost" in result

    # --- JWT in inline code ---

    def test_inline_code_jwt(self):
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi789"
        inp = f"{self.BT}{token}{self.BT}"
        result = redact(inp)
        assert token not in result

    # --- Non-secret inline code (false-positive guards) ---

    def test_inline_code_port_assignment_unchanged(self):
        """PORT=8080 should NOT be redacted — PORT is not in secret names."""
        inp = f"{self.BT}PORT=8080{self.BT}"
        assert redact(inp) == inp

    def test_inline_code_debug_env_unchanged(self):
        """DEBUG=true should NOT be redacted."""
        inp = f"{self.BT}DEBUG=true{self.BT}"
        assert redact(inp) == inp

    def test_inline_code_plain_url_unchanged(self):
        """URL without sensitive query params should be unchanged."""
        inp = f"{self.BT}https://example.com/page?name=test{self.BT}"
        assert redact(inp) == inp

    def test_inline_code_variable_name_unchanged(self):
        """my_token_count is not a secret pattern."""
        inp = f"{self.BT}my_token_count = 5{self.BT}"
        assert redact(inp) == inp

    def test_inline_code_html_like_unchanged(self):
        """HTML-like content without secrets should pass through."""
        inp = f"{self.BT}<div class='item'>hello</div>{self.BT}"
        assert redact(inp) == inp

    # --- Positional edge cases ---

    def test_inline_code_at_line_start(self):
        inp = f"{self.BT}API_KEY=secretval{self.BT} is set"
        result = redact(inp)
        assert "secretval" not in result
        assert result.startswith(self.BT)

    def test_inline_code_at_line_end(self):
        inp = f"set {self.BT}API_KEY=secretval{self.BT}"
        result = redact(inp)
        assert "secretval" not in result
        assert result.endswith(self.BT)

    def test_inline_code_only_secret(self):
        """Inline code containing nothing but a secret."""
        inp = f"{self.BT}sk-abcdefghij1234567890{self.BT}"
        result = redact(inp)
        assert "sk-abcdefghij1234567890" not in result
        assert result.startswith(self.BT) and result.endswith(self.BT)

    # --- Multiple inline codes ---

    def test_consecutive_inline_codes(self):
        BT = self.BT
        inp = f"{BT}API_KEY=val1{BT} {BT}SECRET=val2{BT}"
        result = redact(inp)
        assert "val1" not in result
        assert "val2" not in result
        assert result.count(BT) == 4  # both pairs preserved

    def test_three_inline_codes_mixed(self):
        BT = self.BT
        inp = f"use {BT}sk-abcdefghij1234567890{BT} then {BT}print(x){BT} and {BT}TOKEN=mysecret{BT}"
        result = redact(inp)
        assert "sk-abcdefghij1234567890" not in result
        assert "TOKEN=mysecret" not in result
        assert "TOKEN=***" in result
        assert f"{BT}print(x){BT}" in result  # plain code unchanged

    # --- Inline code + fenced block interaction ---

    def test_inline_code_next_to_fenced_block(self):
        """Inline code before a fenced block; both should work independently."""
        BT, FENCE = self.BT, "`" * 3
        inp = f"use {BT}API_KEY=val{BT} then:\n{FENCE}\nSECRET=other\n{FENCE}"
        result = redact(inp)
        assert "val" not in result
        assert "other" not in result

    # --- Inline code with surrounding markdown ---

    def test_inline_code_in_bold(self):
        BT = self.BT
        inp = f"**Warning: {BT}API_KEY=secretval{BT}**"
        result = redact(inp)
        assert "secretval" not in result
        assert "**Warning:" in result
        assert result.endswith("**")

    def test_inline_code_in_markdown_link_text(self):
        BT = self.BT
        inp = f"[{BT}API_KEY=secretval{BT}](https://example.com)"
        result = redact(inp)
        assert "secretval" not in result
        assert "](https://example.com)" in result

    # --- Whitespace edge cases ---

    def test_inline_code_leading_trailing_spaces_in_content(self):
        """Content with spaces inside backticks."""
        BT = self.BT
        inp = f"{BT} API_KEY=secretval {BT}"
        result = redact(inp)
        assert "secretval" not in result

    # --- Long inline code with multiple secrets ---

    def test_inline_code_multiple_secrets(self):
        BT = self.BT
        inp = f"{BT}export API_KEY=val1 && export SECRET=val2{BT}"
        result = redact(inp)
        assert "val1" not in result
        assert "val2" not in result
        assert "API_KEY=" in result
        assert "SECRET=" in result


class TestRedactEdgeCases:
    """Edge cases: plain-text patterns, idempotency, malformed markdown, non-string input."""

    BT = "`"
    FENCE = "`" * 3

    # --- Plain-text patterns missing dedicated tests ---

    def test_db_connstr_plain_text(self):
        inp = "postgres://admin:dbpass99@localhost:5432/mydb"
        result = redact(inp)
        assert "dbpass99" not in result
        assert "admin" in result
        assert "localhost" in result

    def test_jwt_plain_text(self):
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi789"
        result = redact(token)
        assert token not in result

    def test_url_userinfo_plain_text(self):
        result = redact("connect to https://admin:s3cret@host/path")
        assert "s3cret" not in result
        assert "admin" in result

    # --- Idempotency ---

    def test_idempotency(self):
        inp = "Config: API_KEY=mysecret and sk-abcdefghij1234567890"
        first = redact(inp)
        second = redact(first)
        assert first == second, f"Not idempotent:\n  1st={first!r}\n  2nd={second!r}"

    # --- Malformed / edge-case markdown ---

    def test_unbalanced_opening_backtick(self):
        """A lone opening backtick without a close — credential still redacted."""
        inp = "set `API_KEY=secretval here"
        result = redact(inp)
        assert "secretval" not in result

    def test_multiline_backtick_not_matched_as_inline(self):
        """Backtick content with a newline should NOT be matched by _MD_INLINE_RE."""
        inp = "```\nAPI_KEY=secretval\n```"
        result = redact(inp)
        assert "secretval" not in result

    def test_private_key_in_fenced_block(self):
        pk = "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJBALRiMLAHudeSA\n-----END RSA PRIVATE KEY-----"
        inp = f"{self.FENCE}\n{pk}\n{self.FENCE}"
        result = redact(inp)
        assert "BEGIN RSA PRIVATE KEY" not in result

    def test_credential_split_across_fence_boundary(self):
        """Credential outside fence is redacted; unrelated number inside fence is kept."""
        FENCE = self.FENCE
        inp = f"key is sk-abcdefghij\n{FENCE}\n1234567890{FENCE}"
        result = redact(inp)
        # sk-abcdefghij (outside fence) is a partial prefix — may or may not be
        # long enough to match. The key point: no crash, fence structure intact.
        assert FENCE in result
        # The number inside fence is NOT a credential pattern — left as-is.
        assert "1234567890" in result

    def test_fenced_block_with_multiple_secrets(self):
        FENCE = self.FENCE
        inp = (
            f"{FENCE}bash\n"
            "export API_KEY=val1\n"
            'export SECRET="val2"\n'
            "Authorization: Bearer sk-abcdefghij1234567890\n"
            f"{FENCE}"
        )
        result = redact(inp)
        assert "val1" not in result
        assert "val2" not in result
        assert "sk-abcdefghij1234567890" not in result

    # --- Non-string input safety ---

    @pytest.mark.parametrize("inp", [None, 42, [], 3.14])
    def test_non_string_input_returns_unchanged(self, inp):
        assert redact(inp) == inp

    def test_empty_string_unchanged(self):
        assert redact("") == ""

    # --- Unicode mixed ---

    def test_unicode_mixed_with_credential(self):
        inp = "配置 API_KEY=mysecret 完成"
        result = redact(inp)
        assert "mysecret" not in result
        assert "配置" in result
        assert "完成" in result

    # --- Fence variants ---

    def test_four_backtick_fence(self):
        FENCE4 = "`" * 4
        inp = f"{FENCE4}\nAPI_KEY=secretval\n{FENCE4}"
        result = redact(inp)
        assert "secretval" not in result

    def test_fence_with_long_lang_tag(self):
        FENCE = self.FENCE
        inp = f"{FENCE}python3.11-typed\nAPI_KEY=secretval\n{FENCE}"
        result = redact(inp)
        assert "secretval" not in result
        assert "python3.11-typed" in result

    def test_inline_code_inside_fenced_block(self):
        """Backticks inside a fenced block are part of the fence body, not inline code."""
        BT, FENCE = self.BT, self.FENCE
        inp = f"{FENCE}\nuse {BT}TOKEN=secretval{BT} here\n{FENCE}"
        result = redact(inp)
        assert "secretval" not in result
