"""Single source of truth for credential detection regex patterns.

Used by:
- redact.py (runtime output sanitization)
- guard.py (pre-install security scanning)

guard.py must catch everything redact.py catches (superset, never subset).
"""

from __future__ import annotations

from typing import NamedTuple


class CredentialPattern(NamedTuple):
    pattern: str
    name: str
    guard_id: str
    guard_description: str


CREDENTIAL_PATTERNS: list[CredentialPattern] = [
    CredentialPattern(
        r"sk-[A-Za-z0-9_-]{10,}",
        "openai_key",
        "openai_key_leaked",
        "possible OpenAI / Anthropic API key in skill content",
    ),
    CredentialPattern(
        r"sk_[A-Za-z0-9_]{10,}",
        "elevenlabs_key",
        "elevenlabs_key_leaked",
        "possible API key (sk_ prefix) in skill content",
    ),
    CredentialPattern(
        r"ghp_[A-Za-z0-9]{10,}",
        "github_pat",
        "github_pat_leaked",
        "possible GitHub personal access token in skill content",
    ),
    CredentialPattern(
        r"github_pat_[A-Za-z0-9_]{10,}",
        "github_fine_grained_pat",
        "github_fine_grained_pat_leaked",
        "possible GitHub fine-grained PAT in skill content",
    ),
    CredentialPattern(
        r"gho_[A-Za-z0-9]{10,}",
        "github_oauth",
        "github_oauth_token_leaked",
        "possible GitHub OAuth token in skill content",
    ),
    CredentialPattern(
        r"AIza[A-Za-z0-9_-]{30,}",
        "google_api_key",
        "google_api_key_leaked",
        "possible Google API key in skill content",
    ),
    CredentialPattern(
        r"xox[baprs]-[A-Za-z0-9-]{10,}",
        "slack_token",
        "slack_token_leaked",
        "possible Slack token in skill content",
    ),
    CredentialPattern(
        r"AKIA[A-Z0-9]{16}",
        "aws_access_key",
        "aws_access_key_leaked",
        "AWS access key ID in skill content",
    ),
    CredentialPattern(
        r"sk_live_[A-Za-z0-9]{10,}",
        "stripe_live_key",
        "stripe_live_key_leaked",
        "possible Stripe live secret key in skill content",
    ),
    CredentialPattern(
        r"sk_test_[A-Za-z0-9]{10,}",
        "stripe_test_key",
        "stripe_test_key_leaked",
        "possible Stripe test secret key in skill content",
    ),
    CredentialPattern(
        r"SG\.[A-Za-z0-9_-]{10,}",
        "sendgrid_key",
        "sendgrid_key_leaked",
        "possible SendGrid API key in skill content",
    ),
    CredentialPattern(
        r"hf_[A-Za-z0-9]{10,}",
        "huggingface_token",
        "huggingface_token_leaked",
        "possible HuggingFace token in skill content",
    ),
    CredentialPattern(
        r"gsk_[A-Za-z0-9]{10,}",
        "groq_key",
        "groq_key_leaked",
        "possible Groq API key in skill content",
    ),
    CredentialPattern(
        r"tvly-[A-Za-z0-9]{10,}",
        "tavily_key",
        "tavily_key_leaked",
        "possible Tavily API key in skill content",
    ),
    CredentialPattern(
        r"fal_[A-Za-z0-9_-]{10,}",
        "fal_key",
        "fal_key_leaked",
        "possible Fal.ai API key in skill content",
    ),
    CredentialPattern(
        r"pplx-[A-Za-z0-9]{10,}",
        "perplexity_key",
        "perplexity_key_leaked",
        "possible Perplexity API key in skill content",
    ),
    CredentialPattern(
        r"r8_[A-Za-z0-9]{10,}",
        "replicate_token",
        "replicate_token_leaked",
        "possible Replicate API token in skill content",
    ),
    CredentialPattern(
        r"npm_[A-Za-z0-9]{10,}",
        "npm_token",
        "npm_token_leaked",
        "possible npm access token in skill content",
    ),
]
