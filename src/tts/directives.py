"""TTS directive parser.

Parses [[tts:text]] and [[tts:provider=name|text]] directives from agent output.
Follows the pattern from the original openclaw src/tts/directives.ts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TtsDirective:
    text: str
    provider: Optional[str] = None


_DIRECTIVE_RE = re.compile(r"\[\[tts:(.*?)\]\]", re.DOTALL)


def parse_tts_directives(content: str) -> list[TtsDirective]:
    """Extract [[tts:...]] directives from agent text.

    Supports formats:
    - [[tts:Hello world]] → TtsDirective(text="Hello world")
    - [[tts:provider=openai|Hello world]] → TtsDirective(text="Hello world", provider="openai")
    """
    directives = []
    for match in _DIRECTIVE_RE.finditer(content):
        inner = match.group(1).strip()
        provider = None
        text = inner

        # Check for provider= prefix: [[tts:provider=name|text]]
        if "|" in inner:
            parts = inner.split("|", 1)
            prefix = parts[0].strip()
            if prefix.startswith("provider="):
                provider = prefix[len("provider="):].strip()
                text = parts[1].strip()

        if text:
            directives.append(TtsDirective(text=text, provider=provider))

    return directives


def strip_tts_directives(content: str) -> str:
    """Remove all [[tts:...]] tags from text for display."""
    return _DIRECTIVE_RE.sub("", content).strip()
