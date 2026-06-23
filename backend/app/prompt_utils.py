"""
Shared utilities for prompt safety and input sanitization.
Prevents prompt injection attacks across all AI clients.
"""

import re
from typing import Optional

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions?",
    r"disregard\s+(previous|prior|above|all)\s+instructions?",
    r"forget\s+(previous|prior|above|all)\s+instructions?",
    r"override\s+(system|safety|previous|all)\s*(instructions?|prompt|rules?)?",
    r"new\s+instructions?\s*:",
    r"updated?\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"you\s+are\s+now\s+(a|an)",
    r"from\s+now\s+on\s+(you\s+are|act|behave)",
    r"roleplay\s+as",
    r"pretend\s+(to\s+be|you\s+are)",
    r"jailbreak",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"prompt\s+injection",
    r"escape\s+(the\s+)?(context|sandbox|restriction)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(are|were)\s+your\s+(original\s+)?instructions",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

MAX_QUERY_LENGTH = 500


def sanitize_user_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Sanitize user input to prevent prompt injection attacks.

    Raises ValueError if empty or contains injection patterns.
    """
    if not text or not text.strip():
        raise ValueError("Query cannot be empty")

    text = text.strip()[:max_length]

    if _INJECTION_RE.search(text):
        raise ValueError(
            "Your query contains patterns that cannot be processed. "
            "Please rephrase your research question."
        )

    # Strip control characters (keep newline and tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def format_injection_boundary(content: str, label: str) -> str:
    """
    Wrap untrusted content in explicit boundary markers so the model treats
    everything inside as data, not instructions.
    """
    return f"[{label}_START]\n{content}\n[{label}_END]"
