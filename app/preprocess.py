"""Stage 1: clean, mask PII, flag risks, and truncate raw customer text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

TRUNCATION_MARKER = " [TRUNCATED]"


class EmptyInputError(ValueError):
    """The feedback contains no usable text."""


@dataclass
class Preprocessed:
    text: str  # cleaned + masked + truncated: the only form that ever leaves the process
    original_length: int
    pii_counts: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- PII masking
# Order matters: more specific / longer patterns first.
_SECRET = re.compile(r"(?i)\b(otp|cvv|cvc|pin|password|passcode)(\s*(?:is|:|-|=)?\s*)(\d{3,8})\b")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_UPI = re.compile(r"(?<![\w@.])[A-Za-z0-9._-]{2,}@[A-Za-z]{2,}\b(?!\.[A-Za-z])")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")  # 13-19 digits, optional separators
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)")
_ACCOUNT_LABELLED = re.compile(
    r"(?i)\b((?:a/c|acct|account)\s*(?:no\.?|number|num|#)?\s*(?:is\s*)?[:\-]?\s*)(\d[\d ]{4,}\d)"
)
_PHONE = re.compile(
    r"(?<![\d\w])(?:"
    r"(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}"  # India mobile
    r"|\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}"  # other international
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"  # NANP style
    r")(?!\d)"
)
_LONG_DIGITS = re.compile(r"(?<![\d.,])\d{9,18}(?![\d])")  # leftover account-like numbers


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def sub(pattern: re.Pattern[str], repl, label: str, s: str) -> str:
        def _r(m: re.Match[str]) -> str:
            counts[label] = counts.get(label, 0) + 1
            return repl(m) if callable(repl) else repl

        return pattern.sub(_r, s)

    text = sub(_SECRET, lambda m: f"{m.group(1)}{m.group(2)}[SECRET]", "secret", text)
    text = sub(_EMAIL, "[EMAIL]", "email", text)
    text = sub(_UPI, "[UPI_ID]", "upi_id", text)
    text = sub(_PAN, "[PAN]", "pan", text)
    text = sub(_CARD, "[CARD_NUMBER]", "card_number", text)
    text = sub(_AADHAAR, "[AADHAAR]", "aadhaar", text)
    text = sub(_ACCOUNT_LABELLED, lambda m: f"{m.group(1)}[ACCOUNT_NUMBER]", "account_number", text)
    text = sub(_PHONE, "[PHONE]", "phone", text)
    text = sub(_LONG_DIGITS, "[ACCOUNT_NUMBER]", "account_number", text)
    return text, counts


# ----------------------------------------------------------- injection signals
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+)*(?:previous|prior|above|earlier|preceding)\b",
    r"disregard\s+(?:all\s+|any\s+|the\s+|your\s+)*(?:previous|prior|above|earlier|rules|instructions)\b",
    r"forget\s+(?:everything|all\s+(?:previous|prior|your)|your\s+instructions)",
    r"(?:reveal|show|print|repeat|output|leak)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+|hidden\s+)?(?:prompt|instructions|rules)",
    r"\b(?:system|developer)\s+(?:prompt|message)\b",
    r"\bnew\s+instructions?\b",
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(?:to\s+be|you\s+are)\b",
    r"\bjailbreak\b",
    r"\b(?:mark|set|classify|rate|label|output|respond\s+with|return)\b[^.\n]{0,60}\b(?:urgency|sentiment|category|escalate)\b",
    r"[\"']?(?:sentiment|urgency|escalate_to_human)[\"']?\s*:\s*[\"']?\w",
    r"</?\s*(?:system|assistant|customer_feedback)\s*>|\[/?inst\]|^\s*#{2,}\s*(?:system|instruction)",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁠﻿]")


def detect_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def _non_latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_latin = sum(1 for c in letters if not unicodedata.name(c, "").startswith("LATIN"))
    return non_latin / len(letters)


def normalize(text: str) -> str:
    """Unicode-normalise (defeats full-width / homoglyph tricks) and drop control characters."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def tidy(text: str) -> str:
    """Strip markup (which also removes forged prompt delimiters) and collapse whitespace."""
    text = _HTML_TAG.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean(text: str) -> str:
    return tidy(normalize(text))


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.8:
        cut = cut[:last_space]
    return cut.rstrip() + TRUNCATION_MARKER, True


def preprocess(raw: str, max_chars: int = 4000) -> Preprocessed:
    if raw is None or not str(raw).strip():
        raise EmptyInputError("feedback text is empty")
    normalized = normalize(str(raw))
    # Detect on the pre-strip text so forged <system> tags are seen, and on the whole message
    # so an injection placed after the truncation point is still flagged.
    injection = detect_injection(normalized)
    cleaned = tidy(normalized)
    if not cleaned:
        raise EmptyInputError("feedback text contains no usable characters")

    masked, counts = mask_pii(cleaned)
    flags: list[str] = []
    if counts:
        flags.append("pii_masked")
    if injection:
        flags.append("prompt_injection_suspected")
    if _non_latin_ratio(masked) > 0.3:
        flags.append("non_english_suspected")

    final, was_truncated = truncate(masked, max_chars)
    if was_truncated:
        flags.append("truncated")
    return Preprocessed(text=final, original_length=len(str(raw)), pii_counts=counts, flags=flags)
