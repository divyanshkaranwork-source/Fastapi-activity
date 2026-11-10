"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when an environment variable has an invalid value."""


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_num(name: str, default: float, cast, minimum: float | None = None):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a {cast.__name__}, got {raw!r}") from None
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_output_tokens: int = 500
    request_timeout: float = 30.0
    max_retries: int = 3  # transport-level attempts per LLM call
    max_input_chars: int = 4000  # ~1k tokens; longer feedback is truncated
    max_batch_rows: int = 200
    max_upload_bytes: int = 2_000_000
    batch_workers: int = 5
    confidence_threshold: float = 0.7
    db_path: str = "data/assessment1.db"
    service_api_key: str = ""  # if set, API callers must send it as X-API-Key
    correction_examples: int = 3  # human corrections reused as few-shot examples
    price_input_per_1m: float = 0.40  # gpt-4.1-mini list price, USD / 1M input tokens
    price_output_per_1m: float = 1.60  # USD / 1M output tokens

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env_str("LLM_PROVIDER", "openai").lower()
        if provider not in {"openai", "mock"}:
            raise ConfigError(f"LLM_PROVIDER must be 'openai' or 'mock', got {provider!r}")
        return cls(
            provider=provider,
            api_key=_env_str("OPENAI_API_KEY", ""),
            model=_env_str("OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            temperature=_env_num("LLM_TEMPERATURE", 0.0, float, 0.0),
            max_output_tokens=_env_num("LLM_MAX_OUTPUT_TOKENS", 500, int, 50),
            request_timeout=_env_num("LLM_TIMEOUT_SECONDS", 30.0, float, 1.0),
            max_retries=_env_num("LLM_MAX_RETRIES", 3, int, 1),
            max_input_chars=_env_num("MAX_INPUT_CHARS", 4000, int, 100),
            max_batch_rows=_env_num("MAX_BATCH_ROWS", 200, int, 1),
            max_upload_bytes=_env_num("MAX_UPLOAD_BYTES", 2_000_000, int, 1000),
            batch_workers=_env_num("BATCH_WORKERS", 5, int, 1),
            confidence_threshold=_env_num("CONFIDENCE_THRESHOLD", 0.7, float, 0.0),
            db_path=_env_str("ASSESSMENT_DB", "data/assessment1.db"),
            service_api_key=_env_str("ASSESSMENT_API_KEY", ""),
            correction_examples=_env_num("CORRECTION_EXAMPLES", 3, int, 0),
            price_input_per_1m=_env_num("PRICE_INPUT_PER_1M", 0.40, float, 0.0),
            price_output_per_1m=_env_num("PRICE_OUTPUT_PER_1M", 1.60, float, 0.0),
        )


def get_settings() -> Settings:
    load_dotenv()
    return Settings.from_env()
