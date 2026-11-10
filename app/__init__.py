"""Assessment-1: bank customer feedback -> structured JSON."""

import os


def _load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from .env (real environment variables win)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except FileNotFoundError:
        pass


_load_env()  # runs before app.llm reads its settings
