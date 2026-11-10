"""Validate and parse an uploaded feedback CSV (shared by the API and the dashboard)."""

from __future__ import annotations

import csv
import io

TEXT_COLUMNS = ("feedback", "text", "message", "comment", "review", "content")


class CsvInputError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def parse_feedback_csv(raw: bytes, max_bytes: int, max_rows: int) -> list[str]:
    if len(raw) > max_bytes:
        raise CsvInputError(413, f"File too large (max {max_bytes} bytes)")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CsvInputError(400, "CSV must be UTF-8 encoded") from None

    try:
        reader = csv.DictReader(io.StringIO(content))
        fields = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
        column = next((fields[c] for c in TEXT_COLUMNS if c in fields), None)
        if column is None:
            found = ", ".join(sorted(f for f in fields if f)) or "none"
            raise CsvInputError(400, f"CSV needs a column named one of {', '.join(TEXT_COLUMNS)} (found: {found})")
        texts: list[str] = []
        for row in reader:
            texts.append((row.get(column) or "").strip())
            if len(texts) > max_rows:
                raise CsvInputError(413, f"Too many rows (max {max_rows})")
    except csv.Error as exc:
        raise CsvInputError(400, f"Could not parse CSV: {exc}") from None
    if not texts:
        raise CsvInputError(400, "CSV has no data rows")
    return texts
