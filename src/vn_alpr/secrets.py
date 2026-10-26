"""Redact credentials before they reach logs, notebooks, or zip files."""

from __future__ import annotations

import re
from pathlib import Path

SECRET_LINE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+"
)
PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}")


def find_secrets(text: str) -> list[str]:
    hits = []
    for pattern in (SECRET_LINE, PRIVATE_KEY, BEARER):
        hits.extend(match.group(0) for match in pattern.finditer(text))
    return hits


def redact(text: str) -> str:
    text = PRIVATE_KEY.sub("[REDACTED_PRIVATE_KEY]", text)
    text = BEARER.sub("Bearer [REDACTED]", text)
    text = SECRET_LINE.sub(lambda match: match.group(0).split("=")[0].split(":")[0] + "=[REDACTED]", text)
    return text


def path_is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    if name in {".env", "credentials.json", "id_rsa", "id_ed25519"}:
        return True
    return "api_key" in name or name.endswith(".pem") or "secret" in name


def scan_tree(root: Path) -> list[str]:
    problems = []
    if not root.exists():
        return problems
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path_is_sensitive(path):
            problems.append(str(path))
            continue
        if path.suffix.lower() not in {".json", ".yaml", ".yml", ".md", ".txt", ".log", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if find_secrets(text):
            problems.append(str(path))
    return problems
