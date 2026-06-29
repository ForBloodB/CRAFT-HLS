from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def json_loads(data: str | None, default: Any = None) -> Any:
    if data in (None, ""):
        return default
    return json.loads(data)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def agent_home() -> Path:
    return ensure_dir(Path(os.environ.get("HLS_AGENT_HOME", ".hls_agent")).resolve())


def which_status(command: str) -> dict[str, Any]:
    found = shutil.which(command)
    return {
        "name": command,
        "available": found is not None,
        "path": found,
    }


def estimate_tokens(text: str) -> int:
    # Conservative, dependency-free estimate for code-heavy prompts.
    return max(1, int(len(text) / 3.6))


def fit_text_to_token_budget(text: str, token_budget: int, *, keep_tail_ratio: float = 0.45) -> str:
    if token_budget <= 0:
        return "[truncated: no token budget]"
    if estimate_tokens(text) <= token_budget:
        return text
    max_chars = max(80, int(token_budget * 3.2))
    if max_chars >= len(text):
        return text
    marker = "\n\n[truncated to fit local context]\n\n"
    for shrink in (1.0, 0.9, 0.8, 0.7, 0.6):
        available = max(40, int(max_chars * shrink) - len(marker))
        tail_chars = int(available * keep_tail_ratio)
        head_chars = max(20, available - tail_chars)
        omitted = len(text) - head_chars - tail_chars
        if omitted <= 0:
            candidate = text[:available]
        else:
            candidate = f"{text[:head_chars]}{marker}{text[-tail_chars:]}"
        if estimate_tokens(candidate) <= token_budget:
            return candidate
    return text[: max(40, int(token_budget * 2.8))]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return default
