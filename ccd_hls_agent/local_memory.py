from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import agent_home, utc_now


def default_memory_path() -> Path:
    return agent_home() / "memory" / "hls_memory.sqlite"


@dataclass(frozen=True)
class MemoryHit:
    memory_id: int
    case_family: str
    failure_class: str
    error_signature: str
    attempted_fix: str
    verified: bool
    reuse_score: float


@dataclass(frozen=True)
class ActionMemoryHit:
    memory_id: int
    case_family: str
    failure_class: str
    error_signature: str
    diagnosis_claim: str
    action_id: str
    action_params: dict[str, Any]
    polarity: str
    verified: bool
    reuse_score: float
    artifact_uri: str


class HLSLocalMemory:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_memory_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hls_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    case_name TEXT,
                    case_family TEXT,
                    top_signature TEXT,
                    array_shapes TEXT,
                    failure_class TEXT,
                    error_signature TEXT,
                    attempted_fix TEXT,
                    tool_result_before TEXT,
                    tool_result_after TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    reuse_score REAL NOT NULL DEFAULT 0.0,
                    artifact_uri TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hls_memory_failure ON hls_memory(failure_class, case_family, verified)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hls_action_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    case_name TEXT,
                    case_family TEXT,
                    diagnosis_claim TEXT,
                    failure_class TEXT,
                    error_signature TEXT,
                    action_id TEXT,
                    action_params TEXT,
                    polarity TEXT,
                    before_flags TEXT,
                    after_flags TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    reuse_conditions TEXT,
                    anti_reuse_conditions TEXT,
                    artifact_uri TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hls_action_memory_failure ON hls_action_memory(failure_class, action_id, polarity)")

    def search(self, *, case_family: str, failure_class: str, error_signature: str, limit: int = 3) -> list[MemoryHit]:
        if not failure_class:
            return []
        terms = [term for term in error_signature.lower().split() if len(term) > 3][:12]
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, case_family, failure_class, error_signature, attempted_fix, verified, reuse_score
                FROM hls_memory
                WHERE failure_class = ?
                ORDER BY verified DESC, reuse_score DESC, id DESC
                LIMIT 50
                """,
                (failure_class,),
            ).fetchall()
        hits: list[MemoryHit] = []
        for row in rows:
            score = float(row[6] or 0.0)
            if row[1] == case_family:
                score += 1.0
            text = str(row[3] or "").lower()
            score += sum(0.1 for term in terms if term in text)
            hits.append(
                MemoryHit(
                    memory_id=int(row[0]),
                    case_family=str(row[1] or ""),
                    failure_class=str(row[2] or ""),
                    error_signature=str(row[3] or ""),
                    attempted_fix=str(row[4] or ""),
                    verified=bool(row[5]),
                    reuse_score=score,
                )
            )
        return sorted(hits, key=lambda hit: hit.reuse_score, reverse=True)[:limit]

    def add_event(
        self,
        *,
        case_name: str,
        case_family: str,
        failure_class: str,
        error_signature: str,
        attempted_fix: str,
        tool_result_before: dict[str, Any],
        tool_result_after: dict[str, Any] | None,
        verified: bool,
        artifact_uri: str,
        top_signature: str = "",
        array_shapes: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO hls_memory (
                    created_at, case_name, case_family, top_signature, array_shapes,
                    failure_class, error_signature, attempted_fix, tool_result_before,
                    tool_result_after, verified, reuse_score, artifact_uri
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    case_name,
                    case_family,
                    top_signature,
                    json.dumps(array_shapes or {}, ensure_ascii=False),
                    failure_class,
                    error_signature,
                    attempted_fix,
                    json.dumps(tool_result_before, ensure_ascii=False),
                    json.dumps(tool_result_after or {}, ensure_ascii=False),
                    1 if verified else 0,
                    1.0 if verified else -0.25,
                    artifact_uri,
                ),
            )

    def search_action_memory(
        self,
        *,
        case_family: str,
        failure_class: str,
        error_signature: str,
        action_ids: list[str] | None = None,
        limit: int = 6,
    ) -> list[ActionMemoryHit]:
        if not failure_class:
            return []
        terms = [term for term in error_signature.lower().split() if len(term) > 3][:12]
        params: list[Any] = [failure_class]
        action_filter = ""
        if action_ids:
            action_filter = " AND action_id IN (" + ",".join(["?"] * len(action_ids)) + ")"
            params.extend(action_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, case_family, failure_class, error_signature, diagnosis_claim,
                       action_id, action_params, polarity, verified, artifact_uri
                FROM hls_action_memory
                WHERE failure_class = ?{action_filter}
                ORDER BY verified DESC, id DESC
                LIMIT 100
                """,
                params,
            ).fetchall()
        hits: list[ActionMemoryHit] = []
        for row in rows:
            polarity = str(row[7] or "positive")
            score = 1.0 if polarity == "positive" else -1.0
            if bool(row[8]):
                score += 0.5 if polarity == "positive" else -0.5
            if row[1] == case_family:
                score += 0.5 if polarity == "positive" else -0.5
            text = str(row[3] or "").lower()
            overlap = sum(0.05 for term in terms if term in text)
            score += overlap if polarity == "positive" else -overlap
            try:
                action_params = json.loads(row[6] or "{}")
            except Exception:
                action_params = {}
            hits.append(
                ActionMemoryHit(
                    memory_id=int(row[0]),
                    case_family=str(row[1] or ""),
                    failure_class=str(row[2] or ""),
                    error_signature=str(row[3] or ""),
                    diagnosis_claim=str(row[4] or ""),
                    action_id=str(row[5] or ""),
                    action_params=action_params if isinstance(action_params, dict) else {},
                    polarity=polarity,
                    verified=bool(row[8]),
                    reuse_score=score,
                    artifact_uri=str(row[9] or ""),
                )
            )
        return sorted(hits, key=lambda hit: abs(hit.reuse_score), reverse=True)[:limit]

    def add_action_event(
        self,
        *,
        case_name: str,
        case_family: str,
        diagnosis_claim: str,
        failure_class: str,
        error_signature: str,
        action_id: str,
        action_params: dict[str, Any] | None,
        polarity: str,
        before_flags: dict[str, Any],
        after_flags: dict[str, Any] | None,
        verified: bool,
        reuse_conditions: dict[str, Any] | None,
        anti_reuse_conditions: dict[str, Any] | None,
        artifact_uri: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO hls_action_memory (
                    created_at, case_name, case_family, diagnosis_claim, failure_class,
                    error_signature, action_id, action_params, polarity, before_flags,
                    after_flags, verified, reuse_conditions, anti_reuse_conditions, artifact_uri
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    case_name,
                    case_family,
                    diagnosis_claim,
                    failure_class,
                    error_signature,
                    action_id,
                    json.dumps(action_params or {}, ensure_ascii=False),
                    polarity,
                    json.dumps(before_flags, ensure_ascii=False),
                    json.dumps(after_flags or {}, ensure_ascii=False),
                    1 if verified else 0,
                    json.dumps(reuse_conditions or {}, ensure_ascii=False),
                    json.dumps(anti_reuse_conditions or {}, ensure_ascii=False),
                    artifact_uri,
                ),
            )


def error_signature_from_capsule(capsule: dict[str, Any]) -> str:
    lines = [*map(str, capsule.get("signal_lines", [])[:8]), *map(str, capsule.get("key_errors", [])[:4])]
    return "\n".join(lines)[:2000]


def render_memory_capsule(hits: list[MemoryHit]) -> str:
    if not hits:
        return "- No verified local memory matched this failure."
    lines = []
    for hit in hits:
        status = "verified" if hit.verified else "negative/unverified"
        lines.append(
            f"- memory#{hit.memory_id} [{status}] class={hit.failure_class}, family={hit.case_family}, "
            f"fix={hit.attempted_fix[:240]}"
        )
    return "\n".join(lines)


def render_action_memory_capsule(hits: list[ActionMemoryHit]) -> str:
    if not hits:
        return "- No verified action memory matched this diagnosis."
    lines = []
    for hit in hits[:6]:
        status = "positive" if hit.polarity == "positive" else "negative"
        verified = "verified" if hit.verified else "unverified"
        lines.append(
            f"- action_memory#{hit.memory_id} [{status}/{verified}] class={hit.failure_class}, "
            f"family={hit.case_family}, action={hit.action_id}, score={hit.reuse_score:.2f}, "
            f"claim={hit.diagnosis_claim[:180]}"
        )
    return "\n".join(lines)
