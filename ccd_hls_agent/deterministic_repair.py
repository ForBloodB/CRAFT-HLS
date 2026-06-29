from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import read_text, write_text


@dataclass(frozen=True)
class DeterministicRepairResult:
    applied: bool
    repair_type: str
    message: str
    changed_files: list[str]


def _top_function(case_dir: Path) -> str:
    return read_text(case_dir / "top.txt").strip()


def _header_signature(header: Path, top: str) -> str | None:
    text = read_text(header)
    pattern = re.compile(
        rf"(?m)^\s*(?:extern\s+)?(?:void|int|float|double|bool|char|unsigned|signed|ap_\w+|[A-Za-z_][A-Za-z0-9_:<>]*_t|[A-Za-z_][A-Za-z0-9_:<>]*)"
        rf"[\w\s:<>,*&\[\]]*\b{re.escape(top)}\s*\([^;{{}}]*\)\s*;",
        re.S,
    )
    match = pattern.search(text)
    return " ".join(match.group(0).split()) if match else None


def _ensure_include(kernel: Path, header: Path) -> bool:
    text = read_text(kernel)
    include = f'#include "{header.name}"'
    if include in text:
        return False
    write_text(kernel, include + "\n" + text)
    return True


def _reject_disallowed_cpp(kernel: Path) -> list[str]:
    text = read_text(kernel)
    hits = []
    for pattern in [r"\bmalloc\s*\(", r"\bfree\s*\(", r"\bnew\s+", r"\bdelete\s+", r"\bstd::vector\b", r"\bfopen\s*\(", r"\bifstream\b", r"\bofstream\b"]:
        if re.search(pattern, text):
            hits.append(pattern)
    return hits


def _repair_signature_name(kernel: Path, header_signature: str, top: str) -> bool:
    text = read_text(kernel)
    if re.search(rf"\b{re.escape(top)}\s*\(", text):
        return False
    match = re.search(
        r"(?m)^(\s*(?:void|int|float|double|bool|char|unsigned|signed|ap_\w+|[A-Za-z_][A-Za-z0-9_:<>]*_t|[A-Za-z_][A-Za-z0-9_:<>]*)[\w\s:<>,*&\[\]]+)([A-Za-z_][A-Za-z0-9_]*)\s*(\([^;{}]*\)\s*\{)",
        text,
    )
    if not match:
        return False
    candidate = match.group(2)
    if candidate in {"if", "for", "while", "switch"}:
        return False
    fixed = text[: match.start(2)] + top + text[match.end(2) :]
    write_text(kernel, fixed)
    return True


def _repair_aes_state_column_misindex(kernel: Path, capsule: dict[str, Any]) -> bool:
    text = read_text(kernel)
    evidence = "\n".join([*map(str, capsule.get("signal_lines", [])), *map(str, capsule.get("key_errors", []))])
    if "uint8_t[4]" not in evidence and "unsigned char[4]" not in evidence:
        return False
    if not re.search(r"\bstate\s*\[\s*i\s*\]\s*\[\s*[0-3]\s*\]", text):
        return False
    fixed = re.sub(r"\bstate\s*\[\s*i\s*\]\s*\[\s*([0-3])\s*\]", r"state[\1][i]", text)
    if fixed == text:
        return False
    write_text(kernel, fixed)
    return True


def _stage_tb_data_files(case_dir: Path, build_dir: Path) -> list[str]:
    copied: list[str] = []
    for pattern in ["*.data", "*.dat", "*.txt", "*.bin", "*.yaml", "*.yml"]:
        for path in case_dir.glob(pattern):
            target = build_dir / path.name
            if not target.exists() or target.read_bytes() != path.read_bytes():
                build_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append(str(target))
    return copied


def apply_deterministic_repair(
    *,
    case_dir: Path,
    kernel: Path,
    header: Path,
    tb: Path,
    failure_capsule: dict[str, Any] | None = None,
    build_dir: Path | None = None,
) -> DeterministicRepairResult:
    capsule = failure_capsule or {}
    failure_class = str(capsule.get("failure_class") or "")
    top = _top_function(case_dir)
    changed: list[str] = []
    messages: list[str] = []

    if _ensure_include(kernel, header):
        changed.append(str(kernel))
        messages.append(f"added include for {header.name}")

    disallowed = _reject_disallowed_cpp(kernel)
    if disallowed:
        messages.append("detected unsupported C++ constructs: " + ", ".join(disallowed))

    signature = _header_signature(header, top) if top else None
    if top and signature and failure_class == "signature_or_top_mismatch" and _repair_signature_name(kernel, signature, top):
        changed.append(str(kernel))
        messages.append(f"renamed generated top function to {top}")

    if failure_class == "array_dimension_type_error" and _repair_aes_state_column_misindex(kernel, capsule):
        changed.append(str(kernel))
        messages.append("rewrote AES-style state[i][col] indexing to state[col][i]")

    if failure_class == "data_file_runtime_error" and build_dir is not None:
        copied = _stage_tb_data_files(case_dir, build_dir)
        changed.extend(copied)
        if copied:
            messages.append(f"staged {len(copied)} runtime data files")

    applied = bool(changed)
    return DeterministicRepairResult(
        applied=applied,
        repair_type=failure_class or "preflight",
        message="; ".join(messages) if messages else "No deterministic repair applied.",
        changed_files=changed,
    )
