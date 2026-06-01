from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from .json_utils import make_json_safe
from .utils import estimate_tokens


LOG_SIGNAL_RE = re.compile(
    r"(error:|fatal error|undefined reference|undefined symbol|not declared|no matching|invalid|cannot|"
    r"mismatch|failed|FAIL|ERROR|CRITICAL WARNING|assertion|couldn.t open|CSim failed|@E)",
    re.I,
)
HIGH_PRIORITY_LOG_RE = re.compile(
    r"(error:|fatal error|undefined reference|undefined symbol|not declared|no matching|assertion|"
    r"couldn.t open|FAIL:|ERROR:|@E|CSim failed|mismatch)",
    re.I,
)


def ensure_text(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text


def truncate_estimated_tokens(text: str | bytes, token_budget: int, *, keep_tail: bool = False) -> str:
    text = ensure_text(text)
    if estimate_tokens(text) <= token_budget:
        return text
    max_chars = max(200, token_budget * 4)
    if keep_tail:
        return "[truncated]\n" + text[-max_chars:]
    return text[:max_chars] + "\n[truncated]"


def extract_signal_lines(text: str | bytes, *, limit: int = 40) -> list[str]:
    lines = []
    for raw in ensure_text(text).splitlines():
        line = raw.strip()
        if line and LOG_SIGNAL_RE.search(line):
            lines.append(line)
    deduped = list(dict.fromkeys(lines))
    return deduped[-limit:]


def extract_error_windows(text: str | bytes, *, context: int = 2, limit: int = 8) -> list[str]:
    lines = ensure_text(text).splitlines()
    windows = []
    for idx, line in enumerate(lines):
        if HIGH_PRIORITY_LOG_RE.search(line):
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            window = "\n".join(lines[start:end]).strip()
            if window:
                windows.append(window)
    deduped = list(dict.fromkeys(windows))
    return deduped[-limit:]


def summarize_failure_history(capsules: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    history = []
    for capsule in capsules[-limit:]:
        history.append(
            {
                "stage": capsule.get("stage"),
                "failure_type": capsule.get("failure_type"),
                "return_code": capsule.get("return_code"),
                "key_errors": capsule.get("key_errors", [])[:4],
                "signal_lines": capsule.get("signal_lines", [])[:6],
            }
        )
    return history


def build_failure_capsule(stage: str, tool_result: Any, *, token_budget: int) -> dict[str, Any]:
    stdout = ensure_text(getattr(tool_result, "stdout", ""))
    stderr = ensure_text(getattr(tool_result, "stderr", ""))
    metrics = make_json_safe(getattr(tool_result, "metrics", {}) or {})
    log_text = stderr + "\n" + stdout
    signal_lines = extract_signal_lines(log_text)
    error_windows = extract_error_windows(log_text)
    timeout = any(bool(v) for k, v in metrics.items() if "timeout" in k)
    if timeout:
        failure_type = "timeout"
    elif any("couldn't open input data file" in line.lower() or "couldn't open" in line.lower() for line in signal_lines):
        failure_type = "runtime_data_file_missing"
    elif any("undefined reference" in line.lower() or "undefined symbol" in line.lower() for line in signal_lines):
        failure_type = "link_or_symbol_error"
    elif any("not declared" in line.lower() or "no matching" in line.lower() for line in signal_lines):
        failure_type = "signature_or_compile_error"
    elif stage == "SYNTH":
        failure_type = "synthesis_error"
    elif stage == "CSIM" and metrics.get("can_compile") is True:
        failure_type = "testbench_failure"
    else:
        failure_type = "compile_error"

    tail_budget = max(120, token_budget - sum(estimate_tokens(line) for line in signal_lines))
    capsule = {
        "stage": stage,
        "failure_type": failure_type,
        "return_code": getattr(tool_result, "return_code", None),
        "command": getattr(tool_result, "command", ""),
        "duration_ms": getattr(tool_result, "duration_ms", 0),
        "metrics": metrics,
        "key_errors": error_windows,
        "signal_lines": signal_lines,
        "stderr_tail": truncate_estimated_tokens(stderr, tail_budget // 2, keep_tail=True),
        "stdout_tail": truncate_estimated_tokens(stdout, tail_budget // 2, keep_tail=True),
    }
    capsule = make_json_safe(capsule)
    capsule_text = json.dumps(capsule, ensure_ascii=False)
    if estimate_tokens(capsule_text) <= token_budget:
        return capsule
    capsule["stderr_tail"] = truncate_estimated_tokens(stderr, max(80, token_budget // 4), keep_tail=True)
    capsule["stdout_tail"] = truncate_estimated_tokens(stdout, max(80, token_budget // 4), keep_tail=True)
    return capsule


def _normalize_failure_text(lines: list[Any]) -> str:
    text = "\n".join(str(line) for line in lines if str(line).strip()).lower()
    text = re.sub(r"/[^\\s:]+", "<path>", text)
    text = re.sub(r"\b\d+\b", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def failure_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left.get("failure_type") != right.get("failure_type"):
        return 0.0
    left_text = _normalize_failure_text(left.get("key_errors") or left.get("signal_lines") or [])
    right_text = _normalize_failure_text(right.get("key_errors") or right.get("signal_lines") or [])
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def repeated_failure_early_stop(capsules: list[dict[str, Any]], *, threshold: float = 0.92) -> tuple[bool, float]:
    if len(capsules) < 2:
        return False, 0.0
    score = failure_similarity(capsules[-2], capsules[-1])
    return score >= threshold, score
