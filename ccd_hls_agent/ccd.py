from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import AtomRecord, Stage
from .utils import estimate_tokens, new_id, read_text, sha256_text


SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}


@dataclass
class StaticScanResult:
    source_files: list[Path]
    top_function: str | None
    description: str
    loops: list[str]
    arrays: list[str]
    pragmas: list[str]
    blocker: str | None
    metrics: dict[str, Any]


def scan_benchmark(benchmark_path: Path, top_override: str | None = None) -> StaticScanResult:
    source_files = sorted([p for p in benchmark_path.glob("*") if p.suffix in SOURCE_EXTENSIONS and p.is_file()])
    top_file = benchmark_path / "top.txt"
    desc_file = benchmark_path / "kernel_description.md"
    top_function = top_override or read_text(top_file).strip() or None
    description = read_text(desc_file)
    code = "\n".join(read_text(p) for p in source_files)

    loops = []
    for idx, match in enumerate(re.finditer(r"\b(for|while)\s*\(", code), start=1):
        loops.append(f"loop_{idx}@char_{match.start()}")

    arrays = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[[^\]]+\]", code)))
    pragmas = [line.strip() for line in code.splitlines() if "#pragma HLS" in line]

    blocker = None
    if not benchmark_path.exists():
        blocker = "benchmark_path 不存在"
    elif not source_files:
        blocker = "未找到 C/C++ 源文件"
    elif top_function is None:
        blocker = "缺少 top.txt 或 top_function"

    return StaticScanResult(
        source_files=source_files,
        top_function=top_function,
        description=description,
        loops=loops,
        arrays=arrays,
        pragmas=pragmas,
        blocker=blocker,
        metrics={
            "source_files": len(source_files),
            "loops": len(loops),
            "arrays": len(arrays),
            "pragmas": len(pragmas),
            "description_tokens_est": estimate_tokens(description) if description else 0,
        },
    )


def atomize_static_scan(
    task_id: str,
    run_id: str,
    benchmark_path: Path,
    scan: StaticScanResult,
) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    code_hash = sha256_text("\n".join(read_text(p) for p in scan.source_files))

    def add(kind: str, scope: str, summary: str, evidence: str | None = None, certainty: float = 0.75) -> None:
        atoms.append(
            AtomRecord(
                atom_id=new_id("atom"),
                task_id=task_id,
                run_id=run_id,
                kind=kind,
                scope=scope,
                stage=Stage.CONTEXT_ATOMIZE.value,
                summary=summary,
                evidence_uri=evidence,
                code_hash=code_hash,
                token_estimate=estimate_tokens(summary),
                certainty_score=certainty,
                value_score=0.0,
            )
        )

    add("task_requirement", "task", f"Benchmark path: {benchmark_path}", str(benchmark_path), 0.8)
    if scan.description:
        add("task_requirement", "task/description", scan.description[:1200], str(benchmark_path / "kernel_description.md"), 0.8)
    if scan.top_function:
        add("code_scope", f"top/{scan.top_function}", f"Top function is {scan.top_function}.", str(benchmark_path / "top.txt"), 0.85)
    if scan.blocker:
        add("static_error", "task/static", scan.blocker, None, 0.9)
    for loop in scan.loops:
        add("code_scope", f"loop/{loop}", f"Detected loop scope {loop}.", None, 0.7)
    for array in scan.arrays:
        add("code_scope", f"array/{array}", f"Detected array-like object {array}.", None, 0.7)
    for idx, pragma in enumerate(scan.pragmas, start=1):
        add("pragma_fact", f"pragma/{idx}", pragma, None, 0.75)
    return atoms


def atomize_tool_output(
    task_id: str,
    run_id: str,
    stage: str,
    stdout: str,
    stderr: str,
    metrics: dict[str, Any],
) -> list[AtomRecord]:
    text = "\n".join(part for part in [stdout, stderr] if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    atoms: list[AtomRecord] = []
    patterns = [
        ("tool_error", re.compile(r"(error|failed|failure|cannot|undefined|invalid)", re.I), 0.9),
        ("tool_warning", re.compile(r"(warning|ignored|bottleneck|violation)", re.I), 0.82),
    ]
    for line in lines[:80]:
        for kind, pattern, certainty in patterns:
            if pattern.search(line):
                atoms.append(
                    AtomRecord(
                        atom_id=new_id("atom"),
                        task_id=task_id,
                        run_id=run_id,
                        kind=kind,
                        scope="tool/log",
                        stage=stage,
                        summary=line[:500],
                        evidence_uri=None,
                        code_hash=None,
                        token_estimate=estimate_tokens(line[:500]),
                        certainty_score=certainty,
                        value_score=0.0,
                    )
                )
                break

    for key, value in metrics.items():
        if isinstance(value, (int, float, str)) and value not in ("", None):
            summary = f"{key}={value}"
            atoms.append(
                AtomRecord(
                    atom_id=new_id("atom"),
                    task_id=task_id,
                    run_id=run_id,
                    kind="synth_metric",
                    scope=f"metric/{key}",
                    stage=stage,
                    summary=summary,
                    token_estimate=estimate_tokens(summary),
                    certainty_score=0.95,
                    value_score=0.0,
                )
            )
    return atoms


def choose_frontier(scan: StaticScanResult, latest_blocker: str | None) -> str:
    if latest_blocker:
        return "task/static"
    if scan.pragmas:
        return "pragma"
    if scan.loops and scan.arrays:
        return f"{scan.loops[0]}/{scan.arrays[0]}"
    if scan.loops:
        return scan.loops[0]
    if scan.top_function:
        return f"top/{scan.top_function}"
    return "task"


def _scope_distance(scope: str, frontier: str) -> int:
    if scope == frontier:
        return 0
    if scope in frontier or frontier in scope:
        return 1
    scope_parts = set(scope.replace("@", "/").split("/"))
    frontier_parts = set(frontier.replace("@", "/").split("/"))
    if scope_parts & frontier_parts:
        return 2
    if "task" in scope_parts or "task" in frontier_parts:
        return 3
    return 4


def score_atoms(
    atoms: list[AtomRecord],
    frontier: str,
    *,
    latest_blocker: str | None = None,
    half_life_steps: float = 6.0,
    tau: float = 2.0,
) -> list[AtomRecord]:
    scored: list[AtomRecord] = []
    selected_summaries: list[str] = []
    for age, atom in enumerate(atoms):
        distance = _scope_distance(atom.scope, frontier)
        causal = math.exp(-distance / tau)
        blocker_rel = 0.2
        if latest_blocker and latest_blocker.lower() in atom.summary.lower():
            blocker_rel = 1.0
        elif atom.kind in {"static_error", "tool_error", "csim_failure", "synth_error"}:
            blocker_rel = 0.8
        elif atom.kind in {"pragma_fact", "synth_metric"}:
            blocker_rel = 0.55

        evidence_strength = atom.certainty_score
        qor_impact = 0.65 if atom.kind in {"synth_metric", "pragma_fact"} else 0.25
        uncertainty_need = 1.0 - min(1.0, max(0.0, atom.certainty_score))
        recency = math.exp(-age / half_life_steps)
        token_cost_norm = min(1.0, atom.token_estimate / 400.0)

        redundancy = 0.0
        if any(atom.summary == prev or atom.summary[:80] == prev[:80] for prev in selected_summaries):
            redundancy = 0.8
        superseded = 0.0 if atom.status == "active" else 0.8

        value = (
            0.24 * causal
            + 0.20 * blocker_rel
            + 0.16 * evidence_strength
            + 0.12 * qor_impact
            + 0.10 * 0.0
            + 0.08 * uncertainty_need
            + 0.06 * recency
            + 0.04 * 0.0
            - 0.16 * redundancy
            - 0.14 * superseded
            - 0.08 * token_cost_norm
        )
        certainty = min(
            1.0,
            max(
                0.0,
                0.35 * evidence_strength
                + 0.25 * 1.0
                + 0.20 * 0.2
                + 0.10 * 0.9
                + 0.10 * 0.8,
            ),
        )
        atom.value_score = round(value, 4)
        atom.certainty_score = round(certainty, 4)
        selected_summaries.append(atom.summary)
        scored.append(atom)
    return sorted(scored, key=lambda item: item.value_score / max(8, item.token_estimate), reverse=True)


def select_context(
    atoms: list[AtomRecord],
    token_budget: int,
    *,
    max_atoms: int = 12,
) -> tuple[list[AtomRecord], list[AtomRecord]]:
    selected: list[AtomRecord] = []
    dropped: list[AtomRecord] = []
    used = 0
    for atom in atoms:
        if len(selected) < max_atoms and used + atom.token_estimate <= token_budget:
            selected.append(atom)
            used += atom.token_estimate
        else:
            dropped.append(atom)
    return selected, dropped


def retrieve_patterns(atoms: list[AtomRecord], query: str, limit: int = 3) -> list[dict[str, Any]]:
    query_terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query.lower()))
    scored = []
    for atom in atoms:
        terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", atom.summary.lower()))
        overlap = len(query_terms & terms)
        if overlap or atom.kind in {"pragma_fact", "tool_error", "synth_metric"}:
            scored.append((overlap + atom.value_score, atom))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "atom_id": atom.atom_id,
            "kind": atom.kind,
            "scope": atom.scope,
            "summary": atom.summary,
            "score": score,
        }
        for score, atom in scored[:limit]
    ]


def build_prompt(
    scan: StaticScanResult,
    selected_atoms: list[AtomRecord],
    patterns: list[dict[str, Any]],
    *,
    stage: str,
    frontier: str,
    token_budget: int,
    input_files: dict[str, str] | None = None,
) -> str:
    selected_lines = "\n".join(
        f"- [{atom.kind} scope={atom.scope} value={atom.value_score:.3f} certainty={atom.certainty_score:.3f}] {atom.summary}"
        for atom in selected_atoms
    )
    pattern_lines = "\n".join(
        f"- [{p['kind']} score={p['score']:.3f}] {p['summary']}"
        for p in patterns
    )
    input_file_lines = ""
    if input_files:
        parts = []
        for name, content in input_files.items():
            parts.append(f"```{name}\n{content[:6000]}\n```")
        input_file_lines = "\n\nInput files:\n" + "\n\n".join(parts)
    prompt = f"""You are editing HLS C/C++ for Vitis HLS.

Stage:
{stage}

Goal:
Generate a minimal unified diff that improves the current HLS-Eval task. Preserve the required top function and testbench-visible interface.

Task capsule:
- top_function: {scan.top_function}
- source_files: {len(scan.source_files)}
- loops: {len(scan.loops)}
- arrays: {len(scan.arrays)}
- pragmas: {len(scan.pragmas)}

Current frontier:
{frontier}

Kernel description:
{scan.description[:1800]}
{input_file_lines}

Selected facts:
{selected_lines or "- No selected context atoms."}

Retrieved patterns:
{pattern_lines or "- No retrieved patterns."}

Allowed actions:
- Return a unified diff only for design source/header files.
- Include a compact JSON metadata block after the diff.

Forbidden actions:
- Do not edit the testbench.
- Do not remove the top function.
- Do not add dynamic memory allocation or unsupported system calls.

Return only:
1. A unified diff.
2. A compact JSON metadata block.
"""
    if estimate_tokens(prompt) > token_budget:
        prompt = prompt[: int(token_budget * 3.6)]
    return prompt


def reduce_trajectory_agentdiet(events: list[dict[str, Any]], window: int = 8) -> list[dict[str, Any]]:
    if len(events) <= window:
        return events
    stable_summary = {
        "stage": "AGENTDIET_SUMMARY",
        "status": "compressed",
        "message": f"Compressed {len(events) - window} older trajectory events.",
        "metrics": {
            "compressed_events": len(events) - window,
            "kept_recent_events": window,
        },
    }
    return [stable_summary] + events[-window:]


def rank_ppa_candidates(scan: StaticScanResult, limit: int = 3) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for loop in scan.loops:
        candidates.append(
            {
                "action": "pipeline",
                "scope": loop,
                "score": 0.72,
                "expected_effect": {"latency": "decrease", "II": "decrease"},
            }
        )
    for array in scan.arrays:
        candidates.append(
            {
                "action": "array_partition",
                "scope": array,
                "score": 0.66,
                "expected_effect": {"II": "decrease", "BRAM": "increase"},
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def extract_unified_diff(text: str) -> str | None:
    if "diff --git" in text or text.startswith("--- "):
        return text
    fenced = re.search(r"```(?:diff)?\s*(.*?)```", text, re.S)
    if fenced:
        candidate = fenced.group(1).strip()
        if "diff --git" in candidate or candidate.startswith("--- "):
            return candidate
    return None
