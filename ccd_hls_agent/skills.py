from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .task_modes import TaskMode
from .utils import estimate_tokens, read_text


SKILL_DIR = Path(__file__).resolve().parent / "hls_skills"


class HLSSkill(BaseModel):
    skill_id: str
    title: str
    route: str
    summary: str
    applicability: list[str]
    anti_patterns: list[str]
    expected_effect: dict[str, str]
    risk: str
    source_uri: str
    token_estimate: int = 1


def load_skills(skill_dir: Path | None = None) -> list[HLSSkill]:
    root = skill_dir or SKILL_DIR
    skills = []
    if not root.exists():
        return skills
    for path in sorted(root.glob("*.json")):
        data = json.loads(read_text(path))
        if not data.get("token_estimate"):
            text = " ".join(
                [
                    str(data.get("route", "")),
                    str(data.get("summary", "")),
                    " ".join(data.get("applicability", [])),
                    " ".join(data.get("anti_patterns", [])),
                ]
            )
            data["token_estimate"] = estimate_tokens(text)
        skills.append(HLSSkill.model_validate(data))
    return skills


def _search_text(
    *,
    task_mode: str,
    scan: Any | None,
    failure_capsule: dict[str, Any] | None,
    latest_metrics: dict[str, Any] | None,
    selected_atoms: list[Any] | None,
) -> str:
    parts = [task_mode]
    if scan is not None:
        parts.extend([getattr(scan, "description", ""), " ".join(getattr(scan, "pragmas", [])), " ".join(getattr(scan, "arrays", []))])
    if failure_capsule:
        parts.extend(
            [
                str(failure_capsule.get("failure_type", "")),
                " ".join(map(str, failure_capsule.get("key_errors", []))),
                " ".join(map(str, failure_capsule.get("signal_lines", []))),
                json.dumps(failure_capsule.get("metrics", {}), ensure_ascii=False),
            ]
        )
    if latest_metrics:
        parts.append(json.dumps(latest_metrics, ensure_ascii=False))
    for atom in selected_atoms or []:
        parts.append(str(getattr(atom, "summary", atom)))
    return "\n".join(parts).lower()


def route_skills(
    *,
    task_mode: str | TaskMode,
    scan: Any | None = None,
    failure_capsule: dict[str, Any] | None = None,
    latest_metrics: dict[str, Any] | None = None,
    selected_atoms: list[Any] | None = None,
    skills: list[HLSSkill] | None = None,
    max_skills: int = 3,
    token_budget: int = 600,
) -> tuple[list[HLSSkill], list[HLSSkill]]:
    all_skills = skills or load_skills()
    text = _search_text(
        task_mode=str(task_mode),
        scan=scan,
        failure_capsule=failure_capsule,
        latest_metrics=latest_metrics,
        selected_atoms=selected_atoms,
    )
    mode = task_mode.value if isinstance(task_mode, TaskMode) else str(task_mode)
    failure_class = str((failure_capsule or {}).get("failure_class") or "")
    failure_type = str((failure_capsule or {}).get("failure_type") or "")
    scored: list[tuple[float, HLSSkill]] = []
    for skill in all_skills:
        score = 0.0
        sid = skill.skill_id
        if sid == "hls_signature_repair" and (
            failure_class == "signature_or_top_mismatch"
            or re.search(r"not declared|no matching|undefined reference|undefined symbol|signature|top function", text)
        ):
            score += 10
        if sid == "hls_data_file_runtime" and (
            failure_class == "data_file_runtime_error"
            or failure_type == "runtime_data_file_missing"
            or re.search(r"couldn.t open|input data file|runtime_data_file_missing", text)
        ):
            score += 10
        if sid == "hls_static_bounds" and (
            failure_class in {"array_dimension_type_error", "missing_include_or_type"}
            or re.search(r"variable length|dynamic|malloc|new |unknown loop|static|array_dimension", text)
        ):
            score += 8
        if sid == "hls_loop_pipeline_unroll" and (
            failure_class == "synth_resource_or_loop_error"
            or mode in {TaskMode.REPAIR_SYNTH.value, TaskMode.OPTIMIZE_PPA.value}
            or re.search(r"loop|pipeline|unroll|latency|ii", text)
        ):
            score += 5
        if sid == "hls_array_partition" and (
            failure_class in {"array_dimension_type_error", "synth_resource_or_loop_error"}
            or mode in {TaskMode.REPAIR_SYNTH.value, TaskMode.OPTIMIZE_PPA.value}
            or re.search(r"memory port|array partition|bram|ram|load/load", text)
        ):
            score += 7
        if sid == "hls_dataflow_fifo" and (
            failure_class == "dataflow_deadlock_or_fifo"
            or re.search(r"fifo|stream|dataflow|deadlock|stall", text)
        ):
            score += 9
        if sid == "prometheus_tiling_fusion" and (
            mode == TaskMode.OPTIMIZE_PPA.value or re.search(r"tiling|fusion|permutation|communication", text)
        ):
            score += 5
        if sid == "autodse_bottleneck" and (
            mode == TaskMode.OPTIMIZE_PPA.value or re.search(r"bottleneck|qor|resource|latency|dse", text)
        ):
            score += 5
        if sid == "variable_loop_bounds" and re.search(r"variable loop|runtime bound|while|loop bound", text):
            score += 6
        if sid == "llm4hls_directive_dse" and (mode == TaskMode.OPTIMIZE_PPA.value or re.search(r"directive|pragma|pareto|ppa", text)):
            score += 6
        if failure_capsule:
            terms = set(re.findall(r"[a-z_][a-z0-9_]+", skill.route.lower() + " " + skill.summary.lower()))
            score += min(1, len(terms & set(re.findall(r"[a-z_][a-z0-9_]+", text))) * 0.05)
        if score > 0:
            scored.append((score, skill))
    scored.sort(key=lambda item: (item[0], -item[1].token_estimate), reverse=True)
    selected: list[HLSSkill] = []
    dropped: list[HLSSkill] = []
    used = 0
    for _, skill in scored:
        if len(selected) < max_skills and used + skill.token_estimate <= token_budget:
            selected.append(skill)
            used += skill.token_estimate
        else:
            dropped.append(skill)
    selected_ids = {skill.skill_id for skill in selected}
    dropped.extend([skill for skill in all_skills if skill.skill_id not in selected_ids and skill not in dropped])
    return selected, dropped


def render_skill_capsule(skills: list[HLSSkill]) -> str:
    if not skills:
        return "- No HLS skills selected."
    lines = []
    for skill in skills:
        applicability = "; ".join(skill.applicability[:3])
        anti = "; ".join(skill.anti_patterns[:2])
        effect = ", ".join(f"{key}:{value}" for key, value in skill.expected_effect.items())
        lines.append(
            f"- [{skill.skill_id}] {skill.summary} Applicability: {applicability}. "
            f"Avoid: {anti}. Expected effect: {effect}. Source: {skill.source_uri}"
        )
    return "\n".join(lines)
