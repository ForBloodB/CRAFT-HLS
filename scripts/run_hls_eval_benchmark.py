#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any

from ccd_hls_agent.budget import BudgetLedger
from ccd_hls_agent.ccd import (
    atomize_static_scan,
    choose_frontier,
    scan_benchmark,
    score_atoms,
    select_context,
)
from ccd_hls_agent.contracts import (
    build_ccd_hls_gen_v2_prompt as render_ccd_hls_gen_v2_prompt,
    build_hls_repair_prompt as render_hls_repair_prompt,
    build_output_code_repair_prompt as render_output_code_repair_prompt,
)
from ccd_hls_agent.deterministic_repair import apply_deterministic_repair
from ccd_hls_agent.failure_analysis import (
    build_failure_capsule as make_failure_capsule,
    extract_error_windows as failure_error_windows,
    extract_signal_lines as failure_signal_lines,
    repeated_failure_early_stop,
    summarize_failure_history as summarize_failure_capsules,
    truncate_estimated_tokens as truncate_text_by_tokens,
)
from ccd_hls_agent.hls_backends import HLSBackendConfig, ToolResult, build_hls_backend, discover_tb_data_files
from ccd_hls_agent.json_utils import make_json_safe
from ccd_hls_agent.local_memory import HLSLocalMemory, error_signature_from_capsule, render_action_memory_capsule, render_memory_capsule
from ccd_hls_agent.model_clients import build_model_client
from ccd_hls_agent.repair_actions import (
    action_ids_for_failure_class,
    action_ids_from_selected,
    build_diagnosis_assertion,
    candidate_score,
    render_action_capsule,
    select_best_candidate,
    select_repair_actions,
)
from ccd_hls_agent.schemas import ModelConfig
from ccd_hls_agent.skills import render_skill_capsule, route_skills
from ccd_hls_agent.task_modes import TaskMode, classify_task_mode, is_generation_stub
from ccd_hls_agent.token_report import TOKEN_STAGES, build_token_report, write_case_token_report, write_token_summary
from ccd_hls_agent.utils import estimate_tokens, fit_text_to_token_budget, json_dumps, new_id, read_text, utc_now, write_text
from ccd_hls_agent.workflow import write_workflow_artifacts_from_stage_records


SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}
DIFF_MARKER_RE = re.compile(r"(?m)^(diff --git|---\s+[ab]/|\+\+\+\s+[ab]/|@@\s)")


@dataclass
class CaseResult:
    experiment_id: str
    method: str
    sample_idx: int
    case_name: str
    case_path: str
    tags: list[str]
    can_parse: bool
    can_compile: bool
    can_pass_testbench: bool
    can_synthesize: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_duration_ms: int
    tool_calls: int
    wall_time_ms: int
    error: str | None
    metrics: dict[str, Any]
    artifacts_dir: str


def result_to_json_dict(result: CaseResult) -> dict[str, Any]:
    return make_json_safe(asdict(result))


def normalize_model_config(model_data: dict[str, Any]) -> dict[str, Any]:
    api_key_env = str(model_data.get("api_key_env") or "")
    if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        model_data = {**model_data, "api_key": api_key_env, "api_key_env": None}
    return model_data


def public_model_dump(model: ModelConfig) -> dict[str, Any]:
    data = model.model_dump()
    data["api_key"] = "***" if model.api_key else None
    if data.get("api_key_env"):
        data["api_key_env"] = str(data["api_key_env"])
    return data


def llm_prompt_budget(model: ModelConfig, system_prompt: str, *, reserve_tokens: int = 256) -> int:
    return max(256, int(model.context_window) - int(model.max_tokens) - estimate_tokens(system_prompt) - reserve_tokens)


def fit_prompt_for_model(prompt: str, model: ModelConfig, system_prompt: str) -> tuple[str, dict[str, Any]]:
    budget = llm_prompt_budget(model, system_prompt)
    original_tokens = estimate_tokens(prompt)
    if original_tokens <= budget:
        return prompt, {
            "prompt_fit_applied": False,
            "prompt_tokens_est_before_fit": original_tokens,
            "prompt_tokens_est_after_fit": original_tokens,
            "prompt_token_budget": budget,
            "context_window": model.context_window,
        }
    fitted = fit_text_to_token_budget(prompt, budget)
    return fitted, {
        "prompt_fit_applied": True,
        "prompt_tokens_est_before_fit": original_tokens,
        "prompt_tokens_est_after_fit": estimate_tokens(fitted),
        "prompt_token_budget": budget,
        "context_window": model.context_window,
    }


def discover_cases(data_dir: Path) -> list[Path]:
    return sorted(config.parent for config in data_dir.rglob("hls_eval_config.toml"))


def parse_tags(config_path: Path) -> list[str]:
    text = read_text(config_path)
    match = re.search(r"tags\s*=\s*\[(.*?)\]", text, re.S)
    if not match:
        return []
    return [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]


def find_header(case_dir: Path) -> Path:
    headers = sorted([p for p in case_dir.glob("*") if p.suffix in {".h", ".hh", ".hpp"}])
    if not headers:
        raise FileNotFoundError(f"No header file in {case_dir}")
    return headers[0]


def find_tb(case_dir: Path) -> Path:
    matches = sorted(case_dir.glob("*_tb.cpp"))
    if not matches:
        raise FileNotFoundError(f"No *_tb.cpp in {case_dir}")
    return matches[0]


def find_kernel_cpp(case_dir: Path) -> Path:
    matches = sorted([p for p in case_dir.glob("*.cpp") if not p.name.endswith("_tb.cpp")])
    if not matches:
        raise FileNotFoundError(f"No kernel .cpp in {case_dir}")
    return matches[0]


def prepare_generation_case(src_case: Path, dst_case: Path) -> dict[str, Path]:
    if dst_case.exists():
        shutil.rmtree(dst_case)
    shutil.copytree(src_case, dst_case)
    header = find_header(dst_case)
    tb = find_tb(dst_case)
    kernel = find_kernel_cpp(dst_case)
    top = read_text(dst_case / "top.txt").strip()
    write_text(
        kernel,
        f'#include "{header.name}"\n\n'
        f"// HLS-Eval generation mode: reference implementation hidden.\n"
        f"// The model must replace this stub with the implementation for top function: {top}.\n",
    )
    return {"header": header, "tb": tb, "kernel": kernel, "description": dst_case / "kernel_description.md"}


def ensure_hls_eval_importable(hls_eval_root: Path | None) -> None:
    if not hls_eval_root:
        return
    root = hls_eval_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def build_hls_eval_zero_shot_prompt(description: Path, tb: Path, header: Path, hls_eval_root: Path | None = None) -> str:
    ensure_hls_eval_importable(hls_eval_root or Path("external/hls-eval"))
    from hls_eval.prompts import build_prompt_gen_zero_shot

    return build_prompt_gen_zero_shot(description, tb, header)


def build_ccd_hls_gen_v2_prompt(
    description: Path,
    tb: Path,
    header: Path,
    kernel: Path,
    selected_atoms: list[Any],
    *,
    token_budget: int,
    baseline_prompt_tokens: int,
    hls_skill_capsule: str = "- No HLS skills selected.",
) -> tuple[str, list[Any], int]:
    return render_ccd_hls_gen_v2_prompt(
        description,
        tb,
        header,
        kernel,
        selected_atoms,
        token_budget=token_budget,
        baseline_prompt_tokens=baseline_prompt_tokens,
        hls_skill_capsule=hls_skill_capsule,
    )


def extract_output_code(text: str) -> dict[str, str]:
    matches = re.finditer(r"<OUTPUT_CODE\s+name=[\"'](?P<name>[^\"']+)[\"'][^>]*>\s*(?P<code>.*?)\s*</OUTPUT_CODE>", text, re.S)
    out = {m.group("name"): m.group("code") for m in matches}
    if not out:
        raise ValueError("No <OUTPUT_CODE> block found.")
    return out


def has_diff_markers(text: str) -> bool:
    return bool(DIFF_MARKER_RE.search(text))


def source_has_diff_marker(case_dir: Path) -> bool:
    for src in case_dir.glob("*.cpp"):
        if src.name.endswith("_tb.cpp"):
            continue
        if has_diff_markers(read_text(src)[:1000]):
            return True
    return False


def extract_code_blocks(text: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```(?:cpp|c\+\+|c|[A-Za-z0-9_.-]+)?\s*(.*?)```", text, re.S):
        code = match.group(1).strip()
        if code:
            blocks.append(code)
    return blocks


def write_kernel_output_code(text: str, kernel_path: Path) -> tuple[bool, str]:
    if has_diff_markers(text):
        return False, "PATCH_OR_OUTPUT_PARSE_FAILED: model output contains diff markers."
    try:
        generated = extract_output_code(text)
    except ValueError as exc:
        return False, f"PATCH_OR_OUTPUT_PARSE_FAILED: {exc}"
    if set(generated) != {kernel_path.name}:
        return False, f"PATCH_OR_OUTPUT_PARSE_FAILED: expected only OUTPUT_CODE name={kernel_path.name}, got {sorted(generated)}."
    code = generated[kernel_path.name]
    if has_diff_markers(code):
        return False, "PATCH_OR_OUTPUT_PARSE_FAILED: OUTPUT_CODE content contains diff markers."
    write_text(kernel_path, code)
    return True, f"Used OUTPUT_CODE exact match for {kernel_path.name}."


def looks_like_cpp_source(text: str) -> bool:
    if has_diff_markers(text):
        return False
    has_function = bool(
        re.search(
            r"(?m)^\s*(?:template\s*<[^>]+>\s*)?(?:void|int|float|double|bool|char|ap_\w+|[A-Za-z_][A-Za-z0-9_:<>]*_t|[A-Za-z_][A-Za-z0-9_:<>]*)\s+[*&\s]*[A-Za-z_][A-Za-z0-9_:]*\s*\(",
            text,
        )
    )
    return ("#include" in text or has_function) and "{" in text and "}" in text


def extract_cpp_repair_candidate(text: str) -> tuple[str | None, str]:
    if has_diff_markers(text):
        return None, "repair rejected: original response contains diff markers."

    blocks = extract_code_blocks(text)
    cpp_blocks = [block.strip() for block in blocks if looks_like_cpp_source(block)]
    if cpp_blocks:
        return cpp_blocks[-1], "local_secondary_parse_markdown_cpp_block"

    raw = text.strip()
    include_at = raw.find("#include")
    if include_at >= 0:
        raw = raw[include_at:].strip()
    if looks_like_cpp_source(raw):
        return raw, "local_secondary_parse_raw_cpp"
    return None, "local secondary parse found no standalone C/C++ source."


def write_kernel_output_code_with_local_repair(text: str, kernel_path: Path) -> tuple[bool, str, str | None]:
    ok, message = write_kernel_output_code(text, kernel_path)
    if ok:
        return True, message, "output_code_exact"
    if "No <OUTPUT_CODE> block found" not in message:
        return False, message, None

    candidate, repair_mode = extract_cpp_repair_candidate(text)
    if not candidate:
        return False, f"{message}; {repair_mode}", None
    write_text(kernel_path, candidate)
    return True, f"Recovered {kernel_path.name} via {repair_mode}.", repair_mode


def build_output_code_repair_prompt(kernel_name: str, original_response: str) -> str:
    return render_output_code_repair_prompt(kernel_name, original_response)


def truncate_estimated_tokens(text: str, token_budget: int, *, keep_tail: bool = False) -> str:
    return truncate_text_by_tokens(text, token_budget, keep_tail=keep_tail)


def extract_signal_lines(text: str, *, limit: int = 40) -> list[str]:
    return failure_signal_lines(text, limit=limit)


def extract_error_windows(text: str, *, context: int = 2, limit: int = 8) -> list[str]:
    return failure_error_windows(text, context=context, limit=limit)


def summarize_failure_history(capsules: list[dict[str, Any]], *, limit: int = 4) -> list[dict[str, Any]]:
    return summarize_failure_capsules(capsules, limit=limit)


def build_failure_capsule(
    stage: str,
    tool_result: ToolResult,
    *,
    token_budget: int,
) -> dict[str, Any]:
    return make_failure_capsule(stage, tool_result, token_budget=token_budget)


def build_hls_repair_prompt(
    *,
    stage: str,
    kernel: Path,
    header: Path,
    tb: Path,
    description: Path,
    failure_capsule: dict[str, Any],
    failure_history: list[dict[str, Any]],
    attempt: int,
    max_llm_calls: int,
    hls_skill_capsule: str = "- No HLS skills selected.",
    local_memory_capsule: str = "- No verified local memory matched this failure.",
    token_budget: int | None = None,
) -> str:
    return render_hls_repair_prompt(
        stage=stage,
        kernel=kernel,
        header=header,
        tb=tb,
        description=description,
        failure_capsule=failure_capsule,
        failure_history=failure_history,
        attempt=attempt,
        max_llm_calls=max_llm_calls,
        hls_skill_capsule=hls_skill_capsule,
        local_memory_capsule=local_memory_capsule,
        token_budget=token_budget,
    )


def add_stage_record(
    records: list[dict[str, Any]],
    *,
    stage: str,
    status: str,
    message: str,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> None:
    records.append(
        {
            "stage": stage,
            "status": status,
            "message": message,
            "metrics": metrics or {},
            "artifacts": artifacts or {},
            "created_at": utc_now(),
        }
    )


def write_json_artifact(path: Path, data: Any) -> str:
    write_text(path, json.dumps(make_json_safe(data), ensure_ascii=False, indent=2))
    return str(path)


def snapshot_kernel(kernel: Path, workdir: Path, label: str) -> str:
    target = workdir / f"code_snapshot_{label}.cpp"
    write_text(target, read_text(kernel))
    return str(target)


def source_files(case_dir: Path, *, include_tb: bool) -> list[Path]:
    files = sorted([p for p in case_dir.glob("*") if p.suffix in SOURCE_EXTENSIONS])
    if include_tb:
        return files
    return [p for p in files if not p.name.endswith("_tb.cpp")]


def stage_runtime_files(case_dir: Path, build_dir: Path) -> list[str]:
    build_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    runtime_files = discover_tb_data_files(source_files(case_dir, include_tb=True))
    for item in runtime_files:
        target = build_dir / item.name
        shutil.copy2(item, target)
        copied.append(str(target))
    return copied


def backend_config(hls_eval_root: Path | None, hls_part: str | None, hls_platform: str | None) -> HLSBackendConfig:
    defaults = HLSBackendConfig()
    return HLSBackendConfig(
        hls_eval_root=str(hls_eval_root) if hls_eval_root else None,
        part=hls_part or defaults.part,
        platform=hls_platform or defaults.platform,
    )


def backend_case_config(top_function: str | None, hls_part: str | None, hls_platform: str | None) -> dict[str, Any]:
    defaults = HLSBackendConfig()
    return {
        "top_function": top_function,
        "part": hls_part or defaults.part,
        "platform": hls_platform or defaults.platform,
    }


async def evaluate_design(
    backend_kind: str,
    hls_eval_root: Path | None,
    hls_part: str | None,
    hls_platform: str | None,
    case_dir: Path,
    build_dir: Path,
    top_function: str | None,
) -> tuple[dict[str, bool], dict[str, Any], int]:
    backend = build_hls_backend(backend_kind, backend_config(hls_eval_root, hls_part, hls_platform))
    config = backend_case_config(top_function, hls_part, hls_platform)
    tool_calls = 0
    stage_runtime_files(case_dir, build_dir)
    csim = await asyncio.to_thread(backend.run_csim, build_dir, source_files(case_dir, include_tb=True), config)
    tool_calls += 1
    synth: ToolResult | None = None
    if csim.return_code == 0:
        synth = await asyncio.to_thread(backend.run_synth, build_dir, source_files(case_dir, include_tb=False), config)
        tool_calls += 1
    can_compile = bool(csim.metrics.get("can_compile", csim.return_code == 0))
    can_pass = bool(csim.metrics.get("can_pass_testbench", csim.return_code == 0))
    can_synth = bool(synth and synth.return_code == 0)
    metrics = {
        "csim": csim.metrics,
        "synth": synth.metrics if synth else {},
        "csim_return_code": csim.return_code,
        "synth_return_code": synth.return_code if synth else None,
        "csim_stdout_tail": csim.stdout[-2000:],
        "csim_stderr_tail": csim.stderr[-2000:],
        "synth_stdout_tail": synth.stdout[-2000:] if synth else "",
        "synth_stderr_tail": synth.stderr[-2000:] if synth else "",
    }
    return (
        {
            "can_compile": can_compile,
            "can_pass_testbench": can_pass,
            "can_synthesize": can_synth,
        },
        metrics,
        tool_calls,
    )


async def run_csim_stage(
    backend_kind: str,
    hls_eval_root: Path | None,
    hls_part: str | None,
    hls_platform: str | None,
    case_dir: Path,
    build_dir: Path,
    top_function: str | None,
) -> tuple[dict[str, bool], dict[str, Any], ToolResult]:
    backend = build_hls_backend(backend_kind, backend_config(hls_eval_root, hls_part, hls_platform))
    config = backend_case_config(top_function, hls_part, hls_platform)
    runtime_files = stage_runtime_files(case_dir, build_dir)
    csim = await asyncio.to_thread(backend.run_csim, build_dir, source_files(case_dir, include_tb=True), config)
    flags = {
        "can_compile": bool(csim.metrics.get("can_compile", csim.return_code == 0)),
        "can_pass_testbench": bool(csim.metrics.get("can_pass_testbench", csim.return_code == 0)),
        "can_synthesize": False,
    }
    metrics = {
        "csim": csim.metrics,
        "csim_return_code": csim.return_code,
        "csim_runtime_files": runtime_files,
        "csim_stdout_tail": csim.stdout[-2000:],
        "csim_stderr_tail": csim.stderr[-2000:],
    }
    return flags, metrics, csim


async def run_synth_stage(
    backend_kind: str,
    hls_eval_root: Path | None,
    hls_part: str | None,
    hls_platform: str | None,
    case_dir: Path,
    build_dir: Path,
    top_function: str | None,
) -> tuple[bool, dict[str, Any], ToolResult]:
    backend = build_hls_backend(backend_kind, backend_config(hls_eval_root, hls_part, hls_platform))
    config = backend_case_config(top_function, hls_part, hls_platform)
    synth = await asyncio.to_thread(backend.run_synth, build_dir, source_files(case_dir, include_tb=False), config)
    can_synthesize = synth.return_code == 0
    metrics = {
        "synth": synth.metrics,
        "synth_return_code": synth.return_code,
        "synth_stdout_tail": synth.stdout[-2000:],
        "synth_stderr_tail": synth.stderr[-2000:],
    }
    return can_synthesize, metrics, synth


async def run_ccd_gen_v2_case(
    experiment_id: str,
    case_path: Path,
    workdir: Path,
    model: ModelConfig,
    backend_kind: str,
    hls_eval_root: Path | None,
    sample_idx: int,
    prompt_budget: int,
    max_llm_calls: int = 5,
    repair_log_token_budget: int = 1200,
    early_stop_similarity_threshold: float = 0.92,
    method_name: str = "ccd_hls_gen_v2",
    llm_call_budget: int | None = None,
    csim_budget: int | None = None,
    synth_budget: int | None = None,
    cosim_budget: int | None = 0,
    unified_credit_budget: int | None = None,
    skill_token_budget: int = 600,
    enable_deterministic_repair: bool = True,
    enable_local_memory: bool = True,
    memory_path: Path | None = None,
    candidate_count: int = 1,
    candidate_policy: str = "repair_only",
    hls_part: str | None = None,
    hls_platform: str | None = None,
) -> CaseResult:
    t0 = time.monotonic()
    files = prepare_generation_case(case_path, workdir / "design")
    task_mode = TaskMode.GENERATE
    budget = BudgetLedger.from_limits(
        llm_calls=max_llm_calls if llm_call_budget is None else llm_call_budget,
        csim_calls=csim_budget,
        synth_calls=synth_budget,
        cosim_calls=cosim_budget,
        unified_credits=unified_credit_budget,
    )
    local_memory = HLSLocalMemory(memory_path) if enable_local_memory else None
    zero_prompt = build_hls_eval_zero_shot_prompt(files["description"], files["tb"], files["header"], hls_eval_root)
    scan = scan_benchmark(workdir / "design")
    frontier = choose_frontier(scan, scan.blocker)
    atoms = atomize_static_scan("experiment", "run", workdir / "design", scan)
    scored = score_atoms(atoms, frontier, latest_blocker=scan.blocker)
    candidate_atoms, dropped = select_context(scored, 800, max_atoms=6)
    selected_skills, dropped_skills = route_skills(
        task_mode=task_mode,
        scan=scan,
        selected_atoms=candidate_atoms,
        token_budget=skill_token_budget,
    )
    skill_capsule = render_skill_capsule(selected_skills)
    prompt, selected, max_prompt_tokens = build_ccd_hls_gen_v2_prompt(
        files["description"],
        files["tb"],
        files["header"],
        files["kernel"],
        candidate_atoms,
        token_budget=prompt_budget,
        baseline_prompt_tokens=estimate_tokens(zero_prompt),
        hls_skill_capsule=skill_capsule,
    )
    dropped = dropped + [atom for atom in candidate_atoms if atom not in selected]
    write_text(workdir / "prompt.txt", prompt)
    write_text(workdir / "selected_atoms.json", json_dumps([a.model_dump() for a in selected]))
    write_text(workdir / "dropped_atoms.json", json_dumps([a.model_dump() for a in dropped]))
    write_text(workdir / "selected_skills.json", json_dumps([skill.model_dump() for skill in selected_skills]))
    write_text(workdir / "dropped_skills.json", json_dumps([skill.model_dump() for skill in dropped_skills]))

    client = build_model_client(model)
    can_parse = False
    error = None
    prompt_tokens = 0
    completion_tokens = 0
    llm_duration_ms = 0
    llm_calls_used = 0
    tool_calls = 0
    flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
    stage_records: list[dict[str, Any]] = []
    failure_capsules: list[dict[str, Any]] = []
    candidate_manifest: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    pending_memory_updates: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "output_contract": "output_code_xml",
        "parse_mode": None,
        "fallback_reason": None,
        "stage_status": "INIT",
        "task_mode": task_mode.value,
        "terminal_stage": None,
        "stopped_reason": None,
        "llm_calls_used": 0,
        "max_llm_calls": max_llm_calls,
        "budget_summary": budget.summary(),
        "repair_log_token_budget": repair_log_token_budget,
        "early_stop_enabled": early_stop_similarity_threshold > 0,
        "early_stop_similarity_threshold": early_stop_similarity_threshold,
        "early_stop_triggered": False,
        "early_stop_similarity": None,
        "attempt_count": 0,
        "repair_rounds": 0,
        "budget_exhausted": False,
        "failure_capsules": failure_capsules,
        "stage_records": stage_records,
        "format_repair_attempted": False,
        "format_repair_mode": None,
        "format_repair_prompt_tokens": 0,
        "format_repair_completion_tokens": 0,
        "format_repair_duration_ms": 0,
        "source_has_diff_marker": False,
        "selected_atoms": len(selected),
        "dropped_atoms": len(dropped),
        "selected_atoms_tokens": sum(atom.token_estimate for atom in selected),
        "selected_skills": [skill.skill_id for skill in selected_skills],
        "selected_skill_tokens": sum(skill.token_estimate for skill in selected_skills),
        "dropped_skills": len(dropped_skills),
        "deterministic_repair_enabled": enable_deterministic_repair,
        "deterministic_repair_attempts": 0,
        "deterministic_repair_applied": 0,
        "local_memory_enabled": enable_local_memory,
        "local_memory_path": str(memory_path) if memory_path else None,
        "local_memory_hits": 0,
        "action_memory_hits": 0,
        "local_memory_positive_updates": 0,
        "local_memory_negative_updates": 0,
        "action_memory_positive_updates": 0,
        "action_memory_negative_updates": 0,
        "diagnosis_assertions": 0,
        "selected_actions": [],
        "candidate_count": candidate_count,
        "candidate_policy": candidate_policy,
        "candidate_evaluations": 0,
        "selected_candidate_score": None,
        "action_candidate_applied": 0,
        "max_prompt_tokens": max_prompt_tokens,
        "zero_shot_prompt_tokens_est": estimate_tokens(zero_prompt),
        "frontier": frontier,
    }

    def set_terminal(stage: str, reason: str) -> None:
        metrics["terminal_stage"] = stage
        metrics["stopped_reason"] = reason
        metrics["stage_status"] = "FAILED" if stage != "DONE" else "DONE"
        metrics["budget_exhausted"] = reason.startswith("budget_exhausted") or (llm_calls_used >= max_llm_calls and stage != "DONE")

    def case_family() -> str:
        return case_path.parent.name

    def top_signature() -> str:
        top = read_text(workdir / "design" / "top.txt").strip()
        header_text = read_text(files["header"])
        match = re.search(rf"(?m)^\s*[^;{{}}]*\b{re.escape(top)}\s*\([^;{{}}]*\)\s*;", header_text) if top else None
        return " ".join(match.group(0).split()) if match else ""

    def memory_hits_for(capsule: dict[str, Any]) -> tuple[str, list[Any]]:
        if local_memory is None:
            return "- Local memory disabled.", []
        signature = error_signature_from_capsule(capsule)
        hits = local_memory.search(
            case_family=case_family(),
            failure_class=str(capsule.get("failure_class") or ""),
            error_signature=signature,
        )
        metrics["local_memory_hits"] = int(metrics.get("local_memory_hits") or 0) + len(hits)
        return render_memory_capsule(hits), hits

    def action_memory_hits_for(capsule: dict[str, Any], action_ids: list[str]) -> tuple[str, list[Any]]:
        if local_memory is None:
            return "- Action memory disabled.", []
        signature = error_signature_from_capsule(capsule)
        hits = local_memory.search_action_memory(
            case_family=case_family(),
            failure_class=str(capsule.get("failure_class") or ""),
            error_signature=signature,
            action_ids=action_ids,
        )
        metrics["action_memory_hits"] = int(metrics.get("action_memory_hits") or 0) + len(hits)
        return render_action_memory_capsule(hits), hits

    def write_repair_context_artifacts(
        *,
        stage: str,
        attempt: int,
        capsule: dict[str, Any],
        action_memory_hits: list[Any],
        selected_actions: list[Any],
        assertion: Any,
    ) -> dict[str, str]:
        assertion_path = write_json_artifact(workdir / f"diagnosis_assertion_{stage.lower()}_attempt_{attempt}.json", assertion.to_dict())
        selected_actions_path = write_json_artifact(
            workdir / f"selected_actions_{stage.lower()}_attempt_{attempt}.json",
            [action.to_dict() for action in selected_actions],
        )
        write_json_artifact(workdir / "selected_actions.json", [action.to_dict() for action in selected_actions])
        action_memory_path = write_json_artifact(
            workdir / f"action_memory_hits_{stage.lower()}_attempt_{attempt}.json",
            [getattr(hit, "__dict__", str(hit)) for hit in action_memory_hits],
        )
        metrics["diagnosis_assertions"] = int(metrics.get("diagnosis_assertions") or 0) + 1
        metrics["selected_actions"] = [action.action_id for action in selected_actions]
        return {
            "diagnosis_assertion": assertion_path,
            "selected_actions": selected_actions_path,
            "action_memory_hits": action_memory_path,
        }

    def prepare_repair_context(
        *,
        stage: str,
        attempt: int,
        capsule: dict[str, Any],
        task_mode_value: str,
    ) -> dict[str, Any]:
        preliminary_actions = action_ids_for_failure_class(str(capsule.get("failure_class") or ""))
        action_memory_capsule, action_hits = action_memory_hits_for(capsule, preliminary_actions)
        memory_refs = [f"action_memory#{getattr(hit, 'memory_id', '?')}" for hit in action_hits]
        assertion = build_diagnosis_assertion(
            case_name=case_path.name,
            stage=stage,
            task_mode=task_mode_value,
            failure_capsule=capsule,
            attempt=attempt,
            memory_refs=memory_refs,
            early_stop_similarity_threshold=early_stop_similarity_threshold,
        )
        selected_actions = select_repair_actions(assertion, action_memory_hits=action_hits, limit=3)
        artifacts = write_repair_context_artifacts(
            stage=stage,
            attempt=attempt,
            capsule=capsule,
            action_memory_hits=action_hits,
            selected_actions=selected_actions,
            assertion=assertion,
        )
        action_capsule = render_action_capsule(
            assertion,
            selected_actions,
            action_memory_capsule=action_memory_capsule,
        )
        return {
            "assertion": assertion,
            "selected_actions": selected_actions,
            "action_ids": action_ids_from_selected(selected_actions),
            "action_capsule": action_capsule,
            "action_memory_hits": action_hits,
            "artifacts": artifacts,
        }

    def update_memory(
        *,
        capsule: dict[str, Any],
        attempted_fix: str,
        before: dict[str, Any],
        after: dict[str, Any] | None,
        verified: bool,
        assertion: Any | None = None,
        action_id: str | None = None,
        action_params: dict[str, Any] | None = None,
    ) -> None:
        if local_memory is None:
            return
        local_memory.add_event(
            case_name=case_path.name,
            case_family=case_family(),
            top_signature=top_signature(),
            failure_class=str(capsule.get("failure_class") or capsule.get("failure_type") or ""),
            error_signature=error_signature_from_capsule(capsule),
            attempted_fix=attempted_fix,
            tool_result_before=before,
            tool_result_after=after,
            verified=verified,
            artifact_uri=str(workdir),
        )
        key = "local_memory_positive_updates" if verified else "local_memory_negative_updates"
        metrics[key] = int(metrics.get(key) or 0) + 1
        if action_id:
            local_memory.add_action_event(
                case_name=case_path.name,
                case_family=case_family(),
                diagnosis_claim=str(getattr(assertion, "claim", "") if assertion else capsule.get("recommended_policy") or ""),
                failure_class=str(capsule.get("failure_class") or capsule.get("failure_type") or ""),
                error_signature=error_signature_from_capsule(capsule),
                action_id=action_id,
                action_params=action_params or {"attempted_fix": attempted_fix},
                polarity="positive" if verified else "negative",
                before_flags=before,
                after_flags=after,
                verified=verified,
                reuse_conditions={
                    "case_family": case_family(),
                    "failure_class": str(capsule.get("failure_class") or ""),
                },
                anti_reuse_conditions={} if verified else {"avoid_same_action_on_same_error_signature": True},
                artifact_uri=str(workdir),
            )
            action_key = "action_memory_positive_updates" if verified else "action_memory_negative_updates"
            metrics[action_key] = int(metrics.get(action_key) or 0) + 1

    def remember_pending(
        capsule: dict[str, Any],
        attempted_fix: str,
        before: dict[str, Any],
        *,
        assertion: Any | None = None,
        action_id: str | None = None,
        action_params: dict[str, Any] | None = None,
    ) -> None:
        pending_memory_updates.append(
            {
                "capsule": capsule,
                "attempted_fix": attempted_fix,
                "before": before,
                "assertion": assertion,
                "action_id": action_id,
                "action_params": action_params,
            }
        )

    def verify_pending(after: dict[str, Any], *, success: bool) -> None:
        if not pending_memory_updates:
            return
        pending = pending_memory_updates.pop()
        update_memory(
            capsule=pending["capsule"],
            attempted_fix=pending["attempted_fix"],
            before=pending["before"],
            after=after,
            verified=success,
            assertion=pending.get("assertion"),
            action_id=pending.get("action_id"),
            action_params=pending.get("action_params"),
        )

    def deterministic_repair_for(
        capsule: dict[str, Any],
        *,
        build_dir: Path | None = None,
        action_ids: list[str] | None = None,
        assertion: Any | None = None,
    ) -> bool:
        if not enable_deterministic_repair:
            return False
        metrics["deterministic_repair_attempts"] = int(metrics.get("deterministic_repair_attempts") or 0) + 1
        repair = apply_deterministic_repair(
            case_dir=workdir / "design",
            kernel=files["kernel"],
            header=files["header"],
            tb=files["tb"],
            failure_capsule=capsule,
            build_dir=build_dir,
            action_ids=action_ids,
        )
        if repair.applied:
            metrics["deterministic_repair_applied"] = int(metrics.get("deterministic_repair_applied") or 0) + 1
            add_stage_record(
                stage_records,
                stage="DETERMINISTIC_REPAIR",
                status="completed",
                message=repair.message,
                metrics={
                    "repair_type": repair.repair_type,
                    "failure_class": capsule.get("failure_class"),
                    "assertion_id": getattr(assertion, "assertion_id", None),
                    "recommended_actions": action_ids or [],
                    "not_applicable_reason": repair.not_applicable_reason,
                },
                artifacts={"code_snapshot": snapshot_kernel(files["kernel"], workdir, f"deterministic_repair_{metrics['deterministic_repair_attempts']}")},
            )
            return True
        add_stage_record(
            stage_records,
            stage="DETERMINISTIC_REPAIR",
            status="skipped",
            message=repair.message,
            metrics={
                "repair_type": repair.repair_type,
                "failure_class": capsule.get("failure_class"),
                "assertion_id": getattr(assertion, "assertion_id", None),
                "recommended_actions": action_ids or [],
                "not_applicable_reason": repair.not_applicable_reason,
            },
        )
        return False

    def candidate_case_files(candidate_design: Path) -> dict[str, Path]:
        return {
            "header": find_header(candidate_design),
            "tb": find_tb(candidate_design),
            "kernel": find_kernel_cpp(candidate_design),
            "description": candidate_design / "kernel_description.md",
        }

    def copy_candidate_design(stage: str, attempt: int, candidate_id: int) -> tuple[Path, Path, dict[str, Path]]:
        candidate_root = workdir / "candidates" / f"{stage.lower()}_attempt_{attempt}" / f"candidate_{candidate_id}"
        if candidate_root.exists():
            shutil.rmtree(candidate_root)
        candidate_design = candidate_root / "design"
        shutil.copytree(workdir / "design", candidate_design)
        return candidate_root, candidate_design, candidate_case_files(candidate_design)

    def persist_candidate_artifacts(selected_candidate: dict[str, Any] | None = None) -> None:
        write_json_artifact(workdir / "candidate_manifest.json", candidate_manifest)
        write_json_artifact(workdir / "candidate_results.json", candidate_results)
        if selected_candidate is not None:
            write_json_artifact(workdir / "selected_candidate.json", selected_candidate)

    def negative_memory_penalty(action_id: str, action_hits: list[Any]) -> float:
        return 50.0 if any(str(getattr(hit, "action_id", "")) == action_id and str(getattr(hit, "polarity", "")) == "negative" for hit in action_hits) else 0.0

    async def evaluate_candidate(
        *,
        stage: str,
        attempt: int,
        candidate_id: int,
        action_id: str,
        source: str,
        candidate_root: Path,
        candidate_design: Path,
        candidate_files: dict[str, Path],
        token_cost: int = 0,
        parse_ok: bool = True,
        parse_message: str = "",
        not_applicable_reason: str | None = None,
        neg_penalty: float = 0.0,
    ) -> dict[str, Any]:
        nonlocal tool_calls
        top_function = read_text(candidate_design / "top.txt").strip()
        flags_candidate = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
        candidate_metrics: dict[str, Any] = {}
        artifacts: dict[str, str] = {"code_snapshot": str(candidate_files["kernel"])}
        status = "skipped"
        message = not_applicable_reason or parse_message
        if parse_ok and not not_applicable_reason:
            if not budget.consume("csim_calls", stage=f"{stage}_CANDIDATE", label=f"attempt_{attempt}_candidate_{candidate_id}"):
                message = budget.exhausted_reason("csim_calls")
            else:
                csim_build = candidate_root / "build" / "csim"
                csim_flags, csim_metrics, csim_tool = await run_csim_stage(
                    backend_kind,
                    hls_eval_root,
                    hls_part,
                    hls_platform,
                    candidate_design,
                    csim_build,
                    top_function,
                )
                tool_calls += 1
                flags_candidate.update(csim_flags)
                candidate_metrics.update(csim_metrics)
                artifacts["csim_result"] = write_json_artifact(
                    candidate_root / "hls_result_csim.json",
                    {"flags": csim_flags, "metrics": csim_metrics, "return_code": csim_tool.return_code, "command": csim_tool.command},
                )
                status = "evaluated"
                message = "CSIM evaluated."
                if csim_flags["can_compile"] and csim_flags["can_pass_testbench"]:
                    if not budget.consume("synth_calls", stage=f"{stage}_CANDIDATE", label=f"attempt_{attempt}_candidate_{candidate_id}"):
                        message = budget.exhausted_reason("synth_calls")
                    else:
                        synth_build = candidate_root / "build" / "synth"
                        can_synth_candidate, synth_metrics, synth_tool = await run_synth_stage(
                            backend_kind,
                            hls_eval_root,
                            hls_part,
                            hls_platform,
                            candidate_design,
                            synth_build,
                            top_function,
                        )
                        tool_calls += 1
                        flags_candidate["can_synthesize"] = can_synth_candidate
                        candidate_metrics.update(synth_metrics)
                        artifacts["synth_result"] = write_json_artifact(
                            candidate_root / "hls_result_synth.json",
                            {
                                "can_synthesize": can_synth_candidate,
                                "metrics": synth_metrics,
                                "return_code": synth_tool.return_code,
                                "command": synth_tool.command,
                            },
                        )
                        message = "CSIM and SYNTH evaluated."
        result = {
            "stage": stage,
            "attempt": attempt,
            "candidate_id": candidate_id,
            "action_id": action_id,
            "source": source,
            "status": status,
            "message": message,
            "not_applicable_reason": not_applicable_reason,
            "parse_ok": parse_ok,
            "parse_message": parse_message,
            "flags": flags_candidate,
            "metrics": candidate_metrics,
            "total_tokens": token_cost,
            "diff_risk_penalty": 0.0,
            "negative_memory_penalty": neg_penalty,
            "kernel_path": str(candidate_files["kernel"]),
            "artifacts": artifacts,
        }
        result["score"] = candidate_score(result)
        candidate_results.append(make_json_safe(result))
        metrics["candidate_evaluations"] = int(metrics.get("candidate_evaluations") or 0) + 1
        metrics["budget_summary"] = budget.summary()
        return result

    async def run_repair_candidates(
        *,
        stage: str,
        repair_stage: str,
        attempt: int,
        capsule: dict[str, Any],
        context: dict[str, Any],
        local_memory_capsule: str,
        before: dict[str, Any],
    ) -> tuple[bool, dict[str, Any] | None]:
        if candidate_count <= 1:
            return False, None
        action_ids = context["action_ids"]
        selected_actions = context["selected_actions"]
        action_hits = context["action_memory_hits"]
        action_capsule = context["action_capsule"]
        candidates_for_attempt: list[dict[str, Any]] = []

        candidate_root, candidate_design, candidate_files = copy_candidate_design(stage, attempt, 0)
        deterministic_repair = apply_deterministic_repair(
            case_dir=candidate_design,
            kernel=candidate_files["kernel"],
            header=candidate_files["header"],
            tb=candidate_files["tb"],
            failure_capsule=capsule,
            build_dir=candidate_root / "build" / "preflight",
            action_ids=action_ids,
        )
        candidate_manifest.append(
            {
                "stage": stage,
                "attempt": attempt,
                "candidate_id": 0,
                "source": "deterministic_action_memory_replay",
                "action_ids": action_ids,
                "candidate_dir": str(candidate_root),
                "applied": deterministic_repair.applied,
                "not_applicable_reason": deterministic_repair.not_applicable_reason,
            }
        )
        if deterministic_repair.applied:
            candidates_for_attempt.append(
                await evaluate_candidate(
                    stage=stage,
                    attempt=attempt,
                    candidate_id=0,
                    action_id=",".join(action_ids) if action_ids else "DETERMINISTIC",
                    source="deterministic_action_memory_replay",
                    candidate_root=candidate_root,
                    candidate_design=candidate_design,
                    candidate_files=candidate_files,
                    parse_message=deterministic_repair.message,
                    neg_penalty=sum(negative_memory_penalty(action_id, action_hits) for action_id in action_ids),
                )
            )
        else:
            skipped = {
                "stage": stage,
                "attempt": attempt,
                "candidate_id": 0,
                "action_id": ",".join(action_ids) if action_ids else "DETERMINISTIC",
                "source": "deterministic_action_memory_replay",
                "status": "skipped",
                "message": deterministic_repair.message,
                "not_applicable_reason": deterministic_repair.not_applicable_reason,
                "flags": {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False},
                "total_tokens": 0,
                "negative_memory_penalty": sum(negative_memory_penalty(action_id, action_hits) for action_id in action_ids),
                "score": -1.0,
                "kernel_path": str(candidate_files["kernel"]),
                "artifacts": {"code_snapshot": str(candidate_files["kernel"])},
            }
            candidate_results.append(make_json_safe(skipped))

        llm_candidates = max(0, candidate_count - 1)
        for offset, action in enumerate(selected_actions[:llm_candidates], start=1):
            if llm_calls_used >= max_llm_calls:
                break
            candidate_root, candidate_design, candidate_files = copy_candidate_design(stage, attempt, offset)
            prompt_capsule = render_action_capsule(context["assertion"], [action], action_memory_capsule=render_action_memory_capsule(action_hits))
            combined_memory_capsule = prompt_capsule + "\n\n## Local Memory\n" + local_memory_capsule
            repair_prompt = build_hls_repair_prompt(
                stage=stage,
                kernel=files["kernel"],
                header=files["header"],
                tb=files["tb"],
                description=files["description"],
                failure_capsule=capsule,
                failure_history=failure_capsules,
                attempt=metrics["repair_rounds"],
                max_llm_calls=max_llm_calls,
                hls_skill_capsule=skill_capsule,
                local_memory_capsule=combined_memory_capsule,
                token_budget=llm_prompt_budget(model, "You repair Vitis HLS C/C++ code. Return only the requested XML OUTPUT_CODE block."),
            )
            repair_result = await call_llm(
                repair_stage,
                repair_prompt,
                "You repair Vitis HLS C/C++ code. Return only the requested XML OUTPUT_CODE block.",
                label=f"{stage.lower()}_candidate_{offset}_attempt_{attempt}",
            )
            candidate_manifest.append(
                {
                    "stage": stage,
                    "attempt": attempt,
                    "candidate_id": offset,
                    "source": "llm_constrained_action",
                    "action_ids": [action.action_id],
                    "candidate_dir": str(candidate_root),
                    "applied": repair_result is not None,
                }
            )
            if repair_result is None:
                continue
            ok, parse_message, parse_mode = write_kernel_output_code_with_local_repair(repair_result.content, candidate_files["kernel"])
            parse_artifact = write_json_artifact(
                candidate_root / "parse_result.json",
                {"ok": ok, "message": parse_message, "parse_mode": parse_mode, "source_has_diff_marker": source_has_diff_marker(candidate_design)},
            )
            if not ok:
                result = {
                    "stage": stage,
                    "attempt": attempt,
                    "candidate_id": offset,
                    "action_id": action.action_id,
                    "source": "llm_constrained_action",
                    "status": "parse_failed",
                    "message": parse_message,
                    "parse_ok": False,
                    "flags": {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False},
                    "total_tokens": repair_result.prompt_tokens + repair_result.completion_tokens,
                    "negative_memory_penalty": negative_memory_penalty(action.action_id, action_hits),
                    "kernel_path": str(candidate_files["kernel"]),
                    "artifacts": {"parse_result": parse_artifact},
                }
                result["score"] = candidate_score(result)
                candidate_results.append(make_json_safe(result))
                candidates_for_attempt.append(result)
                continue
            evaluated = await evaluate_candidate(
                stage=stage,
                attempt=attempt,
                candidate_id=offset,
                action_id=action.action_id,
                source="llm_constrained_action",
                candidate_root=candidate_root,
                candidate_design=candidate_design,
                candidate_files=candidate_files,
                token_cost=repair_result.prompt_tokens + repair_result.completion_tokens,
                parse_ok=True,
                parse_message=parse_message,
                neg_penalty=negative_memory_penalty(action.action_id, action_hits),
            )
            evaluated["artifacts"]["parse_result"] = parse_artifact
            candidates_for_attempt.append(evaluated)

        selected_candidate = select_best_candidate(candidates_for_attempt)
        if selected_candidate:
            metrics["selected_candidate_score"] = selected_candidate.get("score")
            selected_candidate["selected"] = selected_candidate.get("status") == "evaluated" and bool(selected_candidate.get("kernel_path"))
        persist_candidate_artifacts(selected_candidate)
        add_stage_record(
            stage_records,
            stage="CANDIDATE_RERANK",
            status="completed" if selected_candidate else "skipped",
            message="Selected best repair candidate by Vitis tool result." if selected_candidate else "No repair candidate was evaluable.",
            metrics={
                "attempt": attempt,
                "candidate_count": candidate_count,
                "selected_candidate": selected_candidate,
            },
            artifacts={
                "candidate_manifest": str(workdir / "candidate_manifest.json"),
                "candidate_results": str(workdir / "candidate_results.json"),
                "selected_candidate": str(workdir / "selected_candidate.json"),
            },
        )
        if not selected_candidate or not selected_candidate.get("selected"):
            return False, selected_candidate
        shutil.copy2(Path(str(selected_candidate["kernel_path"])), files["kernel"])
        metrics["action_candidate_applied"] = int(metrics.get("action_candidate_applied") or 0) + 1
        if selected_candidate.get("source") == "deterministic_action_memory_replay":
            metrics["deterministic_repair_applied"] = int(metrics.get("deterministic_repair_applied") or 0) + 1
        remember_pending(
            capsule,
            f"candidate:{selected_candidate.get('source')}:{selected_candidate.get('action_id')}",
            before,
            assertion=context["assertion"],
            action_id=str(selected_candidate.get("action_id") or ""),
            action_params={"candidate_id": selected_candidate.get("candidate_id"), "score": selected_candidate.get("score")},
        )
        return True, selected_candidate

    async def call_llm(stage: str, prompt_text: str, system_prompt: str, *, label: str):
        nonlocal selected_skills
        nonlocal prompt_tokens, completion_tokens, llm_duration_ms, llm_calls_used
        if llm_calls_used >= max_llm_calls:
            metrics["budget_summary"] = budget.summary()
            return None
        if not budget.consume("llm_calls", stage=stage, label=label):
            metrics["budget_summary"] = budget.summary()
            return None
        llm_calls_used += 1
        call_prefix = f"llm_call_{llm_calls_used:02d}_{label}"
        prompt_path = workdir / f"{call_prefix}_prompt.txt"
        response_path = workdir / f"{call_prefix}_response.txt"
        prompt_text, prompt_fit_metrics = fit_prompt_for_model(prompt_text, model, system_prompt)
        metrics["prompt_fit_last"] = prompt_fit_metrics
        write_text(prompt_path, prompt_text)
        try:
            result = await client.complete(prompt_text, system_prompt=system_prompt)
        except Exception as exc:
            metrics["llm_calls_used"] = llm_calls_used
            add_stage_record(
                stage_records,
                stage=stage,
                status="failed",
                message=f"LLM call {llm_calls_used}/{max_llm_calls} failed: {exc}",
                metrics={
                    "llm_calls_used": llm_calls_used,
                    "prompt_tokens_est": estimate_tokens(prompt_text),
                    **prompt_fit_metrics,
                    "skill_ids": [skill.skill_id for skill in selected_skills],
                    "skill_tokens": sum(skill.token_estimate for skill in selected_skills),
                    "skill_sources": [skill.source_uri for skill in selected_skills],
                },
                artifacts={"prompt": str(prompt_path), "response": str(response_path)},
            )
            metrics["budget_summary"] = budget.summary()
            raise
        write_text(response_path, result.content)
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        llm_duration_ms += result.duration_ms
        metrics["llm_calls_used"] = llm_calls_used
        add_stage_record(
            stage_records,
            stage=stage,
            status="completed",
            message=f"LLM call {llm_calls_used}/{max_llm_calls} completed.",
            metrics={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "duration_ms": result.duration_ms,
                "llm_calls_used": llm_calls_used,
                **prompt_fit_metrics,
                "skill_ids": [skill.skill_id for skill in selected_skills],
                "skill_tokens": sum(skill.token_estimate for skill in selected_skills),
                "skill_sources": [skill.source_uri for skill in selected_skills],
            },
            artifacts={"prompt": str(prompt_path), "response": str(response_path)},
        )
        metrics["budget_summary"] = budget.summary()
        return result

    async def parse_model_output(stage: str, content: str, *, label: str) -> tuple[bool, str | None]:
        nonlocal can_parse, llm_calls_used
        ok, parse_message, parse_mode = write_kernel_output_code_with_local_repair(content, files["kernel"])
        parse_artifact = write_json_artifact(
            workdir / f"parse_result_{label}.json",
            {"ok": ok, "message": parse_message, "parse_mode": parse_mode, "source_has_diff_marker": source_has_diff_marker(workdir / "design")},
        )
        if ok:
            can_parse = True
            metrics["parse_mode"] = parse_mode or "output_code_exact"
            metrics["parse_message"] = parse_message
            if parse_mode and parse_mode.startswith("local_secondary_parse"):
                metrics["format_repair_attempted"] = True
                metrics["format_repair_mode"] = parse_mode
            add_stage_record(
                stage_records,
                stage="PARSE_VALIDATE",
                status="completed",
                message=parse_message,
                metrics={"parse_mode": metrics["parse_mode"]},
                artifacts={"parse_result": parse_artifact, "code_snapshot": snapshot_kernel(files["kernel"], workdir, label)},
            )
            return True, parse_mode

        metrics["fallback_reason"] = parse_message
        add_stage_record(
            stage_records,
            stage="PARSE_VALIDATE",
            status="failed",
            message=parse_message,
            metrics={"llm_calls_used": llm_calls_used},
            artifacts={"parse_result": parse_artifact},
        )
        if "No <OUTPUT_CODE> block found" not in parse_message:
            return False, None
        if llm_calls_used >= max_llm_calls:
            return False, None

        repair_prompt = build_output_code_repair_prompt(files["kernel"].name, content)
        metrics["format_repair_attempted"] = True
        metrics["format_repair_mode"] = "llm_rewrap_output_code"
        metrics["format_repair_prompt_tokens"] = estimate_tokens(repair_prompt)
        repair_result = await call_llm(
            "FORMAT_REPAIR",
            repair_prompt,
            "You repair format only. Return exactly one XML OUTPUT_CODE block and no other text.",
            label=f"{label}_format_repair",
        )
        if repair_result is None:
            return False, None
        metrics["format_repair_prompt_tokens"] = repair_result.prompt_tokens
        metrics["format_repair_completion_tokens"] = repair_result.completion_tokens
        metrics["format_repair_duration_ms"] = repair_result.duration_ms
        ok, parse_message = write_kernel_output_code(repair_result.content, files["kernel"])
        parse_mode = "llm_rewrap_output_code" if ok else None
        repair_parse_artifact = write_json_artifact(
            workdir / f"parse_result_{label}_format_repair.json",
            {"ok": ok, "message": parse_message, "parse_mode": parse_mode, "source_has_diff_marker": source_has_diff_marker(workdir / "design")},
        )
        metrics["parse_message"] = parse_message
        metrics["source_has_diff_marker"] = source_has_diff_marker(workdir / "design")
        if not ok:
            metrics["fallback_reason"] = parse_message
            add_stage_record(
                stage_records,
                stage="FORMAT_REPAIR",
                status="failed",
                message=parse_message,
                metrics={"llm_calls_used": llm_calls_used},
                artifacts={"parse_result": repair_parse_artifact},
            )
            return False, None
        can_parse = True
        metrics["parse_mode"] = parse_mode
        add_stage_record(
            stage_records,
            stage="FORMAT_REPAIR",
            status="completed",
            message=parse_message,
            metrics={"parse_mode": parse_mode, "llm_calls_used": llm_calls_used},
            artifacts={"parse_result": repair_parse_artifact, "code_snapshot": snapshot_kernel(files["kernel"], workdir, f"{label}_format_repair")},
        )
        return True, parse_mode

    try:
        metrics["stage_status"] = "GENERATION"
        result = await call_llm(
            "GENERATION",
            prompt,
            "You are a precise HLS C/C++ generation engine. Return only the requested XML OUTPUT_CODE block.",
            label="generation",
        )
        if result is None:
            set_terminal("GENERATION", "max_llm_calls_exhausted_before_generation")
            raise ValueError(metrics["stopped_reason"])
        write_text(workdir / "response.txt", result.content)
        parsed, _ = await parse_model_output("PARSE_VALIDATE", result.content, label="generation")
        if not parsed:
            set_terminal("PARSE_VALIDATE", metrics.get("fallback_reason") or "parse_failed")
            raise ValueError(metrics["stopped_reason"])

        validation_attempt = 0
        while True:
            validation_attempt += 1
            metrics["attempt_count"] = validation_attempt
            top_function = read_text(workdir / "design" / "top.txt").strip()
            metrics["stage_status"] = "CSIM"
            if not budget.consume("csim_calls", stage="CSIM", label=f"attempt_{validation_attempt}"):
                metrics["budget_summary"] = budget.summary()
                set_terminal("CSIM", budget.exhausted_reason("csim_calls"))
                add_stage_record(
                    stage_records,
                    stage="CSIM",
                    status="skipped",
                    message="CSIM skipped because the CSIM budget is exhausted.",
                    metrics={"attempt": validation_attempt, "budget_summary": budget.summary()},
                )
                break
            csim_build = workdir / "build" / f"csim_attempt_{validation_attempt}"
            csim_flags, csim_metrics, csim_result = await run_csim_stage(
                backend_kind,
                hls_eval_root,
                hls_part,
                hls_platform,
                workdir / "design",
                csim_build,
                top_function,
            )
            tool_calls += 1
            metrics["budget_summary"] = budget.summary()
            flags.update(csim_flags)
            task_mode = classify_task_mode(
                generation_stub=is_generation_stub(files["kernel"]),
                csim_flags=csim_flags,
            )
            metrics["task_mode"] = task_mode.value
            metrics.update(csim_metrics)
            hls_artifact = write_json_artifact(
                workdir / f"hls_result_csim_attempt_{validation_attempt}.json",
                {"flags": csim_flags, "metrics": csim_metrics, "return_code": csim_result.return_code, "command": csim_result.command},
            )
            if csim_flags["can_compile"] and csim_flags["can_pass_testbench"]:
                verify_pending({"stage": "CSIM", **csim_flags}, success=True)
                add_stage_record(
                    stage_records,
                    stage="CSIM",
                    status="completed",
                    message=f"CSIM passed at attempt {validation_attempt}.",
                    metrics={"attempt": validation_attempt, **csim_flags},
                    artifacts={"hls_result": hls_artifact},
                )
                flags["can_synthesize"] = False
            else:
                capsule = build_failure_capsule("CSIM", csim_result, token_budget=repair_log_token_budget)
                failure_capsules.append(capsule)
                capsule_path = write_json_artifact(workdir / f"failure_capsule_csim_attempt_{validation_attempt}.json", capsule)
                selected_skills, dropped_skills = route_skills(
                    task_mode=metrics.get("task_mode") or task_mode.value,
                    scan=scan,
                    failure_capsule=capsule,
                    latest_metrics=csim_metrics,
                    selected_atoms=selected,
                    token_budget=skill_token_budget,
                )
                skill_capsule = render_skill_capsule(selected_skills)
                local_memory_capsule, memory_hits = memory_hits_for(capsule)
                repair_context = prepare_repair_context(
                    stage="CSIM",
                    attempt=validation_attempt,
                    capsule=capsule,
                    task_mode_value=str(metrics.get("task_mode") or task_mode.value),
                )
                write_text(workdir / "selected_skills.json", json_dumps([skill.model_dump() for skill in selected_skills]))
                write_text(workdir / "dropped_skills.json", json_dumps([skill.model_dump() for skill in dropped_skills]))
                write_text(workdir / "local_memory_hits.json", json_dumps([getattr(hit, "__dict__", str(hit)) for hit in memory_hits]))
                add_stage_record(
                    stage_records,
                    stage="CSIM",
                    status="failed",
                    message=f"CSIM failed at attempt {validation_attempt}.",
                    metrics={
                        "attempt": validation_attempt,
                        **csim_flags,
                        "skill_ids": [skill.skill_id for skill in selected_skills],
                        "skill_tokens": sum(skill.token_estimate for skill in selected_skills),
                        "skill_sources": [skill.source_uri for skill in selected_skills],
                        "assertion_id": repair_context["assertion"].assertion_id,
                        "failure_class": capsule.get("failure_class"),
                        "recommended_actions": repair_context["action_ids"],
                        "confidence": repair_context["assertion"].confidence,
                    },
                    artifacts={"hls_result": hls_artifact, "failure_capsule": capsule_path, **repair_context["artifacts"]},
                )
                if candidate_count > 1:
                    metrics["repair_rounds"] += 1
                    applied_candidate, _selected_candidate = await run_repair_candidates(
                        stage="CSIM",
                        repair_stage="CSIM_REPAIR",
                        attempt=validation_attempt,
                        capsule=capsule,
                        context=repair_context,
                        local_memory_capsule=local_memory_capsule,
                        before={"stage": "CSIM", **csim_flags},
                    )
                    if applied_candidate:
                        continue
                    update_memory(
                        capsule=capsule,
                        attempted_fix="candidate_csim_repair_failed",
                        before={"stage": "CSIM", **csim_flags},
                        after={"selected_candidate": _selected_candidate},
                        verified=False,
                        assertion=repair_context["assertion"],
                        action_id=repair_context["action_ids"][0] if repair_context["action_ids"] else "LLM_SEMANTIC_REPAIR",
                        action_params={"candidate_count": candidate_count},
                    )
                    set_terminal("CSIM_REPAIR", "candidate_repair_failed")
                    break
                if deterministic_repair_for(
                    capsule,
                    build_dir=csim_build,
                    action_ids=repair_context["action_ids"],
                    assertion=repair_context["assertion"],
                ):
                    remember_pending(
                        capsule,
                        "deterministic_repair:" + str(capsule.get("recommended_policy") or capsule.get("failure_class") or ""),
                        {"stage": "CSIM", **csim_flags},
                        assertion=repair_context["assertion"],
                        action_id=",".join(repair_context["action_ids"]),
                        action_params={"selected_actions": repair_context["action_ids"]},
                    )
                    continue
                should_stop, similarity = repeated_failure_early_stop(
                    failure_capsules,
                    threshold=early_stop_similarity_threshold,
                )
                metrics["last_failure_similarity"] = similarity
                if early_stop_similarity_threshold > 0 and should_stop:
                    metrics["early_stop_triggered"] = True
                    metrics["early_stop_similarity"] = similarity
                    set_terminal("CSIM", "early_stop_repeated_failure")
                    add_stage_record(
                        stage_records,
                        stage="EARLY_STOP",
                        status="completed",
                        message="Stopped because two consecutive failure capsules are highly similar.",
                        metrics={
                            "attempt": validation_attempt,
                            "similarity": similarity,
                            "threshold": early_stop_similarity_threshold,
                            "failure_type": capsule.get("failure_type"),
                        },
                    )
                    break
                if llm_calls_used >= max_llm_calls:
                    set_terminal("CSIM", "max_llm_calls_exhausted_after_csim_failure")
                    break
                metrics["repair_rounds"] += 1
                repair_prompt = build_hls_repair_prompt(
                    stage="CSIM",
                    kernel=files["kernel"],
                    header=files["header"],
                    tb=files["tb"],
                    description=files["description"],
                    failure_capsule=capsule,
                    failure_history=failure_capsules,
                    attempt=metrics["repair_rounds"],
                    max_llm_calls=max_llm_calls,
                    hls_skill_capsule=skill_capsule,
                    local_memory_capsule=repair_context["action_capsule"] + "\n\n## Local Memory\n" + local_memory_capsule,
                    token_budget=llm_prompt_budget(model, "You repair Vitis HLS C/C++ code. Return only the requested XML OUTPUT_CODE block."),
                )
                repair_result = await call_llm(
                    "CSIM_REPAIR",
                    repair_prompt,
                    "You repair Vitis HLS C/C++ code. Return only the requested XML OUTPUT_CODE block.",
                    label=f"csim_repair_attempt_{validation_attempt}",
                )
                if repair_result is None:
                    set_terminal("CSIM_REPAIR", "max_llm_calls_exhausted_before_csim_repair")
                    break
                parsed, _ = await parse_model_output("PARSE_VALIDATE", repair_result.content, label=f"csim_repair_attempt_{validation_attempt}")
                if not parsed:
                    update_memory(
                        capsule=capsule,
                        attempted_fix="llm_csim_repair_parse_failed",
                        before={"stage": "CSIM", **csim_flags},
                        after={"parse_failed": True},
                        verified=False,
                        assertion=repair_context["assertion"],
                        action_id=repair_context["action_ids"][0] if repair_context["action_ids"] else "LLM_SEMANTIC_REPAIR",
                        action_params={"parse_failed": True},
                    )
                    set_terminal("PARSE_VALIDATE", metrics.get("fallback_reason") or "repair_output_parse_failed")
                    break
                remember_pending(
                    capsule,
                    "llm_csim_repair",
                    {"stage": "CSIM", **csim_flags},
                    assertion=repair_context["assertion"],
                    action_id=repair_context["action_ids"][0] if repair_context["action_ids"] else "LLM_SEMANTIC_REPAIR",
                    action_params={"selected_actions": repair_context["action_ids"]},
                )
                continue

            metrics["stage_status"] = "SYNTH"
            if not budget.consume("synth_calls", stage="SYNTH", label=f"attempt_{validation_attempt}"):
                metrics["budget_summary"] = budget.summary()
                set_terminal("SYNTH", budget.exhausted_reason("synth_calls"))
                add_stage_record(
                    stage_records,
                    stage="SYNTH",
                    status="skipped",
                    message="Synthesis skipped because the SYNTH budget is exhausted.",
                    metrics={"attempt": validation_attempt, "budget_summary": budget.summary()},
                )
                break
            synth_build = workdir / "build" / f"synth_attempt_{validation_attempt}"
            can_synth, synth_metrics, synth_result = await run_synth_stage(
                backend_kind,
                hls_eval_root,
                hls_part,
                hls_platform,
                workdir / "design",
                synth_build,
                top_function,
            )
            tool_calls += 1
            metrics["budget_summary"] = budget.summary()
            flags["can_synthesize"] = can_synth
            task_mode = classify_task_mode(
                generation_stub=is_generation_stub(files["kernel"]),
                csim_flags={"can_compile": flags["can_compile"], "can_pass_testbench": flags["can_pass_testbench"]},
                synth_passed=can_synth,
            )
            metrics["task_mode"] = task_mode.value
            metrics.update(synth_metrics)
            synth_artifact = write_json_artifact(
                workdir / f"hls_result_synth_attempt_{validation_attempt}.json",
                {"can_synthesize": can_synth, "metrics": synth_metrics, "return_code": synth_result.return_code, "command": synth_result.command},
            )
            if can_synth:
                verify_pending({"stage": "SYNTH", "can_synthesize": True}, success=True)
                add_stage_record(
                    stage_records,
                    stage="SYNTH",
                    status="completed",
                    message=f"Synthesis passed at attempt {validation_attempt}.",
                    metrics={"attempt": validation_attempt, "can_synthesize": True},
                    artifacts={"hls_result": synth_artifact},
                )
                set_terminal("DONE", "synth_passed")
                break

            capsule = build_failure_capsule("SYNTH", synth_result, token_budget=repair_log_token_budget)
            failure_capsules.append(capsule)
            capsule_path = write_json_artifact(workdir / f"failure_capsule_synth_attempt_{validation_attempt}.json", capsule)
            selected_skills, dropped_skills = route_skills(
                task_mode=metrics.get("task_mode") or task_mode.value,
                scan=scan,
                failure_capsule=capsule,
                latest_metrics=synth_metrics,
                selected_atoms=selected,
                token_budget=skill_token_budget,
            )
            skill_capsule = render_skill_capsule(selected_skills)
            local_memory_capsule, memory_hits = memory_hits_for(capsule)
            repair_context = prepare_repair_context(
                stage="SYNTH",
                attempt=validation_attempt,
                capsule=capsule,
                task_mode_value=str(metrics.get("task_mode") or task_mode.value),
            )
            write_text(workdir / "selected_skills.json", json_dumps([skill.model_dump() for skill in selected_skills]))
            write_text(workdir / "dropped_skills.json", json_dumps([skill.model_dump() for skill in dropped_skills]))
            write_text(workdir / "local_memory_hits.json", json_dumps([getattr(hit, "__dict__", str(hit)) for hit in memory_hits]))
            add_stage_record(
                stage_records,
                stage="SYNTH",
                status="failed",
                message=f"Synthesis failed at attempt {validation_attempt}.",
                metrics={
                    "attempt": validation_attempt,
                    "can_synthesize": False,
                    "skill_ids": [skill.skill_id for skill in selected_skills],
                    "skill_tokens": sum(skill.token_estimate for skill in selected_skills),
                    "skill_sources": [skill.source_uri for skill in selected_skills],
                    "assertion_id": repair_context["assertion"].assertion_id,
                    "failure_class": capsule.get("failure_class"),
                    "recommended_actions": repair_context["action_ids"],
                    "confidence": repair_context["assertion"].confidence,
                },
                artifacts={"hls_result": synth_artifact, "failure_capsule": capsule_path, **repair_context["artifacts"]},
            )
            if candidate_count > 1:
                metrics["repair_rounds"] += 1
                applied_candidate, _selected_candidate = await run_repair_candidates(
                    stage="SYNTH",
                    repair_stage="SYNTH_REPAIR",
                    attempt=validation_attempt,
                    capsule=capsule,
                    context=repair_context,
                    local_memory_capsule=local_memory_capsule,
                    before={"stage": "SYNTH", "can_synthesize": False},
                )
                if applied_candidate:
                    flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
                    continue
                update_memory(
                    capsule=capsule,
                    attempted_fix="candidate_synth_repair_failed",
                    before={"stage": "SYNTH", "can_synthesize": False},
                    after={"selected_candidate": _selected_candidate},
                    verified=False,
                    assertion=repair_context["assertion"],
                    action_id=repair_context["action_ids"][0] if repair_context["action_ids"] else "SYNTH_INTERFACE_REPAIR",
                    action_params={"candidate_count": candidate_count},
                )
                set_terminal("SYNTH_REPAIR", "candidate_repair_failed")
                break
            if deterministic_repair_for(
                capsule,
                build_dir=synth_build,
                action_ids=repair_context["action_ids"],
                assertion=repair_context["assertion"],
            ):
                remember_pending(
                    capsule,
                    "deterministic_repair:" + str(capsule.get("recommended_policy") or capsule.get("failure_class") or ""),
                    {"stage": "SYNTH", "can_synthesize": False},
                    assertion=repair_context["assertion"],
                    action_id=",".join(repair_context["action_ids"]),
                    action_params={"selected_actions": repair_context["action_ids"]},
                )
                flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
                continue
            should_stop, similarity = repeated_failure_early_stop(
                failure_capsules,
                threshold=early_stop_similarity_threshold,
            )
            metrics["last_failure_similarity"] = similarity
            if early_stop_similarity_threshold > 0 and should_stop:
                metrics["early_stop_triggered"] = True
                metrics["early_stop_similarity"] = similarity
                set_terminal("SYNTH", "early_stop_repeated_failure")
                add_stage_record(
                    stage_records,
                    stage="EARLY_STOP",
                    status="completed",
                    message="Stopped because two consecutive failure capsules are highly similar.",
                    metrics={
                        "attempt": validation_attempt,
                        "similarity": similarity,
                        "threshold": early_stop_similarity_threshold,
                        "failure_type": capsule.get("failure_type"),
                    },
                )
                break
            if llm_calls_used >= max_llm_calls:
                set_terminal("SYNTH", "max_llm_calls_exhausted_after_synth_failure")
                break
            metrics["repair_rounds"] += 1
            repair_prompt = build_hls_repair_prompt(
                stage="SYNTH",
                kernel=files["kernel"],
                header=files["header"],
                tb=files["tb"],
                description=files["description"],
                failure_capsule=capsule,
                failure_history=failure_capsules,
                attempt=metrics["repair_rounds"],
                max_llm_calls=max_llm_calls,
                hls_skill_capsule=skill_capsule,
                local_memory_capsule=repair_context["action_capsule"] + "\n\n## Local Memory\n" + local_memory_capsule,
                token_budget=llm_prompt_budget(model, "You repair Vitis HLS C/C++ code. Return only the requested XML OUTPUT_CODE block."),
            )
            repair_result = await call_llm(
                "SYNTH_REPAIR",
                repair_prompt,
                "You repair Vitis HLS C/C++ code. Return only the requested XML OUTPUT_CODE block.",
                label=f"synth_repair_attempt_{validation_attempt}",
            )
            if repair_result is None:
                set_terminal("SYNTH_REPAIR", "max_llm_calls_exhausted_before_synth_repair")
                break
            parsed, _ = await parse_model_output("PARSE_VALIDATE", repair_result.content, label=f"synth_repair_attempt_{validation_attempt}")
            if not parsed:
                update_memory(
                    capsule=capsule,
                    attempted_fix="llm_synth_repair_parse_failed",
                    before={"stage": "SYNTH", "can_synthesize": False},
                    after={"parse_failed": True},
                    verified=False,
                    assertion=repair_context["assertion"],
                    action_id=repair_context["action_ids"][0] if repair_context["action_ids"] else "SYNTH_INTERFACE_REPAIR",
                    action_params={"parse_failed": True},
                )
                set_terminal("PARSE_VALIDATE", metrics.get("fallback_reason") or "repair_output_parse_failed")
                break
            remember_pending(
                capsule,
                "llm_synth_repair",
                {"stage": "SYNTH", "can_synthesize": False},
                assertion=repair_context["assertion"],
                action_id=repair_context["action_ids"][0] if repair_context["action_ids"] else "SYNTH_INTERFACE_REPAIR",
                action_params={"selected_actions": repair_context["action_ids"]},
            )
            # Any code change after synth repair must be revalidated with CSIM first.
            flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
            continue

        if metrics["terminal_stage"] != "DONE":
            if pending_memory_updates:
                pending = pending_memory_updates.pop()
                update_memory(
                    capsule=pending["capsule"],
                    attempted_fix=pending["attempted_fix"],
                    before=pending["before"],
                    after={"terminal_stage": metrics.get("terminal_stage"), "stopped_reason": metrics.get("stopped_reason")},
                    verified=False,
                    assertion=pending.get("assertion"),
                    action_id=pending.get("action_id"),
                    action_params=pending.get("action_params"),
                )
            error = metrics.get("stopped_reason") or "repair_loop_failed"
    except Exception as exc:
        error = error or str(exc)
        metrics["source_has_diff_marker"] = source_has_diff_marker(workdir / "design")
        if metrics.get("terminal_stage") is None:
            set_terminal(metrics.get("stage_status") or "FAILED", "exception")
    metrics["llm_calls_used"] = llm_calls_used
    metrics["stage_records"] = stage_records
    metrics["failure_capsules"] = failure_capsules
    metrics["source_has_diff_marker"] = source_has_diff_marker(workdir / "design")
    metrics["budget_summary"] = budget.summary()
    write_json_artifact(workdir / "stage_records.json", stage_records)
    write_json_artifact(workdir / "failure_capsules.json", failure_capsules)
    if candidate_manifest or candidate_results:
        persist_candidate_artifacts()
    write_json_artifact(workdir / "budget_ledger.json", budget.model_dump())
    result = CaseResult(
        experiment_id=experiment_id,
        method=method_name,
        sample_idx=sample_idx,
        case_name=case_path.name,
        case_path=str(case_path),
        tags=parse_tags(case_path / "hls_eval_config.toml"),
        can_parse=can_parse,
        can_compile=flags["can_compile"],
        can_pass_testbench=flags["can_pass_testbench"],
        can_synthesize=flags["can_synthesize"],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        llm_duration_ms=llm_duration_ms,
        tool_calls=tool_calls,
        wall_time_ms=int((time.monotonic() - t0) * 1000),
        error=error,
        metrics=metrics,
        artifacts_dir=str(workdir),
    )
    token_report = write_case_token_report(result, workdir)
    metrics["token_report"] = token_report
    workflow_status = write_workflow_artifacts_from_stage_records(
        workdir,
        stage_records,
        final_status="done" if flags["can_synthesize"] else "failed",
        message=metrics.get("stopped_reason") or ("synth_passed" if flags["can_synthesize"] else "run_failed"),
    )
    metrics["workflow_status"] = workflow_status
    return result


def pass_at_k(successes: list[bool], k: int) -> float:
    n = len(successes)
    c = sum(1 for ok in successes if ok)
    if n == 0:
        return 0.0
    if c == 0:
        return 0.0
    if n < k:
        return 1.0 if c > 0 else 0.0
    return 1.0 - (comb(n - c, k) / comb(n, k) if n - c >= k else 0.0)


def build_operator_memory_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        metrics = result.metrics or {}
        failures = metrics.get("failure_capsules", []) or []
        last_failure = failures[-1] if failures else {}
        fix_stage = None
        for record in metrics.get("stage_records", []) or []:
            if str(record.get("stage", "")).endswith("_REPAIR") and record.get("status") == "completed":
                fix_stage = record.get("stage")
        token_report = metrics.get("token_report") or {}
        token_cost = token_report.get("total_tokens", result.total_tokens)
        rows.append(
            make_json_safe(
                {
                    "case_name": result.case_name,
                    "family": Path(result.case_path).parent.name,
                    "failure_type": last_failure.get("failure_type"),
                    "terminal_stage": metrics.get("terminal_stage"),
                    "fix_stage": fix_stage,
                    "synth_passed": result.can_synthesize,
                    "token_cost": token_cost,
                    "artifact_uri": result.artifacts_dir,
                }
            )
        )
    return rows


def summarize(results: list[CaseResult], out_dir: Path, pass_ks: list[int]) -> None:
    rows = [result_to_json_dict(r) for r in results]
    write_text(out_dir / "results.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    token_summary_rows = write_token_summary(results, out_dir)
    memory_rows = build_operator_memory_rows(results)
    write_text(out_dir / "operator_memory.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in memory_rows) + ("\n" if memory_rows else ""))
    fieldnames = [
        "experiment_id",
        "method",
        "sample_idx",
        "case_name",
        "can_parse",
        "can_compile",
        "can_pass_testbench",
        "can_synthesize",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tool_calls",
        "wall_time_ms",
        "error",
        "artifacts_dir",
    ]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    methods = sorted(set(r.method for r in results))
    summary_rows = []
    for method in methods:
        method_results = [r for r in results if r.method == method]
        n = len(method_results)
        base = {
            "method": method,
            "n_samples_total": n,
            "avg_prompt_tokens": sum(r.prompt_tokens for r in method_results) / max(1, n),
            "avg_completion_tokens": sum(r.completion_tokens for r in method_results) / max(1, n),
            "avg_total_tokens": sum(r.total_tokens for r in method_results) / max(1, n),
            "avg_tool_calls": sum(r.tool_calls for r in method_results) / max(1, n),
            "avg_wall_time_ms": sum(r.wall_time_ms for r in method_results) / max(1, n),
        }
        token_row = next((row for row in token_summary_rows if row.get("method") == method), {})
        base["tokens_per_synth_success"] = token_row.get("tokens_per_synth_success")
        base["tokens_per_csim_success"] = token_row.get("tokens_per_csim_success")
        for stage in TOKEN_STAGES:
            base[f"avg_tokens_{stage.lower()}"] = token_row.get(f"avg_tokens_{stage.lower()}", 0)
            base[f"llm_calls_{stage.lower()}"] = token_row.get(f"llm_calls_{stage.lower()}", 0)
            base[f"tool_calls_{stage.lower()}"] = token_row.get(f"tool_calls_{stage.lower()}", 0)
        for metric in ["can_parse", "can_compile", "can_pass_testbench", "can_synthesize"]:
            base[f"{metric}_rate"] = sum(1 for r in method_results if getattr(r, metric)) / max(1, n)
            for k in pass_ks:
                per_case = {}
                for result in method_results:
                    per_case.setdefault(result.case_name, []).append(getattr(result, metric))
                base[f"{metric}_pass@{k}"] = sum(pass_at_k(v, k) for v in per_case.values()) / max(1, len(per_case))
        summary_rows.append(base)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["method"])
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["# HLS-Eval Experiment Summary", "", f"Generated at: {utc_now()}", ""]
    for row in summary_rows:
        lines.append(f"## {row['method']}")
        lines.append("")
        lines.append(f"- total samples: {row['n_samples_total']}")
        lines.append(f"- Can Parse rate: {row['can_parse_rate']:.2%}")
        lines.append(f"- Can Compile rate: {row['can_compile_rate']:.2%}")
        lines.append(f"- Can Pass Testbench rate: {row['can_pass_testbench_rate']:.2%}")
        lines.append(f"- Can Synthesize rate: {row['can_synthesize_rate']:.2%}")
        lines.append(f"- avg prompt tokens: {row['avg_prompt_tokens']:.1f}")
        lines.append(f"- avg completion tokens: {row['avg_completion_tokens']:.1f}")
        lines.append(f"- avg total tokens: {row['avg_total_tokens']:.1f}")
        if row.get("tokens_per_synth_success") is not None:
            lines.append(f"- tokens per synth success: {float(row['tokens_per_synth_success']):.1f}")
        if row.get("tokens_per_csim_success") is not None:
            lines.append(f"- tokens per csim success: {float(row['tokens_per_csim_success']):.1f}")
        lines.append("- avg tokens by stage:")
        for stage in TOKEN_STAGES:
            lines.append(f"  - {stage}: {float(row.get(f'avg_tokens_{stage.lower()}', 0) or 0):.1f}")
        for k in pass_ks:
            lines.append(f"- Can Synthesize pass@{k}: {row.get(f'can_synthesize_pass@{k}', 0):.2%}")
        lines.append("")
    write_text(out_dir / "summary.md", "\n".join(lines))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCD-HLS benchmark methods. HLS-Eval baselines should be run with the original HLS-Eval runners.")
    parser.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", default="ccd_hls_loop")
    parser.add_argument("--hls-backend", choices=["hls_eval", "vitis", "command", "mock"], default="vitis")
    parser.add_argument("--hls-part", default=None, help="FPGA part for Vitis config. Defaults to HLS_PART or the project default.")
    parser.add_argument("--hls-platform", default=None, help="Vitis platform .xpfm path. Defaults to HLS_PLATFORM when set.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-filter", default=None, help="Regex matched against case path/name.")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--prompt-budget", type=int, default=6000)
    parser.add_argument("--max-llm-calls", type=int, default=5)
    parser.add_argument("--llm-call-budget", type=int, default=None, help="LLM call budget. Defaults to --max-llm-calls.")
    parser.add_argument("--csim-budget", type=int, default=None, help="Maximum CSIM calls per case. Omit for unlimited.")
    parser.add_argument("--synth-budget", type=int, default=None, help="Maximum SYNTH calls per case. Omit for unlimited.")
    parser.add_argument("--cosim-budget", type=int, default=0, help="Maximum COSIM calls per case. Default 0 because COSIM is not in M1-M3 loop.")
    parser.add_argument("--unified-credit-budget", type=int, default=None, help="Optional unified budget consumed by every LLM/tool call.")
    parser.add_argument("--skill-token-budget", type=int, default=600)
    parser.add_argument("--repair-log-token-budget", type=int, default=1200)
    parser.add_argument("--disable-deterministic-repair", action="store_true", help="Disable local rule-based repair before LLM repair.")
    parser.add_argument("--disable-local-memory", action="store_true", help="Disable local SQLite HLS memory retrieval/update.")
    parser.add_argument("--memory-path", type=Path, default=None, help="SQLite memory path. Defaults to .hls_agent/memory/hls_memory.sqlite.")
    parser.add_argument("--candidate-count", type=int, default=1, help="Number of repair candidates to evaluate when a repair stage is reached.")
    parser.add_argument("--candidate-policy", default="repair_only", choices=["repair_only"], help="Candidate generation policy. v1 only enables candidates during repair.")
    parser.add_argument(
        "--early-stop-similarity-threshold",
        type=float,
        default=0.92,
        help="Stop a repair loop when two consecutive failure capsules are this similar. Use 0 to disable.",
    )
    parser.add_argument("--pass-k", default="1,5")
    parser.add_argument("--resume", action="store_true", help="Skip method/case/sample runs that already have result.json in --out-dir.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hls_eval_root = args.hls_eval_root.expanduser().resolve()
    data_dir = (args.data_dir or hls_eval_root / "hls_eval_data").expanduser().resolve()
    model_data = normalize_model_config(json.loads(args.model_config.read_text(encoding="utf-8")))
    model = ModelConfig.model_validate(model_data)
    if not args.dry_run and model.provider_type == "cloud_openai" and not (model.api_key or (model.api_key_env and os.environ.get(model.api_key_env))):
        raise SystemExit("Missing model API key. Use a local model config with api_key or export the configured api_key_env.")

    cases = discover_cases(data_dir)
    if args.case_filter:
        pattern = re.compile(args.case_filter)
        cases = [case for case in cases if pattern.search(str(case))]
    if args.limit:
        cases = cases[: args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    reserved = {"hls_eval_zero_shot", "hls_eval_agentic"}
    requested_reserved = sorted(reserved.intersection(methods))
    if requested_reserved:
        raise SystemExit(
            "These HLS-Eval baselines are reserved for the original upstream runners, not this CCD-HLS runner: "
            f"{', '.join(requested_reserved)}. Use external/hls-eval/hls_eval_experiments/hls_gen_zero_shot__main/exp.py "
            "or external/hls-eval/hls_eval_experiments/hls_gen_agent_miniswe/exp.py."
        )
    pass_ks = [int(k.strip()) for k in args.pass_k.split(",") if k.strip()]
    experiment_id = new_id("exp")
    out_dir = (args.out_dir or Path("experiments") / experiment_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "hls_eval_root": str(hls_eval_root),
        "data_dir": str(data_dir),
        "case_count": len(cases),
        "methods": methods,
        "samples": args.samples,
        "hls_backend": args.hls_backend,
        "hls_part": args.hls_part or os.environ.get("HLS_PART"),
        "hls_platform": args.hls_platform or os.environ.get("HLS_PLATFORM"),
        "prompt_budget": args.prompt_budget,
        "max_llm_calls": args.max_llm_calls,
        "llm_call_budget": args.llm_call_budget if args.llm_call_budget is not None else args.max_llm_calls,
        "csim_budget": args.csim_budget,
        "synth_budget": args.synth_budget,
        "cosim_budget": args.cosim_budget,
        "unified_credit_budget": args.unified_credit_budget,
        "skill_token_budget": args.skill_token_budget,
        "repair_log_token_budget": args.repair_log_token_budget,
        "early_stop_similarity_threshold": args.early_stop_similarity_threshold,
        "deterministic_repair_enabled": not args.disable_deterministic_repair,
        "local_memory_enabled": not args.disable_local_memory,
        "memory_path": str(args.memory_path) if args.memory_path else None,
        "candidate_count": args.candidate_count,
        "candidate_policy": args.candidate_policy,
        "model": public_model_dump(model),
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("Cases:")
        for case in cases:
            print(case)
        return

    results: list[CaseResult] = []
    for case_idx, case in enumerate(cases, start=1):
        for sample_idx in range(args.samples):
            for method in methods:
                workdir = out_dir / method / case.name / f"sample_{sample_idx}"
                workdir.mkdir(parents=True, exist_ok=True)
                result_path = workdir / "result.json"
                if args.resume and result_path.exists():
                    try:
                        result = CaseResult(**json.loads(read_text(result_path)))
                        results.append(result)
                        print(f"[{case_idx}/{len(cases)}] skip existing {method} {case.name} sample={sample_idx}")
                        summarize(results, out_dir, pass_ks)
                        continue
                    except Exception as exc:
                        print(f"[{case_idx}/{len(cases)}] rerun unreadable result {method} {case.name} sample={sample_idx}: {exc}")
                print(f"[{case_idx}/{len(cases)}] {method} {case.name} sample={sample_idx}")
                if method in {"ccd_hls_gen_v2", "ccd_hls_loop"}:
                    result = await run_ccd_gen_v2_case(
                        experiment_id,
                        case,
                        workdir,
                        model,
                        args.hls_backend,
                        hls_eval_root,
                        sample_idx,
                        args.prompt_budget,
                        args.max_llm_calls,
                        args.repair_log_token_budget,
                        args.early_stop_similarity_threshold,
                        method,
                        args.llm_call_budget,
                        args.csim_budget,
                        args.synth_budget,
                        args.cosim_budget,
                        args.unified_credit_budget,
                        args.skill_token_budget,
                        not args.disable_deterministic_repair,
                        not args.disable_local_memory,
                        args.memory_path,
                        args.candidate_count,
                        args.candidate_policy,
                        args.hls_part,
                        args.hls_platform,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                results.append(result)
                write_text(result_path, json.dumps(result_to_json_dict(result), ensure_ascii=False, indent=2))
                summarize(results, out_dir, pass_ks)
    summarize(results, out_dir, pass_ks)
    print(f"Summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
