#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any

from ccd_hls_agent.ccd import (
    atomize_static_scan,
    choose_frontier,
    scan_benchmark,
    score_atoms,
    select_context,
)
from ccd_hls_agent.contracts import (
    build_ccd_hls_gen_v2_prompt as render_ccd_hls_gen_v2_prompt,
    build_hls_eval_zero_shot_prompt as render_hls_eval_zero_shot_prompt,
    build_hls_repair_prompt as render_hls_repair_prompt,
    build_output_code_repair_prompt as render_output_code_repair_prompt,
)
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
from ccd_hls_agent.model_clients import build_model_client
from ccd_hls_agent.schemas import ModelConfig
from ccd_hls_agent.utils import estimate_tokens, json_dumps, new_id, read_text, utc_now, write_text


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


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def normalize_model_config(model_data: dict[str, Any]) -> dict[str, Any]:
    api_key_env = str(model_data.get("api_key_env") or "")
    if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        fallback_env = "DEEPSEEK_API_KEY" if "deepseek" in str(model_data.get("base_url", "")).lower() else "MODEL_API_KEY"
        os.environ.setdefault(fallback_env, api_key_env)
        model_data = {**model_data, "api_key_env": fallback_env}
    return model_data


def public_model_dump(model: ModelConfig) -> dict[str, Any]:
    data = model.model_dump()
    if data.get("api_key_env") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(data["api_key_env"])):
        data["api_key_env"] = "<redacted-invalid-env-field>"
    data["api_key"] = "***" if model.api_key_env else None
    return data


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


def build_hls_eval_zero_shot_prompt(description: Path, tb: Path, header: Path) -> str:
    return render_hls_eval_zero_shot_prompt(description, tb, header, find_kernel_cpp(description.parent))


def build_ccd_hls_gen_v2_prompt(
    description: Path,
    tb: Path,
    header: Path,
    kernel: Path,
    selected_atoms: list[Any],
    *,
    token_budget: int,
    baseline_prompt_tokens: int,
) -> tuple[str, list[Any], int]:
    return render_ccd_hls_gen_v2_prompt(
        description,
        tb,
        header,
        kernel,
        selected_atoms,
        token_budget=token_budget,
        baseline_prompt_tokens=baseline_prompt_tokens,
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


async def evaluate_design(
    backend_kind: str,
    hls_eval_root: Path | None,
    case_dir: Path,
    build_dir: Path,
    top_function: str | None,
) -> tuple[dict[str, bool], dict[str, Any], int]:
    backend = build_hls_backend(
        backend_kind,
        HLSBackendConfig(hls_eval_root=str(hls_eval_root) if hls_eval_root else None),
    )
    config = {"top_function": top_function}
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
    case_dir: Path,
    build_dir: Path,
    top_function: str | None,
) -> tuple[dict[str, bool], dict[str, Any], ToolResult]:
    backend = build_hls_backend(
        backend_kind,
        HLSBackendConfig(hls_eval_root=str(hls_eval_root) if hls_eval_root else None),
    )
    config = {"top_function": top_function}
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
    case_dir: Path,
    build_dir: Path,
    top_function: str | None,
) -> tuple[bool, dict[str, Any], ToolResult]:
    backend = build_hls_backend(
        backend_kind,
        HLSBackendConfig(hls_eval_root=str(hls_eval_root) if hls_eval_root else None),
    )
    config = {"top_function": top_function}
    synth = await asyncio.to_thread(backend.run_synth, build_dir, source_files(case_dir, include_tb=False), config)
    can_synthesize = synth.return_code == 0
    metrics = {
        "synth": synth.metrics,
        "synth_return_code": synth.return_code,
        "synth_stdout_tail": synth.stdout[-2000:],
        "synth_stderr_tail": synth.stderr[-2000:],
    }
    return can_synthesize, metrics, synth


async def run_zero_shot_case(
    experiment_id: str,
    case_path: Path,
    workdir: Path,
    model: ModelConfig,
    backend_kind: str,
    hls_eval_root: Path | None,
    sample_idx: int,
) -> CaseResult:
    t0 = time.monotonic()
    files = prepare_generation_case(case_path, workdir / "design")
    prompt = build_hls_eval_zero_shot_prompt(files["description"], files["tb"], files["header"])
    write_text(workdir / "prompt.txt", prompt)
    client = build_model_client(model)
    can_parse = False
    error = None
    completion = ""
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = 0
    llm_duration_ms = 0
    tool_calls = 0
    flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
    metrics: dict[str, Any] = {}
    try:
        result = await client.complete(prompt)
        completion = result.content
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        llm_duration_ms = result.duration_ms
        write_text(workdir / "response.txt", completion)
        generated = extract_output_code(completion)
        can_parse = True
        for filename, code in generated.items():
            target = workdir / "design" / filename
            write_text(target, code)
        flags, metrics, tool_calls = await evaluate_design(backend_kind, hls_eval_root, workdir / "design", workdir / "build", read_text(workdir / "design" / "top.txt").strip())
    except Exception as exc:
        error = str(exc)
    return CaseResult(
        experiment_id=experiment_id,
        method="hls_eval_zero_shot",
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
) -> CaseResult:
    t0 = time.monotonic()
    files = prepare_generation_case(case_path, workdir / "design")
    zero_prompt = build_hls_eval_zero_shot_prompt(files["description"], files["tb"], files["header"])
    scan = scan_benchmark(workdir / "design")
    frontier = choose_frontier(scan, scan.blocker)
    atoms = atomize_static_scan("experiment", "run", workdir / "design", scan)
    scored = score_atoms(atoms, frontier, latest_blocker=scan.blocker)
    candidate_atoms, dropped = select_context(scored, 800, max_atoms=6)
    prompt, selected, max_prompt_tokens = build_ccd_hls_gen_v2_prompt(
        files["description"],
        files["tb"],
        files["header"],
        files["kernel"],
        candidate_atoms,
        token_budget=prompt_budget,
        baseline_prompt_tokens=estimate_tokens(zero_prompt),
    )
    dropped = dropped + [atom for atom in candidate_atoms if atom not in selected]
    write_text(workdir / "prompt.txt", prompt)
    write_text(workdir / "selected_atoms.json", json_dumps([a.model_dump() for a in selected]))
    write_text(workdir / "dropped_atoms.json", json_dumps([a.model_dump() for a in dropped]))

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
    metrics: dict[str, Any] = {
        "output_contract": "output_code_xml",
        "parse_mode": None,
        "fallback_reason": None,
        "stage_status": "INIT",
        "terminal_stage": None,
        "stopped_reason": None,
        "llm_calls_used": 0,
        "max_llm_calls": max_llm_calls,
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
        "max_prompt_tokens": max_prompt_tokens,
        "zero_shot_prompt_tokens_est": estimate_tokens(zero_prompt),
        "frontier": frontier,
    }

    def set_terminal(stage: str, reason: str) -> None:
        metrics["terminal_stage"] = stage
        metrics["stopped_reason"] = reason
        metrics["stage_status"] = "FAILED" if stage != "DONE" else "DONE"
        metrics["budget_exhausted"] = llm_calls_used >= max_llm_calls and stage != "DONE"

    async def call_llm(stage: str, prompt_text: str, system_prompt: str, *, label: str):
        nonlocal prompt_tokens, completion_tokens, llm_duration_ms, llm_calls_used
        if llm_calls_used >= max_llm_calls:
            return None
        llm_calls_used += 1
        call_prefix = f"llm_call_{llm_calls_used:02d}_{label}"
        prompt_path = workdir / f"{call_prefix}_prompt.txt"
        response_path = workdir / f"{call_prefix}_response.txt"
        write_text(prompt_path, prompt_text)
        result = await client.complete(prompt_text, system_prompt=system_prompt)
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
            },
            artifacts={"prompt": str(prompt_path), "response": str(response_path)},
        )
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
            csim_build = workdir / "build" / f"csim_attempt_{validation_attempt}"
            csim_flags, csim_metrics, csim_result = await run_csim_stage(
                backend_kind,
                hls_eval_root,
                workdir / "design",
                csim_build,
                top_function,
            )
            tool_calls += 1
            flags.update(csim_flags)
            metrics.update(csim_metrics)
            hls_artifact = write_json_artifact(
                workdir / f"hls_result_csim_attempt_{validation_attempt}.json",
                {"flags": csim_flags, "metrics": csim_metrics, "return_code": csim_result.return_code, "command": csim_result.command},
            )
            if csim_flags["can_compile"] and csim_flags["can_pass_testbench"]:
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
                add_stage_record(
                    stage_records,
                    stage="CSIM",
                    status="failed",
                    message=f"CSIM failed at attempt {validation_attempt}.",
                    metrics={"attempt": validation_attempt, **csim_flags},
                    artifacts={"hls_result": hls_artifact, "failure_capsule": capsule_path},
                )
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
                    set_terminal("PARSE_VALIDATE", metrics.get("fallback_reason") or "repair_output_parse_failed")
                    break
                continue

            metrics["stage_status"] = "SYNTH"
            synth_build = workdir / "build" / f"synth_attempt_{validation_attempt}"
            can_synth, synth_metrics, synth_result = await run_synth_stage(
                backend_kind,
                hls_eval_root,
                workdir / "design",
                synth_build,
                top_function,
            )
            tool_calls += 1
            flags["can_synthesize"] = can_synth
            metrics.update(synth_metrics)
            synth_artifact = write_json_artifact(
                workdir / f"hls_result_synth_attempt_{validation_attempt}.json",
                {"can_synthesize": can_synth, "metrics": synth_metrics, "return_code": synth_result.return_code, "command": synth_result.command},
            )
            if can_synth:
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
            add_stage_record(
                stage_records,
                stage="SYNTH",
                status="failed",
                message=f"Synthesis failed at attempt {validation_attempt}.",
                metrics={"attempt": validation_attempt, "can_synthesize": False},
                artifacts={"hls_result": synth_artifact, "failure_capsule": capsule_path},
            )
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
                set_terminal("PARSE_VALIDATE", metrics.get("fallback_reason") or "repair_output_parse_failed")
                break
            # Any code change after synth repair must be revalidated with CSIM first.
            flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
            continue

        if metrics["terminal_stage"] != "DONE":
            error = metrics.get("stopped_reason") or "repair_loop_failed"
    except Exception as exc:
        error = error or str(exc)
        metrics["source_has_diff_marker"] = source_has_diff_marker(workdir / "design")
    metrics["llm_calls_used"] = llm_calls_used
    metrics["stage_records"] = stage_records
    metrics["failure_capsules"] = failure_capsules
    metrics["source_has_diff_marker"] = source_has_diff_marker(workdir / "design")
    write_json_artifact(workdir / "stage_records.json", stage_records)
    write_json_artifact(workdir / "failure_capsules.json", failure_capsules)
    return CaseResult(
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


def summarize(results: list[CaseResult], out_dir: Path, pass_ks: list[int]) -> None:
    rows = [result_to_json_dict(r) for r in results]
    write_text(out_dir / "results.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
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
        for k in pass_ks:
            lines.append(f"- Can Synthesize pass@{k}: {row.get(f'can_synthesize_pass@{k}', 0):.2%}")
        lines.append("")
    write_text(out_dir / "summary.md", "\n".join(lines))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCD-HLS vs HLS-Eval zero-shot benchmark.")
    parser.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", default="ccd_hls_gen_v2,hls_eval_zero_shot")
    parser.add_argument("--hls-backend", choices=["hls_eval", "vitis", "command", "mock"], default="vitis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-filter", default=None, help="Regex matched against case path/name.")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--prompt-budget", type=int, default=6000)
    parser.add_argument("--max-llm-calls", type=int, default=5)
    parser.add_argument("--repair-log-token-budget", type=int, default=1200)
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

    load_env_file(args.env_file)
    hls_eval_root = args.hls_eval_root.expanduser().resolve()
    data_dir = (args.data_dir or hls_eval_root / "hls_eval_data").expanduser().resolve()
    model_data = normalize_model_config(json.loads(args.model_config.read_text(encoding="utf-8")))
    model = ModelConfig.model_validate(model_data)
    if not args.dry_run and model.provider_type == "cloud_openai" and model.api_key_env and not os.environ.get(model.api_key_env):
        raise SystemExit(f"Missing API key env {model.api_key_env}. Fill .env or export it first.")

    cases = discover_cases(data_dir)
    if args.case_filter:
        pattern = re.compile(args.case_filter)
        cases = [case for case in cases if pattern.search(str(case))]
    if args.limit:
        cases = cases[: args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
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
        "prompt_budget": args.prompt_budget,
        "max_llm_calls": args.max_llm_calls,
        "repair_log_token_budget": args.repair_log_token_budget,
        "early_stop_similarity_threshold": args.early_stop_similarity_threshold,
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
                    )
                elif method == "hls_eval_zero_shot":
                    result = await run_zero_shot_case(
                        experiment_id,
                        case,
                        workdir,
                        model,
                        args.hls_backend,
                        hls_eval_root,
                        sample_idx,
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
