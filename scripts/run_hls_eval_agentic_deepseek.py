#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccd_hls_agent.model_clients import build_model_client
from ccd_hls_agent.schemas import ModelConfig
from ccd_hls_agent.utils import estimate_tokens, new_id, read_text, utc_now, write_text
from run_hls_eval_benchmark import (
    discover_cases,
    evaluate_design,
    find_header,
    find_kernel_cpp,
    find_tb,
    normalize_model_config,
    parse_tags,
    public_model_dump,
)


SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
AGENT_SYSTEM_PROMPT = """You are a helpful HLS coding agent that can interact with a shell.

Your response must contain exactly ONE bash code block with ONE command.
Include a short THOUGHT section before the command.
Do not use markdown code blocks for anything except the bash action.
The current working directory is already the design project directory; do not cd to /home/user.
For large source files, split file creation across multiple turns. Never emit an unfinished heredoc or an unclosed bash fence.
When inspecting files, prefer `sed -n '1,160p' file` or `head` over dumping every file at once.
"""


@dataclass
class AgenticResult:
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
    agent_steps: int
    wall_time_ms: int
    error: str | None
    metrics: dict[str, Any]
    artifacts_dir: str


def ensure_hls_eval_importable(hls_eval_root: Path) -> None:
    if str(hls_eval_root) not in sys.path:
        sys.path.insert(0, str(hls_eval_root))


def load_agentic_prompt_builder(hls_eval_root: Path):
    ensure_hls_eval_importable(hls_eval_root)
    from hls_eval.prompts import build_prompt_gen_agentic

    return build_prompt_gen_agentic


def copy_agent_inputs(case_path: Path, agent_run_dir: Path) -> dict[str, Path]:
    if agent_run_dir.exists():
        shutil.rmtree(agent_run_dir)
    agent_run_dir.mkdir(parents=True)
    header = find_header(case_path)
    tb = find_tb(case_path)
    kernel = find_kernel_cpp(case_path)
    description = case_path / "kernel_description.md"
    copied = {
        "header": agent_run_dir / header.name,
        "tb": agent_run_dir / tb.name,
        "kernel": agent_run_dir / kernel.name,
        "description": agent_run_dir / description.name,
    }
    for fp in [description, header, tb]:
        shutil.copy(fp, agent_run_dir / fp.name)
    for fp in case_path.iterdir():
        if not fp.is_file():
            continue
        if fp.name in {description.name, header.name, tb.name, kernel.name, "top.txt", "hls_eval_config.toml"}:
            continue
        if fp.suffix not in SOURCE_EXTENSIONS:
            shutil.copy(fp, agent_run_dir / fp.name)
    return copied


def truncate_observation(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n\n[... {len(text) - limit} characters elided ...]\n\n{tail}"


def extract_bash_action(response: str) -> tuple[str | None, str]:
    if not response.strip():
        return None, "Empty model response."
    blocks = re.findall(r"```(?:bash|sh)?\s*(.*?)```", response, re.S)
    if len(blocks) != 1:
        return None, f"Expected exactly one bash code block, found {len(blocks)}."
    command = blocks[0].strip()
    if not command:
        return None, "Bash code block was empty."
    return command, "ok"


def is_unsafe_command(command: str) -> bool:
    lowered = command.lower()
    blocked = [
        "sudo ",
        "rm -rf /",
        "mkfs",
        "shutdown",
        "reboot",
        "dd if=",
        "chmod -r 777 /",
        "chown -r ",
    ]
    return any(item in lowered for item in blocked)


def run_shell_action(command: str, cwd: Path, timeout_seconds: float, observation_limit: int) -> dict[str, Any]:
    if is_unsafe_command(command):
        return {
            "return_code": 126,
            "stdout": "",
            "stderr": "Command rejected by safety guard.",
            "duration_ms": 0,
            "timeout": False,
        }
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        return {
            "return_code": proc.returncode,
            "stdout": truncate_observation(proc.stdout, observation_limit),
            "stderr": truncate_observation(proc.stderr, observation_limit),
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        return {
            "return_code": -1,
            "stdout": truncate_observation(stdout, observation_limit),
            "stderr": truncate_observation(stderr + "\nTIMEOUT", observation_limit),
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "timeout": True,
        }


def build_turn_prompt(task_prompt: str, steps: list[dict[str, Any]], max_recent_steps: int) -> str:
    older = steps[:-max_recent_steps] if len(steps) > max_recent_steps else []
    recent = steps[-max_recent_steps:]
    older_summary = ""
    if older:
        older_summary = "Older step summary:\n" + "\n".join(
            f"- step {s['step']}: rc={s.get('return_code')} command={(s.get('command') or '')[:120]}"
            for s in older
        )
    recent_text = ""
    if recent:
        parts = []
        for step in recent:
            parts.append(
                "\n".join(
                    [
                        f"Step {step['step']}",
                        f"Command: {step.get('command')}",
                        f"Return code: {step.get('return_code')}",
                        f"Stdout:\n{step.get('stdout', '')}",
                        f"Stderr:\n{step.get('stderr', '')}",
                    ]
                )
            )
        recent_text = "Recent observations:\n" + "\n\n".join(parts)
    return f"""{task_prompt}

## Agent Workflow
Work step by step:
1. Inspect the available files when needed.
2. Create the missing kernel implementation file.
3. Do not modify the header or testbench.
4. When finished, submit with exactly:
{SUBMIT_COMMAND}

Each response must contain exactly one bash code block with one command.
The shell starts in the design project directory. Do not use /home/user.
For large implementations, keep each command compact. Use `cat > file <<'EOF'` for the first chunk and `cat >> file <<'EOF'` for later chunks rather than emitting an oversized unfinished command.
When reading files, use targeted `sed`/`head` commands instead of dumping all files at once.

{older_summary}

{recent_text}

Return the next action now.
"""


def write_agent_artifacts(workdir: Path, steps: list[dict[str, Any]]) -> None:
    write_text(workdir / "agent_steps.jsonl", "\n".join(json.dumps(step, ensure_ascii=False) for step in steps) + ("\n" if steps else ""))
    write_text(workdir / "trace.json", json.dumps(steps, ensure_ascii=False, indent=2))
    lines = []
    for step in steps:
        lines.extend(
            [
                f"## Step {step['step']}",
                f"Command: {step.get('command')}",
                f"Return code: {step.get('return_code')}",
                "Stdout:",
                step.get("stdout", ""),
                "Stderr:",
                step.get("stderr", ""),
                "",
            ]
        )
    write_text(workdir / "agent_output.txt", "\n".join(lines))


async def run_agentic_case(
    experiment_id: str,
    case_path: Path,
    workdir: Path,
    model: ModelConfig,
    hls_eval_root: Path,
    backend_kind: str,
    sample_idx: int,
    step_limit: int,
    step_timeout_seconds: float,
    max_recent_steps: int,
    max_format_errors: int,
    observation_limit: int,
) -> AgenticResult:
    t0 = time.monotonic()
    build_prompt_gen_agentic = load_agentic_prompt_builder(hls_eval_root)
    agent_run_dir = workdir / "agent_run_dir"
    files = copy_agent_inputs(case_path, agent_run_dir)
    original_header = read_text(case_path / files["header"].name)
    original_tb = read_text(case_path / files["tb"].name)
    task_prompt = build_prompt_gen_agentic(
        fn_design_description=files["description"].name,
        fn_design_h=files["header"].name,
        fn_design_tb=files["tb"].name,
        fn_design_kernel=files["kernel"].name,
    )
    write_text(workdir / "raw_agent_prompt.txt", task_prompt)

    client = build_model_client(model)
    steps: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    llm_duration_ms = 0
    shell_commands = 0
    agent_submitted = False
    agent_limit_exceeded = False
    consecutive_format_errors = 0
    error: str | None = None
    flags = {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}
    hls_tool_calls = 0
    metrics: dict[str, Any] = {
        "agent_submitted": False,
        "agent_limit_exceeded": False,
        "shell_commands": 0,
        "hls_tool_calls": 0,
    }

    for step_idx in range(1, step_limit + 1):
        turn_prompt = build_turn_prompt(task_prompt, steps, max_recent_steps)
        result = await client.complete(turn_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        llm_duration_ms += result.duration_ms
        response = result.content
        command, parse_message = extract_bash_action(response)
        step_record: dict[str, Any] = {
            "step": step_idx,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "llm_duration_ms": result.duration_ms,
            "response": response,
            "command": command,
        }
        if command is None:
            consecutive_format_errors += 1
            step_record.update(
                {
                    "return_code": 2,
                    "stdout": "",
                    "stderr": parse_message,
                    "timeout": False,
                }
            )
            steps.append(step_record)
            write_agent_artifacts(workdir, steps)
            if consecutive_format_errors >= max_format_errors:
                agent_limit_exceeded = True
                error = "AGENT_FORMAT_ERROR_LIMIT"
                break
            continue
        consecutive_format_errors = 0
        if command.strip() == SUBMIT_COMMAND:
            agent_submitted = True
            step_record.update(
                {
                    "return_code": 0,
                    "stdout": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                    "stderr": "",
                    "timeout": False,
                }
            )
            steps.append(step_record)
            write_agent_artifacts(workdir, steps)
            break
        shell_result = await asyncio.to_thread(run_shell_action, command, agent_run_dir, step_timeout_seconds, observation_limit)
        shell_commands += 1
        step_record.update(shell_result)
        steps.append(step_record)
        write_agent_artifacts(workdir, steps)
    else:
        agent_limit_exceeded = True
        error = "AGENT_LIMIT_EXCEEDED"

    can_find_kernel_file = files["kernel"].exists()
    has_modified_header = read_text(files["header"]) != original_header
    has_modified_testbench = read_text(files["tb"]) != original_tb
    can_parse = bool(agent_submitted and can_find_kernel_file and not has_modified_header and not has_modified_testbench)
    metrics.update(
        {
            "agent_submitted": agent_submitted,
            "agent_limit_exceeded": agent_limit_exceeded,
            "can_find_kernel_file": can_find_kernel_file,
            "has_modified_header": has_modified_header,
            "has_modified_testbench": has_modified_testbench,
            "shell_commands": shell_commands,
            "agent_steps": len(steps),
        }
    )

    if can_find_kernel_file:
        write_text(workdir / files["kernel"].name, read_text(files["kernel"]))

    if can_parse:
        design_generated = workdir / "design_generated"
        if design_generated.exists():
            shutil.rmtree(design_generated)
        design_generated.mkdir(parents=True)
        for fp in agent_run_dir.iterdir():
            if fp.is_file() and (fp.suffix not in SOURCE_EXTENSIONS or fp.name in {files["header"].name, files["tb"].name, files["description"].name, files["kernel"].name}):
                shutil.copy(fp, design_generated / fp.name)
        flags, hls_metrics, hls_tool_calls = await evaluate_design(
            backend_kind,
            hls_eval_root,
            design_generated,
            workdir / "build",
            read_text(case_path / "top.txt").strip(),
        )
        metrics.update(hls_metrics)
    elif error is None:
        error = "AGENT_OUTPUT_PARSE_FAILED"

    metrics["hls_tool_calls"] = hls_tool_calls
    metrics["tool_calls_total"] = shell_commands + hls_tool_calls
    write_agent_artifacts(workdir, steps)
    return AgenticResult(
        experiment_id=experiment_id,
        method="hls_eval_agentic_deepseek_direct",
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
        tool_calls=shell_commands + hls_tool_calls,
        agent_steps=len(steps),
        wall_time_ms=int((time.monotonic() - t0) * 1000),
        error=error,
        metrics=metrics,
        artifacts_dir=str(workdir),
    )


def summarize(results: list[AgenticResult], out_dir: Path) -> None:
    rows = [asdict(r) for r in results]
    write_text(out_dir / "results.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""))
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
        "agent_steps",
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

    n = len(results)
    summary = {
        "method": "hls_eval_agentic_deepseek_direct",
        "n_samples_total": n,
        "avg_prompt_tokens": sum(r.prompt_tokens for r in results) / max(1, n),
        "avg_completion_tokens": sum(r.completion_tokens for r in results) / max(1, n),
        "avg_total_tokens": sum(r.total_tokens for r in results) / max(1, n),
        "avg_agent_steps": sum(r.agent_steps for r in results) / max(1, n),
        "avg_tool_calls": sum(r.tool_calls for r in results) / max(1, n),
        "avg_wall_time_ms": sum(r.wall_time_ms for r in results) / max(1, n),
    }
    for metric in ["can_parse", "can_compile", "can_pass_testbench", "can_synthesize"]:
        summary[f"{metric}_rate"] = sum(1 for r in results if getattr(r, metric)) / max(1, n)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    lines = [
        "# HLS-Eval Agentic DeepSeek Direct Summary",
        "",
        f"Generated at: {utc_now()}",
        "",
        f"- total samples: {n}",
        f"- Can Parse rate: {summary['can_parse_rate']:.2%}",
        f"- Can Compile rate: {summary['can_compile_rate']:.2%}",
        f"- Can Pass Testbench rate: {summary['can_pass_testbench_rate']:.2%}",
        f"- Can Synthesize rate: {summary['can_synthesize_rate']:.2%}",
        f"- avg prompt tokens: {summary['avg_prompt_tokens']:.1f}",
        f"- avg completion tokens: {summary['avg_completion_tokens']:.1f}",
        f"- avg total tokens: {summary['avg_total_tokens']:.1f}",
        f"- avg agent steps: {summary['avg_agent_steps']:.1f}",
        f"- avg tool calls: {summary['avg_tool_calls']:.1f}",
        f"- avg wall time ms: {summary['avg_wall_time_ms']:.1f}",
    ]
    write_text(out_dir / "summary.md", "\n".join(lines))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a direct DeepSeek HLS-Eval agentic benchmark.")
    parser.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hls-backend", choices=["hls_eval", "vitis", "command", "mock"], default="vitis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--step-limit", type=int, default=40)
    parser.add_argument("--step-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-recent-steps", type=int, default=8)
    parser.add_argument("--max-format-errors", type=int, default=8)
    parser.add_argument("--observation-limit", type=int, default=4000)
    parser.add_argument("--model-max-tokens", type=int, default=8192)
    parser.add_argument("--model-timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hls_eval_root = args.hls_eval_root.expanduser().resolve()
    data_dir = (args.data_dir or hls_eval_root / "hls_eval_data").expanduser().resolve()
    model_data = normalize_model_config(json.loads(args.model_config.read_text(encoding="utf-8")))
    model = ModelConfig.model_validate(model_data)
    model.max_tokens = max(model.max_tokens, args.model_max_tokens)
    model.timeout = max(model.timeout, args.model_timeout)
    if model.provider_type == "cloud_openai" and not (model.api_key or (model.api_key_env and os.environ.get(model.api_key_env))):
        raise SystemExit("Missing model API key. Use a local model config with api_key or export the configured api_key_env.")

    cases = discover_cases(data_dir)
    if args.case_filter:
        pattern = re.compile(args.case_filter)
        cases = [case for case in cases if pattern.search(str(case))]
    if args.limit:
        cases = cases[: args.limit]

    experiment_id = new_id("exp")
    out_dir = (args.out_dir or Path("experiments") / experiment_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "hls_eval_root": str(hls_eval_root),
        "data_dir": str(data_dir),
        "case_count": len(cases),
        "samples": args.samples,
        "hls_backend": args.hls_backend,
        "step_limit": args.step_limit,
        "step_timeout_seconds": args.step_timeout_seconds,
        "max_format_errors": args.max_format_errors,
        "observation_limit": args.observation_limit,
        "model": public_model_dump(model),
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        for case in cases:
            print(case)
        return

    results: list[AgenticResult] = []
    completed: set[tuple[str, int]] = set()
    results_path = out_dir / "results.jsonl"
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result = AgenticResult(**row)
            results.append(result)
            completed.add((result.case_name, result.sample_idx))
        if results:
            summarize(results, out_dir)

    for case_idx, case in enumerate(cases, start=1):
        for sample_idx in range(args.samples):
            if (case.name, sample_idx) in completed:
                print(f"[{case_idx}/{len(cases)}] skip completed {case.name} sample={sample_idx}")
                continue
            workdir = out_dir / "hls_eval_agentic_deepseek_direct" / case.name / f"sample_{sample_idx}"
            workdir.mkdir(parents=True, exist_ok=True)
            print(f"[{case_idx}/{len(cases)}] hls_eval_agentic_deepseek_direct {case.name} sample={sample_idx}")
            try:
                result = await run_agentic_case(
                    experiment_id=experiment_id,
                    case_path=case,
                    workdir=workdir,
                    model=model,
                    hls_eval_root=hls_eval_root,
                    backend_kind=args.hls_backend,
                    sample_idx=sample_idx,
                    step_limit=args.step_limit,
                    step_timeout_seconds=args.step_timeout_seconds,
                    max_recent_steps=args.max_recent_steps,
                    max_format_errors=args.max_format_errors,
                    observation_limit=args.observation_limit,
                )
            except Exception as exc:
                result = AgenticResult(
                    experiment_id=experiment_id,
                    method="hls_eval_agentic_deepseek_direct",
                    sample_idx=sample_idx,
                    case_name=case.name,
                    case_path=str(case),
                    tags=parse_tags(case / "hls_eval_config.toml"),
                    can_parse=False,
                    can_compile=False,
                    can_pass_testbench=False,
                    can_synthesize=False,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    llm_duration_ms=0,
                    tool_calls=0,
                    agent_steps=0,
                    wall_time_ms=0,
                    error=f"RUNNER_EXCEPTION: {exc}",
                    metrics={"runner_exception": str(exc)},
                    artifacts_dir=str(workdir),
                )
            results.append(result)
            write_text(workdir / "result.json", json.dumps(asdict(result), ensure_ascii=False, indent=2))
            summarize(results, out_dir)
    summarize(results, out_dir)
    print(f"Summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
