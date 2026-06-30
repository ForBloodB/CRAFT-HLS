from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .task_contracts import (
    CONTRACT_META,
    copy_contract_case,
    ensure_contract_locked,
    lock_contract,
    prepare_hls_eval_contract,
    review_text,
    summarize_resolution,
)

TITLE = "CRAFT-HLS Agent TUI"
CALL_RE = re.compile(r"llm_call_(\d+)_(.+)_(prompt|response)\.txt$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LLMCall:
    index: int
    label: str
    prompt_path: Path | None = None
    response_path: Path | None = None


@dataclass
class Artifact:
    label: str
    path: Path


@dataclass(frozen=True)
class DashboardData:
    run_dir: Path
    stages: list[dict[str, Any]]
    calls: list[LLMCall]
    result: dict[str, Any]
    failure_capsules: list[dict[str, Any]]


@dataclass(frozen=True)
class RecentRun:
    run_dir: Path
    case_name: str
    method: str
    status: str
    can_compile: bool | None
    can_pass_testbench: bool | None
    can_synthesize: bool | None
    updated_at: float


@dataclass(frozen=True)
class LaunchData:
    root: Path
    runs: list[RecentRun]
    help_text: str


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"
    if limit and len(data) > limit:
        return data[:limit] + f"\n\n... <truncated {len(data) - limit} chars>"
    return data


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(read_text(path))
    except Exception:
        return default


def short_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def find_run_dir(path: Path) -> Path:
    path = path.expanduser()
    if (path / "result.json").exists() or (path / "stage_records.json").exists():
        return path
    candidates = sorted(
        list(path.rglob("result.json")) + list(path.rglob("stage_records.json")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent
    raise FileNotFoundError(f"No result.json or stage_records.json found under {path}")


def collect_llm_calls(run_dir: Path) -> list[LLMCall]:
    calls: dict[tuple[int, str], LLMCall] = {}
    for path in sorted(run_dir.glob("llm_call_*_*txt")):
        match = CALL_RE.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        label = match.group(2)
        kind = match.group(3)
        call = calls.setdefault((index, label), LLMCall(index=index, label=label))
        if kind == "prompt":
            call.prompt_path = path
        else:
            call.response_path = path
    return [calls[key] for key in sorted(calls)]


def load_stage_records(run_dir: Path) -> list[dict[str, Any]]:
    records = load_json(run_dir / "stage_records.json", None)
    if records is None:
        result = load_json(run_dir / "result.json", {})
        records = result.get("metrics", {}).get("stage_records", [])
    return records or []


def summary_stage(run_dir: Path) -> dict[str, Any]:
    result = load_json(run_dir / "result.json", {})
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    workflow = load_json(run_dir / "workflow_status.json", {})
    artifacts = {
        "result": str(run_dir / "result.json"),
        "stage_records": str(run_dir / "stage_records.json"),
        "failure_capsules": str(run_dir / "failure_capsules.json"),
        "token_report": str(run_dir / "token_report.json"),
        "budget_ledger": str(run_dir / "budget_ledger.json"),
        "selected_skills": str(run_dir / "selected_skills.json"),
        "workflow_status": str(run_dir / "workflow_status.json"),
        "workflow_events": str(run_dir / "workflow_events.jsonl"),
    }
    for name in ["selected_actions", "candidate_manifest", "candidate_results", "selected_candidate"]:
        path = run_dir / f"{name}.json"
        if path.exists():
            artifacts[name] = str(path)
    return {
        "stage": "SUMMARY",
        "status": "completed" if result else "running",
        "message": f"{result.get('method', 'run')} / {result.get('case_name', run_dir.name)}",
        "metrics": {
            "can_parse": result.get("can_parse"),
            "can_compile": result.get("can_compile"),
            "can_pass_testbench": result.get("can_pass_testbench"),
            "can_synthesize": result.get("can_synthesize"),
            "task_mode": metrics.get("task_mode"),
            "llm_calls_used": metrics.get("llm_calls_used"),
            "budget_summary": metrics.get("budget_summary"),
            "selected_skills": metrics.get("selected_skills"),
            "workflow_status": workflow or metrics.get("workflow_status"),
            "repair_rounds": metrics.get("repair_rounds"),
            "terminal_stage": metrics.get("terminal_stage"),
            "stopped_reason": metrics.get("stopped_reason"),
            "total_tokens": result.get("total_tokens"),
            "tool_calls": result.get("tool_calls"),
            "selected_actions": metrics.get("selected_actions"),
            "candidate_count": metrics.get("candidate_count"),
            "candidate_evaluations": metrics.get("candidate_evaluations"),
            "selected_candidate_score": metrics.get("selected_candidate_score"),
        },
        "artifacts": artifacts,
    }


def load_view_model(run_dir: Path) -> dict[str, Any]:
    run_dir = find_run_dir(run_dir)
    stages = [summary_stage(run_dir)] + load_stage_records(run_dir)
    calls = collect_llm_calls(run_dir)
    if not stages:
        stages = [summary_stage(run_dir)]
    return {
        "run_dir": run_dir,
        "stages": stages,
        "calls": calls,
        "result": load_json(run_dir / "result.json", {}),
        "failure_capsules": load_json(run_dir / "failure_capsules.json", []),
        "workflow_status": load_json(run_dir / "workflow_status.json", {}),
    }


def load_dashboard(run_dir: Path) -> DashboardData:
    model = load_view_model(run_dir)
    return DashboardData(
        run_dir=model["run_dir"],
        stages=model.get("stages", []),
        calls=model.get("calls", []),
        result=model.get("result", {}) or {},
        failure_capsules=model.get("failure_capsules", []) or [],
    )


def stage_artifacts(stage: dict[str, Any], run_dir: Path, calls: list[LLMCall]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for label, value in (stage.get("artifacts") or {}).items():
        path = Path(str(value))
        if not path.is_absolute() and not path.exists():
            path = run_dir / path
        if not path.exists():
            relocated = run_dir / path.name
            if relocated.exists():
                path = relocated
        artifacts.append(Artifact(label=label, path=path))
    if stage.get("stage") == "SUMMARY":
        for call in calls:
            if call.prompt_path:
                artifacts.append(Artifact(label=f"call_{call.index:02d}_prompt:{call.label}", path=call.prompt_path))
            if call.response_path:
                artifacts.append(Artifact(label=f"call_{call.index:02d}_response:{call.label}", path=call.response_path))
    return artifacts


def artifact_preview(artifact: Artifact | None) -> str:
    if artifact is None:
        return "No artifact selected."
    if not artifact.path.exists():
        return f"{artifact.label}\n{artifact.path}\n\n<missing>"
    if artifact.path.suffix == ".json":
        data = load_json(artifact.path, None)
        if data is not None:
            return f"{artifact.label}\n{artifact.path}\n\n" + json.dumps(data, ensure_ascii=False, indent=2)
    return f"{artifact.label}\n{artifact.path}\n\n" + read_text(artifact.path, limit=30000)


def compact_json(data: Any, width: int = 1200) -> str:
    text = json.dumps(data or {}, ensure_ascii=False, indent=2)
    if len(text) > width:
        return text[:width] + "\n... <truncated>"
    return text


def print_snapshot(run_dir: Path) -> None:
    model = load_view_model(run_dir)
    run_dir = model["run_dir"]
    result = model["result"] or {}
    print(f"Run dir: {short_path(run_dir)}")
    print(f"Case: {result.get('case_name', run_dir.name)}")
    print(f"Method: {result.get('method', 'unknown')}")
    print(
        "Result: "
        f"parse={result.get('can_parse')} compile={result.get('can_compile')} "
        f"tb={result.get('can_pass_testbench')} synth={result.get('can_synthesize')}"
    )
    print("")
    print("LLM/API calls:")
    for call in model["calls"]:
        prompt = short_path(call.prompt_path) if call.prompt_path else "<missing>"
        response = short_path(call.response_path) if call.response_path else "<missing>"
        print(f"  {call.index}. {call.label}")
        print(f"     input : {prompt}")
        print(f"     output: {response}")
    if not model["calls"]:
        print("  <none>")
    print("")
    print("Stages:")
    for index, stage in enumerate(model["stages"], start=1):
        artifacts = stage_artifacts(stage, run_dir, model["calls"])
        print(f"  {index:02d}. [{stage.get('status')}] {stage.get('stage')} - {stage.get('message', '')}")
        if stage.get("metrics"):
            print(f"      metrics: {compact_json(stage.get('metrics'), 300).replace(chr(10), ' ')}")
        if artifacts:
            print("      artifacts:")
            for artifact in artifacts:
                print(f"        - {artifact.label}: {short_path(artifact.path)}")


def default_out_dir(case_path: Path, max_llm_calls: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / "tui_demos" / f"tui_demo_{case_path.name}_loop{max_llm_calls}_{stamp}"


def launch_help_text() -> str:
    return "\n".join(
        [
            "No previous runs found under experiments/.",
            "",
            "Common commands:",
            "  HLS-agent run --case-path external/hls-eval/hls_eval_data/machsuite/md_knn",
            "  HLS-agent view experiments/.../sample_0",
            "  HLS-agent view experiments/.../sample_0 --snapshot",
            "",
            "Setup:",
            '  python -m pip install -e ".[tui]"',
            "  Put cloud model keys in an ignored local model config or export the configured api_key_env.",
        ]
    )


def recent_runs(root: Path = Path("experiments"), limit: int = 20) -> list[RecentRun]:
    if not root.exists():
        return []
    run_dirs: dict[Path, float] = {}
    for marker in list(root.rglob("result.json")) + list(root.rglob("stage_records.json")):
        run_dirs[marker.parent] = max(run_dirs.get(marker.parent, 0.0), marker.stat().st_mtime)
    runs = []
    for run_dir, mtime in run_dirs.items():
        result = load_json(run_dir / "result.json", {})
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        terminal = metrics.get("terminal_stage") or ("DONE" if result.get("can_synthesize") else "unknown")
        runs.append(
            RecentRun(
                run_dir=run_dir,
                case_name=str(result.get("case_name") or run_dir.parent.name or run_dir.name),
                method=str(result.get("method") or run_dir.parent.parent.name or "unknown"),
                status=str(terminal),
                can_compile=result.get("can_compile"),
                can_pass_testbench=result.get("can_pass_testbench"),
                can_synthesize=result.get("can_synthesize"),
                updated_at=mtime,
            )
        )
    return sorted(runs, key=lambda item: item.updated_at, reverse=True)[:limit]


def load_launch(root: Path = Path("experiments"), limit: int = 20) -> LaunchData:
    return LaunchData(root=root, runs=recent_runs(root, limit), help_text=launch_help_text())


def status_mark(status: str | None) -> str:
    value = str(status or "?").lower()
    return {
        "completed": "OK",
        "failed": "FAIL",
        "running": "RUN",
        "started": "RUN",
        "skipped": "SKIP",
    }.get(value, value[:4].upper())


def stage_rows(stages: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    rows = []
    for index, stage in enumerate(stages):
        rows.append(
            (
                f"{index:02d}",
                status_mark(stage.get("status")),
                str(stage.get("stage", "?")),
                str(stage.get("message", ""))[:120],
            )
        )
    return rows


def artifact_rows(artifacts: list[Artifact]) -> list[tuple[str, str, str]]:
    rows = []
    for index, artifact in enumerate(artifacts, start=1):
        rows.append((str(index), artifact.label, short_path(artifact.path)))
    return rows


def bool_mark(value: bool | None) -> str:
    if value is True:
        return "Y"
    if value is False:
        return "N"
    return "-"


def run_rows(runs: list[RecentRun]) -> list[tuple[str, str, str, str]]:
    return [
        (
            str(index + 1),
            run.status,
            f"{run.case_name}  C:{bool_mark(run.can_compile)} TB:{bool_mark(run.can_pass_testbench)} S:{bool_mark(run.can_synthesize)}",
            short_path(run.run_dir),
        )
        for index, run in enumerate(runs)
    ]


def load_env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def check_command(name: str) -> DoctorCheck:
    found = shutil.which(name)
    if found:
        return DoctorCheck(name, "OK", found)
    return DoctorCheck(name, "FAIL", f"{name} not found in PATH")


def doctor_checks(
    *,
    model_config: Path = Path("configs/deepseek_v4_flash.json"),
    hls_eval_root: Path = Path("external/hls-eval"),
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []

    model_data = load_json(model_config, None)
    if not isinstance(model_data, dict):
        checks.append(DoctorCheck("model config", "FAIL", f"Cannot read {short_path(model_config)}"))
    else:
        checks.append(DoctorCheck("model config", "OK", f"{short_path(model_config)} model={model_data.get('model')}"))
        api_key = str(model_data.get("api_key") or "")
        api_key_env = str(model_data.get("api_key_env") or "")
        provider = str(model_data.get("provider_type") or "")
        if provider == "cloud_openai":
            if api_key:
                checks.append(DoctorCheck("model api key", "OK", "api_key is set in local config"))
            elif api_key_env and os.environ.get(api_key_env):
                checks.append(DoctorCheck("model api key", "OK", f"{api_key_env} is set"))
            else:
                detail = f"{api_key_env} is not set" if api_key_env else "cloud_openai profile has no api_key or api_key_env"
                checks.append(DoctorCheck("model api key", "FAIL", detail))
        else:
            checks.append(DoctorCheck("model api key", "OK", f"provider={provider or 'unknown'}"))

    for command in ["v++", "vitis-run"]:
        checks.append(check_command(command))

    root = hls_eval_root.expanduser()
    if (root / "hls_eval").is_dir():
        checks.append(DoctorCheck("HLS-Eval root", "OK", short_path(root)))
    elif root.exists():
        checks.append(DoctorCheck("HLS-Eval root", "WARN", f"{short_path(root)} exists but no hls_eval package found"))
    else:
        checks.append(DoctorCheck("HLS-Eval root", "FAIL", f"{short_path(root)} not found"))

    return checks


def doctor_text(checks: list[DoctorCheck]) -> str:
    lines = ["HLS-agent doctor", ""]
    for check in checks:
        lines.append(f"[{check.status:<4}] {check.name}: {check.detail}")
    return "\n".join(lines)


def print_doctor(checks: list[DoctorCheck]) -> None:
    print(doctor_text(checks))


def result_summary(data: DashboardData) -> str:
    result = data.result or {}
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    workflow = load_json(data.run_dir / "workflow_status.json", {}) or metrics.get("workflow_status") or {}
    lines = [
        f"Run: {short_path(data.run_dir)}",
        f"Method: {result.get('method', 'unknown')}",
        f"Case: {result.get('case_name', data.run_dir.name)}",
        "",
        "Outcome",
        f"  parse     : {result.get('can_parse')}",
        f"  compile   : {result.get('can_compile')}",
        f"  testbench : {result.get('can_pass_testbench')}",
        f"  synth     : {result.get('can_synthesize')}",
        "",
        "Agent Budget",
        f"  task mode : {metrics.get('task_mode')}",
        f"  llm calls : {metrics.get('llm_calls_used')} / {metrics.get('max_llm_calls')}",
        f"  repairs   : {metrics.get('repair_rounds')}",
        f"  terminal  : {metrics.get('terminal_stage')}",
        f"  stopped   : {metrics.get('stopped_reason')}",
        f"  skills    : {', '.join(metrics.get('selected_skills', []) or []) or '-'}",
        "",
        "Workflow",
        f"  current   : {workflow.get('current_stage_label') or workflow.get('current_stage') or '-'}",
        f"  status    : {workflow.get('status') or '-'}",
        f"  attempt   : {workflow.get('attempt_index') if workflow else '-'}",
        "",
        "Tokens",
        f"  prompt    : {result.get('prompt_tokens')}",
        f"  completion: {result.get('completion_tokens')}",
        f"  total     : {result.get('total_tokens')}",
        "",
        "Budget Ledger",
        compact_json(metrics.get("budget_summary"), 800),
    ]
    return "\n".join(lines)


def truncate_for_tui(text: str, limit: int = 120_000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n... <truncated {omitted} chars> ...\n\n{text[-half:]}"


def run_benchmark(args: argparse.Namespace) -> Path:
    case_path = args.case_path.expanduser()
    out_dir = args.out_dir or default_out_dir(case_path, args.max_llm_calls)
    runner = PROJECT_ROOT / "scripts" / "run_hls_eval_benchmark.py"
    runner_arg = str(runner if runner.exists() else Path("scripts/run_hls_eval_benchmark.py"))
    command = [
        sys.executable,
        runner_arg,
        "--hls-eval-root",
        str(args.hls_eval_root),
        "--data-dir",
        str(args.data_dir),
        "--model-config",
        str(args.model_config),
        "--methods",
        args.method,
        "--samples",
        "1",
        "--hls-backend",
        args.hls_backend,
        "--max-llm-calls",
        str(args.max_llm_calls),
        "--llm-call-budget",
        str(args.llm_call_budget if args.llm_call_budget is not None else args.max_llm_calls),
        "--cosim-budget",
        str(args.cosim_budget),
        "--skill-token-budget",
        str(args.skill_token_budget),
        "--repair-log-token-budget",
        str(args.repair_log_token_budget),
        "--early-stop-similarity-threshold",
        str(args.early_stop_similarity_threshold),
        "--case-filter",
        re.escape(case_path.name) + "$",
        "--out-dir",
        str(out_dir),
    ]
    if getattr(args, "hls_part", None):
        command.extend(["--hls-part", str(args.hls_part)])
    if getattr(args, "hls_platform", None):
        command.extend(["--hls-platform", str(args.hls_platform)])
    if args.csim_budget is not None:
        command.extend(["--csim-budget", str(args.csim_budget)])
    if args.synth_budget is not None:
        command.extend(["--synth-budget", str(args.synth_budget)])
    if args.unified_credit_budget is not None:
        command.extend(["--unified-credit-budget", str(args.unified_credit_budget)])
    if getattr(args, "disable_deterministic_repair", False):
        command.append("--disable-deterministic-repair")
    if getattr(args, "disable_local_memory", False):
        command.append("--disable-local-memory")
    if getattr(args, "memory_path", None) is not None:
        command.extend(["--memory-path", str(args.memory_path)])
    command.extend(["--candidate-count", str(getattr(args, "candidate_count", 1))])
    command.extend(["--candidate-policy", str(getattr(args, "candidate_policy", "repair_only"))])
    if getattr(args, "candidate_synth_timeout_sec", None) is not None:
        command.extend(["--candidate-synth-timeout-sec", str(args.candidate_synth_timeout_sec)])
    print("Running:", flush=True)
    print(" ".join(command), flush=True)
    print("", flush=True)
    proc = subprocess.run(command, text=True, env=os.environ.copy())
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    run_dir = out_dir / args.method / case_path.name / "sample_0"
    if not run_dir.exists():
        run_dir = find_run_dir(out_dir)
    return run_dir


def run_contract_backend(args: argparse.Namespace) -> Path:
    contract_dir = args.contract_dir.expanduser().resolve()
    meta = ensure_contract_locked(contract_dir)
    task_id = str(meta.get("task_id") or contract_dir.name)
    case_root = contract_dir / "_materialized_hls_eval"
    case_dir = copy_contract_case(contract_dir, case_root / task_id)
    out_dir = args.out_dir or Path("experiments") / f"contract_{task_id}"
    runner_args = argparse.Namespace(
        case_path=case_dir,
        hls_eval_root=args.hls_eval_root,
        data_dir=case_root,
        model_config=args.model_config,
        hls_backend=args.hls_backend,
        hls_part=args.hls_part,
        hls_platform=args.hls_platform,
        method=args.method,
        max_llm_calls=args.max_llm_calls,
        llm_call_budget=args.llm_call_budget,
        csim_budget=args.csim_budget,
        synth_budget=args.synth_budget,
        cosim_budget=args.cosim_budget,
        unified_credit_budget=args.unified_credit_budget,
        skill_token_budget=args.skill_token_budget,
        repair_log_token_budget=args.repair_log_token_budget,
        early_stop_similarity_threshold=args.early_stop_similarity_threshold,
        disable_deterministic_repair=args.disable_deterministic_repair,
        disable_local_memory=args.disable_local_memory,
        memory_path=args.memory_path,
        candidate_count=args.candidate_count,
        candidate_policy=args.candidate_policy,
        candidate_synth_timeout_sec=args.candidate_synth_timeout_sec,
        out_dir=out_dir,
    )
    run_dir = run_benchmark(runner_args)
    resolution = summarize_resolution(contract_dir, run_dir)
    print(json.dumps(resolution, ensure_ascii=False, indent=2))
    return run_dir


def handle_contract_command(args: argparse.Namespace) -> Path | None:
    if args.contract_command == "prepare":
        source = read_text(args.input) if args.input else args.request
        if not source:
            raise SystemExit("contract prepare requires --input or --request")
        out = args.out.expanduser().resolve()
        meta = prepare_hls_eval_contract(source, out, target_platform=args.target_platform, task_id=args.task_id)
        print(f"Contract prepared: {short_path(out)}")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return out
    if args.contract_command == "review":
        print(review_text(args.contract_dir.expanduser().resolve()))
        return None
    if args.contract_command == "lock":
        meta = lock_contract(args.contract_dir.expanduser().resolve())
        print("Contract locked.")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return None
    if args.contract_command == "run":
        run_dir = run_contract_backend(args)
        print(f"Run artifacts: {short_path(run_dir)}")
        if args.snapshot:
            print_snapshot(run_dir)
        return run_dir
    if args.contract_command == "status":
        contract_dir = args.contract_dir.expanduser().resolve()
        print(review_text(contract_dir))
        for name in ["workflow_status.json", "workflow_token_summary.json", "resolution_report.json", CONTRACT_META]:
            path = contract_dir / name
            if path.exists():
                print("")
                print(f"## {name}")
                print(read_text(path, 4000))
        return None
    raise SystemExit("Unknown contract command")


def default_run_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(["run"])
    args.snapshot = False
    args.no_view = False
    return args


def prompt_run_args() -> argparse.Namespace:
    args = default_run_args()
    print("Run New Case")
    print("Press Enter to keep the default value.")

    def ask(label: str, current: Any) -> str:
        value = input(f"{label} [{current}]: ").strip()
        return value or str(current)

    args.case_path = Path(ask("case path", args.case_path))
    args.hls_eval_root = Path(ask("hls-eval root", args.hls_eval_root))
    args.data_dir = Path(ask("data dir", args.data_dir))
    args.model_config = Path(ask("model config", args.model_config))
    backend = ask("hls backend (hls_eval/vitis/command/mock)", args.hls_backend)
    if backend not in {"hls_eval", "vitis", "command", "mock"}:
        print(f"Unknown backend {backend!r}; using {args.hls_backend!r}.")
    else:
        args.hls_backend = backend
    args.method = ask("method", args.method)
    try:
        args.max_llm_calls = int(ask("max LLM calls", args.max_llm_calls))
    except ValueError:
        print(f"Invalid max LLM calls; using {args.max_llm_calls}.")
    return args


def print_launch_snapshot(root: Path = Path("experiments"), limit: int = 20) -> None:
    launch = load_launch(root, limit)
    if not launch.runs:
        print(launch.help_text)
        return
    print(f"Recent runs under {short_path(root)}:")
    for index, run in enumerate(launch.runs, start=1):
        print(f"  {index:02d}. [{run.status}] {run.method} / {run.case_name}")
        print(
            "      "
            f"C/TB/S={bool_mark(run.can_compile)}/{bool_mark(run.can_pass_testbench)}/{bool_mark(run.can_synthesize)} "
            f"{short_path(run.run_dir)}"
        )


def _textual_imports() -> tuple[Any, ...]:
    try:
        from rich.text import Text
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.widgets import DataTable, Footer, Header, Static
    except ImportError as exc:
        raise SystemExit(
            "Textual is required for the agent TUI. Install it with:\n"
            '  python -m pip install -e ".[tui]"\n'
            "You can still use --snapshot without Textual."
        ) from exc
    return Text, App, ComposeResult, Horizontal, Vertical, VerticalScroll, DataTable, Footer, Header, Static


def open_textual_tui(run_dir: Path) -> None:
    Text, App, ComposeResult, Horizontal, Vertical, VerticalScroll, DataTable, Footer, Header, Static = _textual_imports()

    class HLSAgentTui(App[None]):
        CSS = """
        Screen { background: $surface; }
        #main { height: 1fr; }
        #stages_panel { width: 35%; min-width: 34; border: solid $primary; }
        #artifacts_panel { width: 30%; min-width: 32; border: solid $secondary; }
        #preview_panel { width: 1fr; border: solid $accent; }
        .panel_title { text-style: bold; color: $text; padding: 0 1; height: 1; }
        #summary { height: 18; padding: 0 1; }
        #preview { padding: 0 1; }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "reload", "Reload"),
            ("j", "next_stage", "Next stage"),
            ("k", "previous_stage", "Previous stage"),
            ("right", "next_artifact", "Next artifact"),
            ("left", "previous_artifact", "Previous artifact"),
            ("l", "next_artifact", "Next artifact"),
            ("h", "previous_artifact", "Previous artifact"),
        ]

        def __init__(self, initial_run_dir: Path) -> None:
            super().__init__()
            self.data = load_dashboard(initial_run_dir)
            self.stage_index = 0
            self.artifact_index = 0

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="stages_panel"):
                    yield Static("Trajectory", classes="panel_title")
                    yield DataTable(id="stages")
                with Vertical(id="artifacts_panel"):
                    yield Static("Run Summary", classes="panel_title")
                    yield Static(id="summary")
                    yield Static("Artifacts", classes="panel_title")
                    yield DataTable(id="artifacts")
                with VerticalScroll(id="preview_panel"):
                    yield Static("Preview", classes="panel_title")
                    yield Static(id="preview")
            yield Footer()

        def on_mount(self) -> None:
            self.title = TITLE
            self.sub_title = short_path(self.data.run_dir)
            stages = self.query_one("#stages", DataTable)
            stages.cursor_type = "row"
            stages.add_columns("#", "Status", "Stage", "Message")
            artifacts = self.query_one("#artifacts", DataTable)
            artifacts.cursor_type = "row"
            artifacts.add_columns("#", "Artifact", "Path")
            self._render_all()

        def action_reload(self) -> None:
            self.data = load_dashboard(self.data.run_dir)
            self.stage_index = min(self.stage_index, max(0, len(self.data.stages) - 1))
            self.artifact_index = 0
            self.sub_title = short_path(self.data.run_dir)
            self._render_all()

        def action_next_stage(self) -> None:
            self.stage_index = min(max(0, len(self.data.stages) - 1), self.stage_index + 1)
            self.artifact_index = 0
            self._render_all()

        def action_previous_stage(self) -> None:
            self.stage_index = max(0, self.stage_index - 1)
            self.artifact_index = 0
            self._render_all()

        def action_next_artifact(self) -> None:
            artifacts = self._current_artifacts()
            self.artifact_index = min(max(0, len(artifacts) - 1), self.artifact_index + 1)
            self._render_all()

        def action_previous_artifact(self) -> None:
            self.artifact_index = max(0, self.artifact_index - 1)
            self._render_all()

        def _current_stage(self) -> dict[str, Any]:
            if not self.data.stages:
                return {}
            return self.data.stages[self.stage_index]

        def _current_artifacts(self) -> list[Artifact]:
            if not self.data.stages:
                return []
            return stage_artifacts(self._current_stage(), self.data.run_dir, self.data.calls)

        def _render_all(self) -> None:
            self._render_stages()
            self._render_summary()
            self._render_artifacts()
            self._render_preview()

        def _render_stages(self) -> None:
            table = self.query_one("#stages", DataTable)
            table.clear(columns=False)
            for row in stage_rows(self.data.stages):
                table.add_row(*row)
            self._move_table_cursor(table, self.stage_index)

        def _render_summary(self) -> None:
            self.query_one("#summary", Static).update(Text(result_summary(self.data)))

        def _render_artifacts(self) -> None:
            artifacts = self._current_artifacts()
            self.artifact_index = min(self.artifact_index, max(0, len(artifacts) - 1))
            table = self.query_one("#artifacts", DataTable)
            table.clear(columns=False)
            for row in artifact_rows(artifacts):
                table.add_row(*row)
            self._move_table_cursor(table, self.artifact_index)

        def _render_preview(self) -> None:
            artifacts = self._current_artifacts()
            artifact = artifacts[self.artifact_index] if artifacts else None
            stage = self._current_stage()
            if artifact is None:
                text = "\n".join(
                    [
                        f"{stage.get('stage', 'No stage')} / {stage.get('status', '')}",
                        "",
                        str(stage.get("message", "")),
                        "",
                        "Metrics:",
                        compact_json(stage.get("metrics"), 4000),
                    ]
                )
            else:
                text = artifact_preview(artifact)
            self.query_one("#preview", Static).update(Text(truncate_for_tui(text)))

        def _move_table_cursor(self, table: Any, row: int) -> None:
            if row < 0:
                return
            try:
                table.move_cursor(row=row, animate=False)
            except TypeError:
                try:
                    table.move_cursor(row=row)
                except Exception:
                    return
            except Exception:
                return

    HLSAgentTui(run_dir).run()


def open_launch_tui(root: Path = Path("experiments")) -> tuple[str, Path | None] | None:
    Text, App, ComposeResult, Horizontal, Vertical, VerticalScroll, DataTable, Footer, Header, Static = _textual_imports()

    menu_items = [
        ("Open Recent Run", "Browse recent experiment artifacts"),
        ("Run New Case", "Start with the default CCD-HLS LOOP settings"),
        ("Snapshot", "Print the latest run summary and exit"),
        ("Health Check", "Inspect model config, Vitis CLI, and HLS-Eval"),
        ("Help", "Show common commands and key bindings"),
    ]

    def default_run_lines() -> list[str]:
        args = default_run_args()
        return [
            "Default Run",
            "",
            f"case path       : {args.case_path}",
            f"hls backend     : {args.hls_backend}",
            f"method          : {args.method}",
            f"model config    : {args.model_config}",
            f"max LLM calls   : {args.max_llm_calls}",
            "",
            "Press Enter or n to confirm and edit these values in the terminal.",
            "For custom values, use:",
            "  HLS-agent run --case-path ... --model-config ... --max-llm-calls ...",
        ]

    class HLSAgentLaunch(App[tuple[str, Path | None] | None]):
        CSS = """
        Screen { background: $surface; }
        #main { height: 1fr; }
        #runs_panel { width: 62%; min-width: 58; border: solid $primary; }
        #help_panel { width: 1fr; border: solid $secondary; }
        .panel_title { text-style: bold; color: $text; padding: 0 1; height: 1; }
        #help { padding: 0 1; }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "reload", "Refresh"),
            ("enter", "open_selected", "Open"),
            ("o", "open_selected", "Open"),
            ("n", "new_run", "New run"),
            ("escape", "back", "Back"),
            ("j", "next_run", "Next"),
            ("k", "previous_run", "Previous"),
        ]

        def __init__(self, launch_root: Path) -> None:
            super().__init__()
            self.launch_root = launch_root
            self.data = load_launch(launch_root)
            self.mode = "menu"
            self.selected_index = 0
            self.doctor = doctor_checks()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="runs_panel"):
                    yield Static(id="table_title", classes="panel_title")
                    yield DataTable(id="runs")
                with VerticalScroll(id="help_panel"):
                    yield Static("Actions", classes="panel_title")
                    yield Static(id="help")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "HLS-agent"
            self.sub_title = short_path(self.launch_root)
            table = self.query_one("#runs", DataTable)
            table.cursor_type = "row"
            table.add_columns("#", "Status", "Item", "Detail")
            self._render_all()

        def action_reload(self) -> None:
            self.data = load_launch(self.launch_root)
            self.doctor = doctor_checks()
            self.selected_index = min(self.selected_index, max(0, len(self._rows()) - 1))
            self._render_all()

        def action_next_run(self) -> None:
            self.selected_index = min(max(0, len(self._rows()) - 1), self.selected_index + 1)
            self._render_all()

        def action_previous_run(self) -> None:
            self.selected_index = max(0, self.selected_index - 1)
            self._render_all()

        def action_back(self) -> None:
            if self.mode != "menu":
                self.mode = "menu"
                self.selected_index = 0
                self._render_all()

        def action_open_selected(self) -> None:
            if self.mode == "menu":
                self._activate_menu()
            elif self.mode == "recent" and self.data.runs:
                self.exit(("open", self.data.runs[self.selected_index].run_dir))
            elif self.mode == "run":
                self.exit(("run_prompt", None))
            elif self.mode == "snapshot":
                run_dir = self.data.runs[0].run_dir if self.data.runs else None
                self.exit(("snapshot", run_dir))

        def action_new_run(self) -> None:
            self.exit(("run_prompt", None))

        def _activate_menu(self) -> None:
            label = menu_items[self.selected_index][0]
            if label == "Open Recent Run":
                self.mode = "recent"
                self.selected_index = 0
            elif label == "Run New Case":
                self.mode = "run"
                self.selected_index = 0
            elif label == "Snapshot":
                self.mode = "snapshot"
                self.selected_index = 0
            elif label == "Health Check":
                self.mode = "doctor"
                self.selected_index = 0
            elif label == "Help":
                self.mode = "help"
                self.selected_index = 0
            self._render_all()

        def _rows(self) -> list[tuple[str, str, str, str]]:
            if self.mode == "menu":
                return [(str(index + 1), "MENU", label, detail) for index, (label, detail) in enumerate(menu_items)]
            if self.mode == "recent":
                return run_rows(self.data.runs)
            if self.mode == "run":
                args = default_run_args()
                return [
                    ("1", "CASE", "case path", str(args.case_path)),
                    ("2", "BACK", "hls backend", str(args.hls_backend)),
                    ("3", "METH", "method", str(args.method)),
                    ("4", "MODEL", "model config", str(args.model_config)),
                    ("5", "CALL", "max LLM calls", str(args.max_llm_calls)),
                ]
            if self.mode == "doctor":
                return [(str(index + 1), check.status, check.name, check.detail) for index, check in enumerate(self.doctor)]
            if self.mode == "snapshot":
                if not self.data.runs:
                    return [("1", "NONE", "No recent run", "Nothing to snapshot")]
                run = self.data.runs[0]
                return [("1", run.status, f"{run.method} / {run.case_name}", short_path(run.run_dir))]
            return [("1", "HELP", "HLS-agent", "Common commands and key bindings")]

        def _render_all(self) -> None:
            self.query_one("#table_title", Static).update(self._title_text())
            table = self.query_one("#runs", DataTable)
            table.clear(columns=False)
            rows = self._rows()
            self.selected_index = min(self.selected_index, max(0, len(rows) - 1))
            for row in rows:
                table.add_row(*row)
            self._move_table_cursor(table, self.selected_index)
            self.query_one("#help", Static).update(Text(self._help_text()))

        def _title_text(self) -> str:
            return {
                "menu": "Start",
                "recent": "Recent Runs",
                "run": "Run New Case",
                "doctor": "Health Check",
                "snapshot": "Snapshot",
                "help": "Help",
            }.get(self.mode, "Start")

        def _help_text(self) -> str:
            if self.mode == "menu":
                return "\n".join(
                    [
                        "Choose an action.",
                        "",
                        "Keys",
                        "  Enter/o : open selected action",
                        "  n       : start default run",
                        "  r       : refresh",
                        "  q       : quit",
                    ]
                )
            if self.mode == "recent":
                if not self.data.runs:
                    return self.data.help_text + "\n\nEsc returns to the start menu."
                selected = self.data.runs[self.selected_index]
                return "\n".join(
                    [
                        "Keys",
                        "  Enter/o : open selected run",
                        "  Esc     : back",
                        "  r       : refresh",
                        "  q       : quit",
                        "",
                        "Selected",
                        f"  case   : {selected.case_name}",
                        f"  method : {selected.method}",
                        f"  stage  : {selected.status}",
                        f"  C/TB/S : {bool_mark(selected.can_compile)}/{bool_mark(selected.can_pass_testbench)}/{bool_mark(selected.can_synthesize)}",
                        f"  path   : {short_path(selected.run_dir)}",
                    ]
                )
            if self.mode == "run":
                return "\n".join(default_run_lines())
            if self.mode == "doctor":
                return doctor_text(self.doctor) + "\n\nEsc returns to the start menu. r refreshes checks."
            if self.mode == "snapshot":
                if not self.data.runs:
                    return "No recent run is available for snapshot.\n\nEsc returns to the start menu."
                return "Press Enter to print the latest run snapshot and exit the TUI.\n\nEsc returns to the start menu."
            return "\n".join(
                [
                    "Common commands",
                    "  HLS-agent",
                    "  HLS-agent recent",
                    "  HLS-agent doctor",
                    "  HLS-agent view experiments/.../sample_0",
                    "  HLS-agent run --case-path external/hls-eval/hls_eval_data/machsuite/md_knn",
                    "",
                    "Install",
                    '  python -m pip install -e ".[tui]"',
                    "",
                    "Esc returns to the start menu.",
                ]
            )

        def _move_table_cursor(self, table: Any, row: int) -> None:
            if row < 0:
                return
            try:
                table.move_cursor(row=row, animate=False)
            except TypeError:
                try:
                    table.move_cursor(row=row)
                except Exception:
                    return
            except Exception:
                return

    return HLSAgentLaunch(root).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Textual agent TUI for CRAFT-HLS / CCD-HLS LOOP artifacts. "
            "Run HLS-agent without a subcommand to open the launcher."
        )
    )
    parser.add_argument("--snapshot", action="store_true", help="Print recent runs or the latest run summary without opening Textual.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"), help="Root scanned by the launcher.")
    sub = parser.add_subparsers(dest="command")

    view = sub.add_parser("view", help="Open the Textual TUI for an existing run.")
    view.add_argument("run_dir", type=Path)
    view.add_argument("--snapshot", action="store_true", help="Print a non-interactive summary.")

    run = sub.add_parser("run", help="Run one CCD-HLS case, then open the Textual TUI.")
    run.add_argument("--case-path", type=Path, default=Path("external/hls-eval/hls_eval_data/machsuite/spmv_crs"))
    run.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))
    run.add_argument("--data-dir", type=Path, default=Path("external/hls-eval/hls_eval_data"))
    run.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.json"))
    run.add_argument("--hls-backend", choices=["hls_eval", "vitis", "command", "mock"], default="vitis")
    run.add_argument("--hls-part", default=None)
    run.add_argument("--hls-platform", default=None)
    run.add_argument("--method", default="ccd_hls_loop")
    run.add_argument("--max-llm-calls", type=int, default=2)
    run.add_argument("--llm-call-budget", type=int, default=None)
    run.add_argument("--csim-budget", type=int, default=None)
    run.add_argument("--synth-budget", type=int, default=None)
    run.add_argument("--cosim-budget", type=int, default=0)
    run.add_argument("--unified-credit-budget", type=int, default=None)
    run.add_argument("--skill-token-budget", type=int, default=600)
    run.add_argument("--repair-log-token-budget", type=int, default=1200)
    run.add_argument("--early-stop-similarity-threshold", type=float, default=0.92)
    run.add_argument("--disable-deterministic-repair", action="store_true")
    run.add_argument("--disable-local-memory", action="store_true")
    run.add_argument("--memory-path", type=Path, default=None)
    run.add_argument("--candidate-count", type=int, default=1)
    run.add_argument("--candidate-policy", default="repair_only", choices=["repair_only"])
    run.add_argument("--candidate-synth-timeout-sec", type=float, default=180.0)
    run.add_argument("--out-dir", type=Path, default=None)
    run.add_argument("--no-view", action="store_true")
    run.add_argument("--snapshot", action="store_true", help="Print a summary after the run.")

    recent = sub.add_parser("recent", help="Print recent CCD-HLS runs.")
    recent.add_argument("--limit", type=int, default=20)
    recent.add_argument("--snapshot", action="store_true", help="Accepted for CLI symmetry; recent always prints text.")

    doctor = sub.add_parser("doctor", help="Check local HLS-agent configuration.")
    doctor.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.json"))
    doctor.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))

    contract = sub.add_parser("contract", help="Prepare, review, lock, and run HLS-Eval-like contracts.")
    contract_sub = contract.add_subparsers(dest="contract_command", required=True)

    prepare = contract_sub.add_parser("prepare", help="Create an HLS-Eval-like contract directory from a request.")
    prepare.add_argument("--input", type=Path, default=None)
    prepare.add_argument("--request", default="")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--task-id", default=None)
    prepare.add_argument("--target-platform", default="KV260")

    review = contract_sub.add_parser("review", help="Review contract completeness and token accounting.")
    review.add_argument("contract_dir", type=Path)

    lock = contract_sub.add_parser("lock", help="Lock a complete contract before execution.")
    lock.add_argument("contract_dir", type=Path)

    run_contract = contract_sub.add_parser("run", help="Run a locked contract through CCD-HLS LOOP.")
    run_contract.add_argument("contract_dir", type=Path)
    run_contract.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))
    run_contract.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.local.json"))
    run_contract.add_argument("--hls-backend", choices=["hls_eval", "vitis", "command", "mock"], default="vitis")
    run_contract.add_argument("--hls-part", default=None)
    run_contract.add_argument("--hls-platform", default=None)
    run_contract.add_argument("--method", default="ccd_hls_loop")
    run_contract.add_argument("--max-llm-calls", type=int, default=2)
    run_contract.add_argument("--llm-call-budget", type=int, default=None)
    run_contract.add_argument("--csim-budget", type=int, default=None)
    run_contract.add_argument("--synth-budget", type=int, default=None)
    run_contract.add_argument("--cosim-budget", type=int, default=0)
    run_contract.add_argument("--unified-credit-budget", type=int, default=None)
    run_contract.add_argument("--skill-token-budget", type=int, default=600)
    run_contract.add_argument("--repair-log-token-budget", type=int, default=1200)
    run_contract.add_argument("--early-stop-similarity-threshold", type=float, default=0.92)
    run_contract.add_argument("--disable-deterministic-repair", action="store_true")
    run_contract.add_argument("--disable-local-memory", action="store_true")
    run_contract.add_argument("--memory-path", type=Path, default=None)
    run_contract.add_argument("--candidate-count", type=int, default=1)
    run_contract.add_argument("--candidate-policy", default="repair_only", choices=["repair_only"])
    run_contract.add_argument("--candidate-synth-timeout-sec", type=float, default=180.0)
    run_contract.add_argument("--out-dir", type=Path, default=None)
    run_contract.add_argument("--snapshot", action="store_true")

    status = contract_sub.add_parser("status", help="Show contract workflow status.")
    status.add_argument("contract_dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "view":
        if args.snapshot or not sys.stdout.isatty():
            print_snapshot(args.run_dir)
            return
        open_textual_tui(args.run_dir)
        return

    if args.command == "run":
        run_dir = run_benchmark(args)
        print("")
        print(f"Run artifacts: {short_path(run_dir)}")
        if args.snapshot or args.no_view or not sys.stdout.isatty():
            print("")
            print_snapshot(run_dir)
            return
        open_textual_tui(run_dir)
        return

    if args.command == "recent":
        print_launch_snapshot(args.experiments_root, args.limit)
        return

    if args.command == "doctor":
        print_doctor(
            doctor_checks(
                model_config=args.model_config,
                hls_eval_root=args.hls_eval_root,
            )
        )
        return

    if args.command == "contract":
        handle_contract_command(args)
        return

    launch = load_launch(args.experiments_root)
    if args.snapshot or not sys.stdout.isatty():
        if launch.runs:
            print_snapshot(launch.runs[0].run_dir)
        else:
            print_launch_snapshot(args.experiments_root)
        return

    action = open_launch_tui(args.experiments_root)
    if action is None:
        return
    kind, run_dir = action
    if kind == "open" and run_dir is not None:
        open_textual_tui(run_dir)
    elif kind == "run_default":
        run_args = default_run_args()
        run_dir = run_benchmark(run_args)
        open_textual_tui(run_dir)
    elif kind == "run_prompt":
        run_args = prompt_run_args()
        run_dir = run_benchmark(run_args)
        open_textual_tui(run_dir)
    elif kind == "snapshot":
        if run_dir is not None:
            print_snapshot(run_dir)
        else:
            print_launch_snapshot(args.experiments_root)


if __name__ == "__main__":
    main()
