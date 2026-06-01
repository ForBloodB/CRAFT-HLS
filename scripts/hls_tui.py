#!/usr/bin/env python3
from __future__ import annotations

import argparse
import curses
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CALL_RE = re.compile(r"llm_call_(\d+)_(.+)_(prompt|response)\.txt$")


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
    return {
        "stage": "SUMMARY",
        "status": "completed" if result else "running",
        "message": f"{result.get('method', 'run')} / {result.get('case_name', run_dir.name)}",
        "metrics": {
            "can_parse": result.get("can_parse"),
            "can_compile": result.get("can_compile"),
            "can_pass_testbench": result.get("can_pass_testbench"),
            "can_synthesize": result.get("can_synthesize"),
            "llm_calls_used": metrics.get("llm_calls_used"),
            "repair_rounds": metrics.get("repair_rounds"),
            "terminal_stage": metrics.get("terminal_stage"),
            "stopped_reason": metrics.get("stopped_reason"),
            "total_tokens": result.get("total_tokens"),
            "tool_calls": result.get("tool_calls"),
        },
        "artifacts": {
            "result": str(run_dir / "result.json"),
            "stage_records": str(run_dir / "stage_records.json"),
            "failure_capsules": str(run_dir / "failure_capsules.json"),
        },
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
    }


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


class TuiState:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.stage_index = 0
        self.artifact_index = 0
        self.scroll = 0
        self.model: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        self.model = load_view_model(self.run_dir)
        self.run_dir = self.model["run_dir"]
        self.stage_index = min(self.stage_index, max(0, len(self.stages) - 1))
        self.artifact_index = min(self.artifact_index, max(0, len(self.artifacts) - 1))

    @property
    def stages(self) -> list[dict[str, Any]]:
        return self.model.get("stages", [])

    @property
    def calls(self) -> list[LLMCall]:
        return self.model.get("calls", [])

    @property
    def current_stage(self) -> dict[str, Any]:
        return self.stages[self.stage_index]

    @property
    def artifacts(self) -> list[Artifact]:
        return stage_artifacts(self.current_stage, self.run_dir, self.calls)

    @property
    def current_artifact(self) -> Artifact | None:
        if not self.artifacts:
            return None
        return self.artifacts[self.artifact_index]


def addstr(win: Any, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        height, width = win.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        win.addnstr(y, x, text, max(0, width - x - 1), attr)
    except curses.error:
        return


def wrapped_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines() or [""]:
        if len(raw) <= width:
            lines.append(raw)
        else:
            lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False, drop_whitespace=False) or [""])
    return lines


def draw_tui(stdscr: Any, state: TuiState) -> None:
    curses.curs_set(0)
    stdscr.nodelay(False)
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 16 or width < 80:
            addstr(stdscr, 0, 0, "Terminal too small for CCD-HLS TUI. Resize or use --snapshot.")
            key = stdscr.getch()
            if key in (ord("q"), 27):
                return
            continue

        left_w = min(42, max(28, width // 3))
        right_x = left_w + 2
        right_w = width - right_x - 1
        addstr(stdscr, 0, 0, f"CCD-HLS LOOP TUI | {short_path(state.run_dir)}", curses.A_BOLD)
        addstr(stdscr, 1, 0, "Up/Down: stage  Left/Right: artifact  PgUp/PgDn: scroll  r: reload  q: quit")
        addstr(stdscr, 2, 0, "-" * (width - 1))

        addstr(stdscr, 3, 0, "Stages", curses.A_BOLD)
        list_height = height - 5
        first = max(0, state.stage_index - list_height + 2)
        for row, stage in enumerate(state.stages[first : first + list_height], start=4):
            absolute = first + row - 4
            status = str(stage.get("status", "?"))
            mark = {"completed": "OK", "failed": "FAIL", "running": "RUN", "started": "RUN"}.get(status, status[:4].upper())
            label = f"{absolute:02d} [{mark:<4}] {stage.get('stage', '?')}"
            attr = curses.A_REVERSE if absolute == state.stage_index else 0
            addstr(stdscr, row, 0, label[: left_w - 1], attr)

        stage = state.current_stage
        artifacts = state.artifacts
        artifact = state.current_artifact
        addstr(stdscr, 3, right_x, f"{stage.get('stage')} / {stage.get('status')}", curses.A_BOLD)
        detail_lines = [
            f"Message: {stage.get('message', '')}",
            "Metrics:",
            compact_json(stage.get("metrics"), 1400),
            "",
            "Artifacts:",
        ]
        if artifacts:
            for idx, item in enumerate(artifacts):
                prefix = ">" if idx == state.artifact_index else " "
                detail_lines.append(f"{prefix} {idx + 1}. {item.label}: {short_path(item.path)}")
        else:
            detail_lines.append("  <none>")
        detail_lines.append("")
        detail_lines.append("Selected artifact preview:")
        detail_lines.extend(artifact_preview(artifact).splitlines())

        view_lines = []
        for line in detail_lines:
            view_lines.extend(wrapped_lines(line, right_w - 2))
        max_scroll = max(0, len(view_lines) - (height - 5))
        state.scroll = min(state.scroll, max_scroll)
        for row, line in enumerate(view_lines[state.scroll : state.scroll + height - 5], start=4):
            addstr(stdscr, row, right_x, line)

        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return
        if key in (curses.KEY_UP, ord("k")):
            state.stage_index = max(0, state.stage_index - 1)
            state.artifact_index = 0
            state.scroll = 0
        elif key in (curses.KEY_DOWN, ord("j")):
            state.stage_index = min(len(state.stages) - 1, state.stage_index + 1)
            state.artifact_index = 0
            state.scroll = 0
        elif key == curses.KEY_LEFT:
            state.artifact_index = max(0, state.artifact_index - 1)
            state.scroll = 0
        elif key == curses.KEY_RIGHT:
            state.artifact_index = min(max(0, len(state.artifacts) - 1), state.artifact_index + 1)
            state.scroll = 0
        elif key in (curses.KEY_NPAGE, ord(" ")):
            state.scroll += max(4, height - 8)
        elif key == curses.KEY_PPAGE:
            state.scroll = max(0, state.scroll - max(4, height - 8))
        elif key == ord("r"):
            state.reload()
            state.scroll = 0


def open_tui(run_dir: Path) -> None:
    state = TuiState(run_dir)
    curses.wrapper(draw_tui, state)


def default_out_dir(case_path: Path, max_llm_calls: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / "tui_demos" / f"tui_demo_{case_path.name}_loop{max_llm_calls}_deepseek_{stamp}"


def run_benchmark(args: argparse.Namespace) -> Path:
    case_path = args.case_path.expanduser()
    out_dir = args.out_dir or default_out_dir(case_path, args.max_llm_calls)
    runner = Path("scripts/run_hls_eval_benchmark.py")
    command = [
        sys.executable,
        str(runner),
        "--hls-eval-root",
        str(args.hls_eval_root),
        "--data-dir",
        str(args.data_dir),
        "--model-config",
        str(args.model_config),
        "--env-file",
        str(args.env_file),
        "--methods",
        args.method,
        "--samples",
        "1",
        "--hls-backend",
        args.hls_backend,
        "--max-llm-calls",
        str(args.max_llm_calls),
        "--repair-log-token-budget",
        str(args.repair_log_token_budget),
        "--case-filter",
        re.escape(case_path.name) + "$",
        "--out-dir",
        str(out_dir),
    ]
    print("Running:", flush=True)
    print(" ".join(command), flush=True)
    print("", flush=True)
    env = os.environ.copy()
    proc = subprocess.run(command, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    run_dir = out_dir / args.method / case_path.name / "sample_0"
    if not run_dir.exists():
        run_dir = find_run_dir(out_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small TUI for CCD-HLS LOOP run artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)

    view = sub.add_parser("view", help="Open a TUI for an existing case run directory.")
    view.add_argument("run_dir", type=Path, help="Case sample dir, or an experiment dir containing result.json.")
    view.add_argument("--snapshot", action="store_true", help="Print a non-interactive summary instead of opening curses.")

    run = sub.add_parser("run", help="Run one ccd_hls_loop case, then open the TUI.")
    run.add_argument("--case-path", type=Path, default=Path("external/hls-eval/hls_eval_data/machsuite/spmv_crs"))
    run.add_argument("--hls-eval-root", type=Path, default=Path("external/hls-eval"))
    run.add_argument("--data-dir", type=Path, default=Path("external/hls-eval/hls_eval_data"))
    run.add_argument("--model-config", type=Path, default=Path("configs/deepseek_v4_flash.json"))
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--hls-backend", choices=["hls_eval", "vitis", "command", "mock"], default="vitis")
    run.add_argument("--method", default="ccd_hls_loop")
    run.add_argument("--max-llm-calls", type=int, default=2)
    run.add_argument("--repair-log-token-budget", type=int, default=1200)
    run.add_argument("--out-dir", type=Path, default=None)
    run.add_argument("--no-view", action="store_true", help="Do not open curses after the run.")
    run.add_argument("--snapshot", action="store_true", help="Print a summary after the run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "view":
        if args.snapshot or not sys.stdout.isatty():
            print_snapshot(args.run_dir)
        else:
            open_tui(args.run_dir)
        return
    run_dir = run_benchmark(args)
    print("")
    print(f"Run artifacts: {short_path(run_dir)}")
    if args.snapshot or args.no_view or not sys.stdout.isatty():
        print("")
        print_snapshot(run_dir)
        return
    open_tui(run_dir)


if __name__ == "__main__":
    main()
