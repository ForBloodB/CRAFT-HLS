#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ccd_hls_agent.hls_backends import HLSBackendConfig, ToolResult, build_hls_backend
from ccd_hls_agent.json_utils import make_json_safe
from ccd_hls_agent.utils import read_text, utc_now, write_text


SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}
DEFAULT_RUNS = {
    "hls_eval_zero_shot": Path("experiments/full/full_deepseek_v4_flash_vitis_20260531_final"),
    "hls_eval_agentic": Path("experiments/full/hls_eval_agentic_deepseek_94x1_20260531"),
    "ccd_hls_v2": Path("experiments/full/full_ccd_hls_gen_v2_deepseek_20260531"),
    "ccd_hls_loop": Path("experiments/full/full_ccd_hls_gen_v2_repair_deepseek_20260531_163049"),
}


@dataclass
class CosimResult:
    source_run: str
    source_dir: str
    case_name: str
    sample_idx: int
    case_path: str
    can_synthesize: bool
    cosim_attempted: bool
    can_cosim: bool
    return_code: int | None
    skipped_reason: str | None
    duration_ms: int
    command: str
    metrics: dict[str, Any]
    artifacts_dir: str


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    result_path = run_dir / "results.jsonl"
    return [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_generated_design(row: dict[str, Any]) -> Path:
    artifacts = Path(row["artifacts_dir"])
    candidates = [artifacts / "design", artifacts / "design_generated", artifacts / "agent_run_dir"]
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("*.cpp")):
            return candidate
    raise FileNotFoundError(f"No generated design directory found under {artifacts}")


def stage_design(row: dict[str, Any], workdir: Path) -> Path:
    src = find_generated_design(row)
    dst = workdir / "design"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    case_path = Path(row["case_path"])
    for name in ["top.txt", "hls_eval_config.toml", "kernel_description.md"]:
        source = case_path / name
        target = dst / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)

    config_text = read_text(case_path / "hls_eval_config.toml")
    for line in config_text.splitlines():
        item = line.strip().strip(",").strip("\"'")
        if not item or item.startswith("[") or item.startswith("tags") or item.startswith("tb_data"):
            continue
        source = case_path / item
        target = dst / item
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    return dst


def source_files(design_dir: Path) -> list[Path]:
    return sorted(p for p in design_dir.glob("*") if p.suffix in SOURCE_EXTENSIONS)


def save_tool_artifacts(result: ToolResult, workdir: Path) -> None:
    write_text(workdir / "cosim_stdout.log", result.stdout if isinstance(result.stdout, str) else result.stdout.decode("utf-8", "replace"))
    write_text(workdir / "cosim_stderr.log", result.stderr if isinstance(result.stderr, str) else result.stderr.decode("utf-8", "replace"))


def summarize(results: list[CosimResult], out_dir: Path) -> None:
    rows = [make_json_safe(asdict(r)) for r in results]
    write_text(out_dir / "results.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    fieldnames = [
        "source_run",
        "case_name",
        "sample_idx",
        "can_synthesize",
        "cosim_attempted",
        "can_cosim",
        "return_code",
        "skipped_reason",
        "duration_ms",
        "artifacts_dir",
    ]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    summary_rows = []
    for run in sorted({r.source_run for r in results}):
        items = [r for r in results if r.source_run == run]
        attempted = [r for r in items if r.cosim_attempted]
        passed = [r for r in items if r.can_cosim]
        summary_rows.append(
            {
                "source_run": run,
                "samples": len(items),
                "synth_passed": sum(1 for r in items if r.can_synthesize),
                "cosim_attempted": len(attempted),
                "cosim_passed": len(passed),
                "cosim_pass_rate_all": len(passed) / max(1, len(items)),
                "cosim_pass_rate_attempted": len(passed) / max(1, len(attempted)),
                "avg_cosim_duration_ms": sum(r.duration_ms for r in attempted) / max(1, len(attempted)),
            }
        )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["source_run"])
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["# COSIM Validation Summary", "", f"Generated at: {utc_now()}", ""]
    lines.append("| run | samples | synth passed | cosim attempted | cosim passed | pass rate all | pass rate attempted | avg duration ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['source_run']} | {row['samples']} | {row['synth_passed']} | {row['cosim_attempted']} | "
            f"{row['cosim_passed']} | {row['cosim_pass_rate_all']:.2%} | {row['cosim_pass_rate_attempted']:.2%} | "
            f"{row['avg_cosim_duration_ms']:.1f} |"
        )
    write_text(out_dir / "summary.md", "\n".join(lines) + "\n")


def parse_run_selection(value: str) -> dict[str, Path]:
    if value == "all":
        return DEFAULT_RUNS
    selected: dict[str, Path] = {}
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in DEFAULT_RUNS:
            raise ValueError(f"Unknown run key {key}; choose one of {sorted(DEFAULT_RUNS)} or all")
        selected[key] = DEFAULT_RUNS[key]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vitis RTL cosim validation for retained HLS-Eval experiment results.")
    parser.add_argument("--runs", default="all", help="Comma-separated run keys or all.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/cosim/cosim_validation_20260531"))
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import re

    selected = parse_run_selection(args.runs)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = build_hls_backend("vitis", HLSBackendConfig(timeout_seconds=args.timeout_seconds))
    results: list[CosimResult] = []

    for run_key, run_dir in selected.items():
        rows = load_rows(run_dir)
        if args.case_filter:
            pattern = re.compile(args.case_filter)
            rows = [row for row in rows if pattern.search(row["case_name"]) or pattern.search(row["case_path"])]
        if args.limit:
            rows = rows[: args.limit]
        for idx, row in enumerate(rows, start=1):
            sample_idx = int(row.get("sample_idx", 0))
            case_name = row["case_name"]
            workdir = out_dir / run_key / case_name / f"sample_{sample_idx}"
            result_path = workdir / "result.json"
            if args.resume and result_path.exists():
                existing = CosimResult(**json.loads(read_text(result_path)))
                results.append(existing)
                print(f"[{run_key} {idx}/{len(rows)}] skip existing {case_name} sample={sample_idx}")
                summarize(results, out_dir)
                continue
            workdir.mkdir(parents=True, exist_ok=True)
            print(f"[{run_key} {idx}/{len(rows)}] cosim {case_name} sample={sample_idx}")

            if not row.get("can_synthesize"):
                result = CosimResult(
                    source_run=run_key,
                    source_dir=str(run_dir),
                    case_name=case_name,
                    sample_idx=sample_idx,
                    case_path=row["case_path"],
                    can_synthesize=False,
                    cosim_attempted=False,
                    can_cosim=False,
                    return_code=None,
                    skipped_reason="synth_failed",
                    duration_ms=0,
                    command="",
                    metrics={},
                    artifacts_dir=str(workdir),
                )
            else:
                design = stage_design(row, workdir)
                top = read_text(design / "top.txt").strip()
                t0 = time.monotonic()
                tool_result = backend.run_cosim(
                    workdir / "cosim",
                    source_files(design),
                    {"top_function": top, "timeout_seconds": args.timeout_seconds},
                )
                duration_ms = int((time.monotonic() - t0) * 1000)
                save_tool_artifacts(tool_result, workdir)
                result = CosimResult(
                    source_run=run_key,
                    source_dir=str(run_dir),
                    case_name=case_name,
                    sample_idx=sample_idx,
                    case_path=row["case_path"],
                    can_synthesize=True,
                    cosim_attempted=True,
                    can_cosim=tool_result.return_code == 0,
                    return_code=tool_result.return_code,
                    skipped_reason=None,
                    duration_ms=duration_ms,
                    command=tool_result.command,
                    metrics=tool_result.metrics,
                    artifacts_dir=str(workdir),
                )
            write_text(result_path, json.dumps(make_json_safe(asdict(result)), ensure_ascii=False, indent=2))
            results.append(result)
            summarize(results, out_dir)
    summarize(results, out_dir)
    print(f"Summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
