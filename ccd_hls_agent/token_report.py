from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .json_utils import make_json_safe
from .utils import write_text


TOKEN_STAGES = [
    "DIAGNOSIS",
    "CONTRACT_WRITE",
    "CONTRACT_FILL",
    "CONTRACT_REVIEW",
    "CONTRACT_LOCK",
    "GENERATION",
    "FORMAT_REPAIR",
    "CSIM_REPAIR",
    "SYNTH_REPAIR",
    "COSIM_REPAIR",
    "PPA_OPT",
]


def _empty_stage_map() -> dict[str, int]:
    return {stage: 0 for stage in TOKEN_STAGES}


def build_token_report(result: Any) -> dict[str, Any]:
    metrics = getattr(result, "metrics", {}) or {}
    records = metrics.get("stage_records", []) or []
    tokens_by_stage = _empty_stage_map()
    llm_calls_by_stage = _empty_stage_map()
    tool_calls_by_stage = _empty_stage_map()

    for record in records:
        stage = str(record.get("stage") or "")
        stage_key = stage if stage in tokens_by_stage else None
        record_metrics = record.get("metrics") or {}
        prompt = int(record_metrics.get("prompt_tokens") or record_metrics.get("prompt_tokens_est") or 0)
        completion = int(record_metrics.get("completion_tokens") or record_metrics.get("completion_tokens_est") or 0)
        has_llm_call = bool(prompt or completion or record_metrics.get("llm_calls_used"))
        if stage_key and has_llm_call:
            tokens_by_stage[stage_key] += prompt + completion
            llm_calls_by_stage[stage_key] += 1
        if stage in {"CSIM", "SYNTH", "COSIM"}:
            tool_key = {"CSIM": "CSIM_REPAIR", "SYNTH": "SYNTH_REPAIR", "COSIM": "COSIM_REPAIR"}[stage]
            tool_calls_by_stage[tool_key] += 1

    total_prompt = int(getattr(result, "prompt_tokens", 0) or 0)
    total_completion = int(getattr(result, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(result, "total_tokens", total_prompt + total_completion) or 0)
    if not total_tokens:
        stage_total = sum(tokens_by_stage.values())
        if stage_total:
            total_prompt = stage_total
            total_tokens = stage_total
    task_mode = metrics.get("task_mode") or "generate"
    report = {
        "case_name": getattr(result, "case_name", ""),
        "method": getattr(result, "method", ""),
        "sample_idx": getattr(result, "sample_idx", 0),
        "task_mode": task_mode,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "tokens_by_stage": tokens_by_stage,
        "llm_calls_by_stage": llm_calls_by_stage,
        "tool_calls_by_stage": tool_calls_by_stage,
        "tokens_per_success": total_tokens if getattr(result, "can_synthesize", False) else None,
    }
    return make_json_safe(report)


def write_case_token_report(result: Any, out_dir: Path) -> dict[str, Any]:
    report = build_token_report(result)
    write_text(out_dir / "token_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    with (out_dir / "token_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "tokens", "llm_calls", "tool_calls"])
        writer.writeheader()
        for stage in TOKEN_STAGES:
            writer.writerow(
                {
                    "stage": stage,
                    "tokens": report["tokens_by_stage"].get(stage, 0),
                    "llm_calls": report["llm_calls_by_stage"].get(stage, 0),
                    "tool_calls": report["tool_calls_by_stage"].get(stage, 0),
                }
            )
    return report


def aggregate_token_reports(results: list[Any]) -> list[dict[str, Any]]:
    rows = []
    methods = sorted(set(getattr(result, "method", "") for result in results))
    for method in methods:
        method_results = [result for result in results if getattr(result, "method", "") == method]
        reports = [build_token_report(result) for result in method_results]
        count = max(1, len(reports))
        synth_successes = [report["total_tokens"] for result, report in zip(method_results, reports) if getattr(result, "can_synthesize", False)]
        csim_successes = [
            report["total_tokens"]
            for result, report in zip(method_results, reports)
            if getattr(result, "can_compile", False) and getattr(result, "can_pass_testbench", False)
        ]
        row: dict[str, Any] = {
            "method": method,
            "n_samples_total": len(reports),
            "tokens_per_synth_success": sum(synth_successes) / len(synth_successes) if synth_successes else None,
            "tokens_per_csim_success": sum(csim_successes) / len(csim_successes) if csim_successes else None,
        }
        for stage in TOKEN_STAGES:
            row[f"avg_tokens_{stage.lower()}"] = sum(report["tokens_by_stage"].get(stage, 0) for report in reports) / count
            row[f"llm_calls_{stage.lower()}"] = sum(report["llm_calls_by_stage"].get(stage, 0) for report in reports)
            row[f"tool_calls_{stage.lower()}"] = sum(report["tool_calls_by_stage"].get(stage, 0) for report in reports)
        rows.append(make_json_safe(row))
    return rows


def write_token_summary(results: list[Any], out_dir: Path) -> list[dict[str, Any]]:
    rows = aggregate_token_reports(results)
    if not rows:
        write_text(out_dir / "token_summary.csv", "method\n")
        return rows
    fieldnames = list(rows[0].keys())
    with (out_dir / "token_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
