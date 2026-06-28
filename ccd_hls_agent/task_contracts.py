from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .json_utils import make_json_safe
from .utils import estimate_tokens, new_id, read_text, utc_now, write_text
from .workflow import append_workflow_event, build_workflow_status, write_workflow_status


DIAGNOSIS_TYPES = {
    "A_FUNCTIONALLY_CORRECT_BUT_UNOPTIMIZED_BASELINE": "A functionally correct but unoptimized baseline C/C++ implementation.",
    "B_FAILS_COMPILATION_OR_SYNTHESIS": "An HLS design that fails compilation or synthesis.",
    "C_COMPILES_BUT_FAILS_CSIM_COSIM_OR_HIDDEN_TESTS": "An HLS design that compiles successfully but fails C simulation, co-simulation, or hidden functional tests.",
    "D_STRUCTURAL_DEADLOCK_STREAMING_OR_RESOURCE_ISSUE": "An HLS design that exhibits structural issues such as deadlock, invalid streaming behavior, or severe resource inefficiency.",
    "E_OTHER_HLS_COMPILATION_PROBLEM": "Other problems related to HLS compilation.",
}

CONTRACT_META = "contract_meta.json"
DIAGNOSIS_FILE = "diagnosis.json"
CONTRACT_STAGE_RECORDS = "contract_stage_records.json"
CONTRACT_TOKEN_REPORT = "contract_token_report.json"
WORKFLOW_TOKEN_SUMMARY = "workflow_token_summary.json"
DEFAULT_PLATFORM = "KV260"
TODO_SIGNATURE = "// TODO: complete top function signature"
TODO_BEHAVIOR = "TODO: describe the required functional behavior and correctness criteria."
TODO_TESTBENCH = "// TODO: add functional checks for the contract."


def _extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:c|cpp|c\+\+)?\s*(.*?)```", text, re.S | re.I)


def _find_signature(text: str) -> tuple[str, str]:
    match = re.search(
        r"\b(?:void|int|float|double|bool|ap_uint<[^>]+>|ap_int<[^>]+>|[A-Za-z_][\w:<>]*)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*;?",
        text,
    )
    if not match:
        return "", ""
    signature = match.group(0).strip()
    if not signature.endswith(";"):
        signature += ";"
    return signature, match.group(1)


def diagnose_request(source_request: str) -> dict[str, Any]:
    text = source_request or ""
    lower = text.lower()
    code_blocks = _extract_code_blocks(text)
    has_code = bool(code_blocks or re.search(r"\b(?:void|int|float|double)\s+[A-Za-z_]\w*\s*\(", text))
    evidence: list[str] = []
    missing: list[str] = []
    diagnosis_type = "E_OTHER_HLS_COMPILATION_PROBLEM"
    confidence = 0.45

    if re.search(r"deadlock|fifo|stream|dataflow|invalid streaming|hang|stall|severe resource|ii violation|resource ineff", lower):
        diagnosis_type = "D_STRUCTURAL_DEADLOCK_STREAMING_OR_RESOURCE_ISSUE"
        confidence = 0.75
        evidence.append("Detected structural/dataflow/resource keywords.")
    elif re.search(r"mismatch|csim failed|cosim failed|hidden test|testbench|wrong answer|functional test", lower):
        diagnosis_type = "C_COMPILES_BUT_FAILS_CSIM_COSIM_OR_HIDDEN_TESTS"
        confidence = 0.75
        evidence.append("Detected simulation or hidden-test failure keywords.")
    elif re.search(r"compile error|compilation failed|synthesis failed|synth failed|not declared|no matching|undefined reference|vitis.*error", lower):
        diagnosis_type = "B_FAILS_COMPILATION_OR_SYNTHESIS"
        confidence = 0.75
        evidence.append("Detected compile/synthesis failure keywords.")
    elif has_code:
        diagnosis_type = "A_FUNCTIONALLY_CORRECT_BUT_UNOPTIMIZED_BASELINE"
        confidence = 0.6
        evidence.append("Detected C/C++ code or function signature but no explicit failure log.")
    else:
        evidence.append("No concrete code or HLS log detected; contract will describe a generation task with TODO implementation details.")
        missing.append("implementation_or_reference_code")

    signature, top = _find_signature(text)
    if not top:
        top_match = re.search(r"(?:top function|top-level function|顶层函数|函数)\s*[:：`]?\s*`?([A-Za-z_]\w*)`?", text, re.I)
        top = top_match.group(1) if top_match else ""
    if not top:
        missing.append("top.txt")
    if not signature:
        missing.append("header_signature")
    if not re.search(r"test|tb|testbench|expected|输入|输出|验证|正确", text, re.I):
        missing.append("testbench_or_expected_behavior")

    return make_json_safe(
        {
            "diagnosis_type": diagnosis_type,
            "description": DIAGNOSIS_TYPES[diagnosis_type],
            "confidence": confidence,
            "has_code": has_code,
            "code_blocks": code_blocks[:3],
            "inferred_top_function": top,
            "inferred_signature": signature,
            "missing_fields": sorted(set(missing)),
            "evidence": evidence,
            "source_excerpt": text[:4000],
            "created_at": utc_now(),
        }
    )


def _kernel_name(top: str) -> str:
    return top or "kernel"


def _contract_files(meta: dict[str, Any]) -> list[str]:
    top = _kernel_name(str(meta.get("top_function") or "kernel"))
    return [
        "kernel_description.md",
        "top.txt",
        f"{top}.h",
        f"{top}.cpp",
        f"{top}_tb.cpp",
        "hls_eval_config.toml",
        DIAGNOSIS_FILE,
        CONTRACT_META,
    ]


def _write_contract_stage(contract_dir: Path, stage: str, status: str, message: str, *, prompt_tokens: int = 0, completion_tokens: int = 0, tool_calls: int = 1) -> None:
    records = load_contract_stage_records(contract_dir)
    records.append(
        make_json_safe(
            {
                "stage": stage,
                "status": status,
                "message": message,
                "created_at": utc_now(),
                "metrics": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "tool_calls": tool_calls,
                },
            }
        )
    )
    write_text(contract_dir / CONTRACT_STAGE_RECORDS, json.dumps(records, ensure_ascii=False, indent=2))
    write_contract_token_report(contract_dir)
    append_workflow_event(contract_dir, {"stage": stage, "status": status, "message": message, "attempt_index": 0})


def load_contract_stage_records(contract_dir: Path) -> list[dict[str, Any]]:
    path = contract_dir / CONTRACT_STAGE_RECORDS
    if not path.exists():
        return []
    return json.loads(read_text(path))


def write_contract_token_report(contract_dir: Path) -> dict[str, Any]:
    records = load_contract_stage_records(contract_dir)
    stages = ["DIAGNOSIS", "CONTRACT_WRITE", "CONTRACT_FILL", "CONTRACT_REVIEW", "CONTRACT_LOCK"]
    tokens_by_stage = {stage: 0 for stage in stages}
    llm_calls_by_stage = {stage: 0 for stage in stages}
    tool_calls_by_stage = {stage: 0 for stage in stages}
    for record in records:
        stage = str(record.get("stage") or "")
        if stage not in tokens_by_stage:
            continue
        metrics = record.get("metrics") or {}
        tokens = int(metrics.get("total_tokens") or 0)
        tokens_by_stage[stage] += tokens
        if tokens:
            llm_calls_by_stage[stage] += 1
        tool_calls_by_stage[stage] += int(metrics.get("tool_calls") or 0)
    report = make_json_safe(
        {
            "contract_dir": str(contract_dir),
            "total_tokens": sum(tokens_by_stage.values()),
            "tokens_by_stage": tokens_by_stage,
            "llm_calls_by_stage": llm_calls_by_stage,
            "tool_calls_by_stage": tool_calls_by_stage,
        }
    )
    write_text(contract_dir / CONTRACT_TOKEN_REPORT, json.dumps(report, ensure_ascii=False, indent=2))
    return report


def prepare_hls_eval_contract(source_request: str, contract_dir: Path, *, target_platform: str = DEFAULT_PLATFORM, task_id: str | None = None) -> dict[str, Any]:
    contract_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = diagnose_request(source_request)
    top = str(diagnosis.get("inferred_top_function") or "")
    signature = str(diagnosis.get("inferred_signature") or "")
    kernel = _kernel_name(top)
    missing = set(diagnosis.get("missing_fields") or [])
    if not top:
        missing.add("top.txt")
    if not signature:
        missing.add("header_signature")

    write_text(contract_dir / DIAGNOSIS_FILE, json.dumps(diagnosis, ensure_ascii=False, indent=2))
    write_text(contract_dir / "top.txt", (top + "\n") if top else "")
    write_text(
        contract_dir / "kernel_description.md",
        "\n".join(
            [
                "## Phase-0 Diagnosis",
                f"- Type: {diagnosis['diagnosis_type']}",
                f"- Description: {diagnosis['description']}",
                f"- Confidence: {diagnosis['confidence']}",
                f"- Evidence: {'; '.join(diagnosis.get('evidence') or []) or 'TODO'}",
                f"- User confirmation needed: {', '.join(sorted(missing)) or 'none'}",
                "",
                "Kernel Description:",
                source_request.strip() or TODO_BEHAVIOR,
                "",
                "---",
                "",
                f"Top-Level Function: `{top or 'TODO'}`",
                "",
                "Complete Function Signature of the Top-Level Function:",
                f"`{signature or TODO_SIGNATURE}`",
                "",
                "Inputs:",
                "- TODO",
                "",
                "Outputs:",
                "- TODO",
                "",
                "Important Data Structures and Data Types:",
                "- TODO",
                "",
                "Sub-Components:",
                "- TODO",
            ]
        )
        + "\n",
    )
    header = contract_dir / f"{kernel}.h"
    cpp = contract_dir / f"{kernel}.cpp"
    tb = contract_dir / f"{kernel}_tb.cpp"
    write_text(header, "#pragma once\n\n" + (signature if signature else TODO_SIGNATURE) + ("\n" if signature else "\n"))
    write_text(cpp, f'#include "{header.name}"\n\n// TODO: provide or generate HLS implementation for {kernel}.\n')
    write_text(tb, f'#include "{header.name}"\n\nint main() {{\n    {TODO_TESTBENCH}\n    return 0;\n}}\n')
    write_text(
        contract_dir / "hls_eval_config.toml",
        "\n".join(
            [
                'tags = ["contract"]',
                f'target_platform = "{target_platform}"',
                'board = "KV260"' if target_platform.upper() == "KV260" else f'board = "{target_platform}"',
            ]
        )
        + "\n",
    )
    meta = {
        "status": "needs_user_input" if missing else "draft",
        "task_id": task_id or new_id("hls_contract"),
        "target_platform": target_platform,
        "source_request": source_request,
        "diagnosis_type": diagnosis["diagnosis_type"],
        "diagnosis_description": diagnosis["description"],
        "top_function": top,
        "missing_fields": sorted(missing),
        "approved_by_user": False,
        "approved_at": None,
        "contract_hash": None,
        "locked_files": [],
        "llm_fill_rounds": 0,
    }
    write_text(contract_dir / CONTRACT_META, json.dumps(make_json_safe(meta), ensure_ascii=False, indent=2))
    _write_contract_stage(contract_dir, "DIAGNOSIS", "completed", f"Diagnosed as {diagnosis['diagnosis_type']}", prompt_tokens=estimate_tokens(source_request))
    _write_contract_stage(contract_dir, "CONTRACT_WRITE", "completed", "Wrote HLS-Eval-like contract directory.")
    status = build_workflow_status(
        run_dir=contract_dir,
        current_stage="CONTRACT_PREPARE",
        status="needs_user_input" if missing else "draft",
        message="Contract prepared; review and fill TODO fields before locking.",
        contract_uri=str(contract_dir),
    )
    write_workflow_status(contract_dir, status)
    return make_json_safe(meta)


def load_contract_meta(contract_dir: Path) -> dict[str, Any]:
    return json.loads(read_text(contract_dir / CONTRACT_META))


def _existing_contract_files(contract_dir: Path, meta: dict[str, Any]) -> list[Path]:
    return [contract_dir / name for name in _contract_files(meta) if (contract_dir / name).exists()]


def contract_hash(contract_dir: Path, meta: dict[str, Any] | None = None) -> str:
    meta = dict(meta or load_contract_meta(contract_dir))
    meta["approved_at"] = None
    meta["contract_hash"] = None
    payload_parts = []
    for path in sorted(_existing_contract_files(contract_dir, meta)):
        if path.name in {CONTRACT_STAGE_RECORDS, CONTRACT_TOKEN_REPORT}:
            continue
        text = read_text(path)
        if path.name == CONTRACT_META:
            text = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        payload_parts.append(f"## {path.name}\n{text}")
    payload = "\n".join(payload_parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def review_contract(contract_dir: Path) -> dict[str, Any]:
    meta = load_contract_meta(contract_dir)
    missing = []
    top = read_text(contract_dir / "top.txt").strip() if (contract_dir / "top.txt").exists() else ""
    if not top:
        missing.append("top.txt")
    header_files = sorted(contract_dir.glob("*.h"))
    if not header_files or TODO_SIGNATURE in read_text(header_files[0]):
        missing.append("header_signature")
    tb_files = sorted(contract_dir.glob("*_tb.cpp"))
    if not tb_files or TODO_TESTBENCH in read_text(tb_files[0]):
        missing.append("testbench_or_expected_behavior")
    desc = read_text(contract_dir / "kernel_description.md") if (contract_dir / "kernel_description.md").exists() else ""
    if TODO_BEHAVIOR in desc or "Kernel Description:" not in desc:
        missing.append("kernel_description")
    meta["missing_fields"] = sorted(set(missing))
    if meta.get("status") != "approved":
        meta["status"] = "needs_user_input" if missing else "draft"
    write_text(contract_dir / CONTRACT_META, json.dumps(make_json_safe(meta), ensure_ascii=False, indent=2))
    _write_contract_stage(contract_dir, "CONTRACT_REVIEW", "completed", f"Review found {len(missing)} missing fields.")
    return make_json_safe({"meta": meta, "missing_fields": meta["missing_fields"], "token_report": write_contract_token_report(contract_dir)})


def review_text(contract_dir: Path) -> str:
    meta = load_contract_meta(contract_dir)
    token_report_path = contract_dir / CONTRACT_TOKEN_REPORT
    token_report = json.loads(read_text(token_report_path)) if token_report_path.exists() else write_contract_token_report(contract_dir)
    lines = [
        f"Contract: {contract_dir}",
        f"Status: {meta.get('status')}",
        f"Target platform: {meta.get('target_platform')}",
        f"Diagnosis: {meta.get('diagnosis_type')}",
        f"Missing fields: {', '.join(meta.get('missing_fields') or []) or 'none'}",
        f"Approved: {meta.get('approved_by_user')}",
        f"Contract tokens: {token_report.get('total_tokens')}",
    ]
    return "\n".join(lines)


def lock_contract(contract_dir: Path) -> dict[str, Any]:
    review = review_contract(contract_dir)
    meta = review["meta"]
    if meta.get("missing_fields"):
        raise ValueError("Cannot lock contract with missing fields: " + ", ".join(meta["missing_fields"]))
    meta["status"] = "approved"
    meta["approved_by_user"] = True
    meta["approved_at"] = utc_now()
    meta["locked_files"] = [path.name for path in _existing_contract_files(contract_dir, meta)]
    meta["contract_hash"] = contract_hash(contract_dir, meta)
    write_text(contract_dir / CONTRACT_META, json.dumps(make_json_safe(meta), ensure_ascii=False, indent=2))
    _write_contract_stage(contract_dir, "CONTRACT_LOCK", "completed", "Contract locked by explicit user command.")
    write_workflow_status(
        contract_dir,
        build_workflow_status(
            run_dir=contract_dir,
            current_stage="CONTRACT_LOCK",
            status="approved",
            message="Contract locked; backend execution is now allowed.",
            contract_uri=str(contract_dir),
        ),
    )
    return make_json_safe(meta)


def ensure_contract_locked(contract_dir: Path) -> dict[str, Any]:
    meta = load_contract_meta(contract_dir)
    if meta.get("status") != "approved" or meta.get("approved_by_user") is not True or not meta.get("contract_hash"):
        raise ValueError("Contract must be locked with /hls-lock-and-run before execution.")
    current = contract_hash(contract_dir, meta)
    if current != meta["contract_hash"]:
        raise ValueError("Contract changed after lock; review and lock it again.")
    return meta


def mark_contract_running(contract_dir: Path) -> None:
    meta = load_contract_meta(contract_dir)
    meta["status"] = "running"
    write_text(contract_dir / CONTRACT_META, json.dumps(make_json_safe(meta), ensure_ascii=False, indent=2))
    write_workflow_status(
        contract_dir,
        build_workflow_status(run_dir=contract_dir, current_stage="GENERATION", status="running", message="Backend run started.", contract_uri=str(contract_dir)),
    )


def copy_contract_case(contract_dir: Path, case_dir: Path) -> Path:
    ensure_contract_locked(contract_dir)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    for path in contract_dir.iterdir():
        if path.is_file() and path.name not in {CONTRACT_META, DIAGNOSIS_FILE, CONTRACT_STAGE_RECORDS, CONTRACT_TOKEN_REPORT, "workflow_status.json", "workflow_events.jsonl", WORKFLOW_TOKEN_SUMMARY}:
            shutil.copy2(path, case_dir / path.name)
    shutil.copy2(contract_dir / DIAGNOSIS_FILE, case_dir / DIAGNOSIS_FILE)
    shutil.copy2(contract_dir / CONTRACT_META, case_dir / CONTRACT_META)
    return case_dir


def summarize_resolution(contract_dir: Path, run_dir: Path) -> dict[str, Any]:
    meta = load_contract_meta(contract_dir)
    result_path = run_dir / "result.json"
    result = json.loads(read_text(result_path)) if result_path.exists() else {}
    diagnosis = meta.get("diagnosis_type")
    if diagnosis == "A_FUNCTIONALLY_CORRECT_BUT_UNOPTIMIZED_BASELINE":
        solved = bool(result.get("can_synthesize"))
        basis = "SYNTH passed" if solved else "SYNTH did not pass"
    elif diagnosis == "B_FAILS_COMPILATION_OR_SYNTHESIS":
        solved = bool(result.get("can_compile") and result.get("can_synthesize"))
        basis = f"compile={result.get('can_compile')} synth={result.get('can_synthesize')}"
    elif diagnosis == "C_COMPILES_BUT_FAILS_CSIM_COSIM_OR_HIDDEN_TESTS":
        solved = bool(result.get("can_pass_testbench"))
        basis = f"CSIM/testbench pass={result.get('can_pass_testbench')}"
    elif diagnosis == "D_STRUCTURAL_DEADLOCK_STREAMING_OR_RESOURCE_ISSUE":
        solved = bool(result.get("can_synthesize"))
        basis = "SYNTH passed; inspect logs for dataflow/resource warnings"
    else:
        solved = bool(result.get("can_synthesize"))
        basis = f"synth={result.get('can_synthesize')}"
    summary = make_json_safe(
        {
            "diagnosis_type": diagnosis,
            "diagnosis_description": meta.get("diagnosis_description"),
            "resolved": solved,
            "basis": basis,
            "result_path": str(result_path),
            "run_dir": str(run_dir),
        }
    )
    write_text(contract_dir / "resolution_report.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_workflow_token_summary(contract_dir, run_dir)
    return summary


def write_workflow_token_summary(contract_dir: Path, run_dir: Path | None = None) -> dict[str, Any]:
    contract_report_path = contract_dir / CONTRACT_TOKEN_REPORT
    contract_report = json.loads(read_text(contract_report_path)) if contract_report_path.exists() else write_contract_token_report(contract_dir)
    run_report = {}
    if run_dir is not None and (run_dir / "token_report.json").exists():
        run_report = json.loads(read_text(run_dir / "token_report.json"))
    total = int(contract_report.get("total_tokens") or 0) + int(run_report.get("total_tokens") or 0)
    summary = make_json_safe(
        {
            "contract_tokens": contract_report,
            "execution_tokens": run_report,
            "total_tokens": total,
            "run_dir": str(run_dir) if run_dir else None,
        }
    )
    write_text(contract_dir / WORKFLOW_TOKEN_SUMMARY, json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
