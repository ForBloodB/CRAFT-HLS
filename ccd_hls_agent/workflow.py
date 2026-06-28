from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .json_utils import make_json_safe
from .utils import utc_now, write_text


WORKFLOW_STATUS = "workflow_status.json"
WORKFLOW_EVENTS = "workflow_events.jsonl"


def stage_label(stage: str, attempt: int | None = None) -> str:
    labels = {
        "DIAGNOSIS": "第一阶段诊断",
        "CONTRACT_PREPARE": "准备合同",
        "CONTRACT_WRITE": "写出 HLS-Eval 合同",
        "CONTRACT_FILL": "LLM 补全合同",
        "CONTRACT_REVIEW": "用户审阅合同",
        "CONTRACT_LOCK": "锁定合同",
        "CONTRACT_APPROVAL": "用户确认合同",
        "GENERATION": "初次尝试：生成代码",
        "PARSE_VALIDATE": "初次尝试：解析代码",
        "FORMAT_REPAIR": "格式修正",
        "CSIM": "功能仿真 CSIM",
        "CSIM_REPAIR": "修正：诊断并修复 CSIM",
        "SYNTH": "综合 SYNTH",
        "SYNTH_REPAIR": "修正：诊断并修复 SYNTH",
        "COSIM": "RTL 协同仿真 COSIM",
        "COSIM_REPAIR": "修正：诊断并修复 COSIM",
        "DONE": "完成",
        "FAILED": "失败",
    }
    label = labels.get(stage, stage)
    if attempt and stage.endswith("_REPAIR"):
        return f"第{attempt}次修正：{label}"
    return label


def build_workflow_status(
    *,
    run_dir: Path,
    current_stage: str,
    status: str,
    attempt_index: int = 0,
    message: str = "",
    last_artifact_uri: str | None = None,
    contract_uri: str | None = None,
) -> dict[str, Any]:
    return make_json_safe(
        {
            "updated_at": utc_now(),
            "current_stage": current_stage,
            "current_stage_label": stage_label(current_stage, attempt_index if attempt_index > 0 else None),
            "status": status,
            "attempt_index": attempt_index,
            "message": message,
            "last_artifact_uri": last_artifact_uri,
            "contract_uri": contract_uri,
            "run_dir": str(run_dir),
        }
    )


def write_workflow_status(run_dir: Path, status: dict[str, Any]) -> None:
    write_text(run_dir / WORKFLOW_STATUS, json.dumps(make_json_safe(status), ensure_ascii=False, indent=2))


def append_workflow_event(run_dir: Path, event: dict[str, Any]) -> None:
    event = make_json_safe({"created_at": utc_now(), **event})
    path = run_dir / WORKFLOW_EVENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_workflow_artifacts_from_stage_records(run_dir: Path, records: list[dict[str, Any]], *, final_status: str, message: str = "") -> dict[str, Any]:
    attempt_index = 0
    current_stage = "DONE" if final_status == "done" else "FAILED"
    last_artifact_uri = None
    for record in records:
        stage = str(record.get("stage") or "")
        if stage.endswith("_REPAIR"):
            attempt_index += 1
        current_stage = stage or current_stage
        artifacts = record.get("artifacts") or {}
        if artifacts:
            last_artifact_uri = next(reversed(artifacts.values()))
        append_workflow_event(
            run_dir,
            {
                "stage": stage,
                "stage_label": stage_label(stage, attempt_index if stage.endswith("_REPAIR") else None),
                "status": record.get("status"),
                "message": record.get("message"),
                "attempt_index": attempt_index,
                "artifact_uri": last_artifact_uri,
            },
        )
    terminal_stage = "DONE" if final_status == "done" else current_stage
    status = build_workflow_status(
        run_dir=run_dir,
        current_stage=terminal_stage,
        status=final_status,
        attempt_index=attempt_index,
        message=message,
        last_artifact_uri=last_artifact_uri,
    )
    write_workflow_status(run_dir, status)
    return status
