from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .json_utils import make_json_safe


ACTION_ALIGN_SIGNATURE = "ALIGN_SIGNATURE"
ACTION_ADD_MISSING_INCLUDE = "ADD_MISSING_INCLUDE"
ACTION_FIX_ARRAY_SHAPE = "FIX_ARRAY_SHAPE"
ACTION_REWRITE_INDEX_ORDER = "REWRITE_INDEX_ORDER"
ACTION_FIX_STATIC_LOOP_BOUND = "FIX_STATIC_LOOP_BOUND"
ACTION_FIX_DATA_FILE_PATH = "FIX_DATA_FILE_PATH"
ACTION_LLM_SEMANTIC_REPAIR = "LLM_SEMANTIC_REPAIR"
ACTION_SYNTH_INTERFACE_REPAIR = "SYNTH_INTERFACE_REPAIR"

REPAIR_ACTION_IDS = {
    ACTION_ALIGN_SIGNATURE,
    ACTION_ADD_MISSING_INCLUDE,
    ACTION_FIX_ARRAY_SHAPE,
    ACTION_REWRITE_INDEX_ORDER,
    ACTION_FIX_STATIC_LOOP_BOUND,
    ACTION_FIX_DATA_FILE_PATH,
    ACTION_LLM_SEMANTIC_REPAIR,
    ACTION_SYNTH_INTERFACE_REPAIR,
}


@dataclass(frozen=True)
class RepairActionDefinition:
    action_id: str
    applies_to: list[str]
    preconditions: list[str]
    allowed_edits: list[str]
    forbidden_edits: list[str]
    verification: list[str]
    risk: str
    executor: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REPAIR_ACTIONS: dict[str, RepairActionDefinition] = {
    ACTION_ALIGN_SIGNATURE: RepairActionDefinition(
        action_id=ACTION_ALIGN_SIGNATURE,
        applies_to=["signature_or_top_mismatch", "missing_include_or_type"],
        preconditions=["top.txt and header prototype exist"],
        allowed_edits=["rename generated top function", "align kernel prototype with header"],
        forbidden_edits=["change testbench semantics", "invent new top function"],
        verification=["rerun CSIM compile", "rerun CSIM testbench when compile passes"],
        risk="low",
        executor="deterministic",
    ),
    ACTION_ADD_MISSING_INCLUDE: RepairActionDefinition(
        action_id=ACTION_ADD_MISSING_INCLUDE,
        applies_to=["signature_or_top_mismatch", "missing_include_or_type"],
        preconditions=["kernel source exists", "header source exists"],
        allowed_edits=["add required local header include", "add safe standard/ap_int include"],
        forbidden_edits=["move declarations into testbench", "remove required includes"],
        verification=["rerun CSIM compile"],
        risk="low",
        executor="deterministic",
    ),
    ACTION_FIX_ARRAY_SHAPE: RepairActionDefinition(
        action_id=ACTION_FIX_ARRAY_SHAPE,
        applies_to=["array_dimension_type_error"],
        preconditions=["compiler reports array/scalar or dimension misuse"],
        allowed_edits=["align array rank with header/testbench", "fix scalar-vs-array access"],
        forbidden_edits=["change public interface", "add unrelated pragmas"],
        verification=["rerun CSIM compile and testbench"],
        risk="medium",
        executor="deterministic_or_llm",
    ),
    ACTION_REWRITE_INDEX_ORDER: RepairActionDefinition(
        action_id=ACTION_REWRITE_INDEX_ORDER,
        applies_to=["array_dimension_type_error", "functional_mismatch"],
        preconditions=["evidence suggests transposed row/column indexing"],
        allowed_edits=["rewrite local index order", "preserve loop trip counts"],
        forbidden_edits=["change input/output layout unless contract says so"],
        verification=["rerun CSIM compile and testbench"],
        risk="medium",
        executor="deterministic_or_llm",
    ),
    ACTION_FIX_STATIC_LOOP_BOUND: RepairActionDefinition(
        action_id=ACTION_FIX_STATIC_LOOP_BOUND,
        applies_to=["synth_resource_or_loop_error", "synth_interface_error", "dataflow_deadlock_or_fifo", "timeout"],
        preconditions=["SYNTH failed after CSIM pass or timeout indicates unsafe dynamic structure"],
        allowed_edits=["replace dynamic loop bounds with static constants when contract allows", "simplify unsupported loop form"],
        forbidden_edits=["truncate functional domain", "hide tool failures with early returns"],
        verification=["rerun CSIM before SYNTH", "rerun SYNTH only after CSIM passes"],
        risk="medium",
        executor="llm_constrained",
    ),
    ACTION_FIX_DATA_FILE_PATH: RepairActionDefinition(
        action_id=ACTION_FIX_DATA_FILE_PATH,
        applies_to=["data_file_runtime_error"],
        preconditions=["testbench tries to read local data file"],
        allowed_edits=["stage runtime data files", "use basename paths expected by Vitis CSIM"],
        forbidden_edits=["remove file-based checks from testbench"],
        verification=["rerun CSIM"],
        risk="low",
        executor="deterministic",
    ),
    ACTION_LLM_SEMANTIC_REPAIR: RepairActionDefinition(
        action_id=ACTION_LLM_SEMANTIC_REPAIR,
        applies_to=["functional_mismatch", "numeric_tolerance_mismatch", "unknown_hls_failure"],
        preconditions=["compile passes or deterministic compile repair did not apply"],
        allowed_edits=["repair algorithmic behavior in kernel only", "preserve top signature and testbench contract"],
        forbidden_edits=["edit testbench to accept wrong output", "change interface without evidence"],
        verification=["rerun CSIM", "rerun SYNTH only after CSIM passes"],
        risk="high",
        executor="llm_constrained",
    ),
    ACTION_SYNTH_INTERFACE_REPAIR: RepairActionDefinition(
        action_id=ACTION_SYNTH_INTERFACE_REPAIR,
        applies_to=["synth_interface_error", "synth_resource_or_loop_error", "dataflow_deadlock_or_fifo"],
        preconditions=["CSIM passes and SYNTH fails"],
        allowed_edits=["remove unsupported C++ constructs", "add conservative HLS-compatible structure"],
        forbidden_edits=["change functional behavior", "overfit to synthesis log by deleting computation"],
        verification=["rerun CSIM before SYNTH", "rerun SYNTH after CSIM passes"],
        risk="medium",
        executor="llm_constrained",
    ),
}


@dataclass(frozen=True)
class DiagnosisAssertion:
    assertion_id: str
    case_name: str
    stage: str
    task_mode: str
    failure_class: str
    claim: str
    evidence: list[str]
    confidence: float
    affected_symbols: list[str]
    recommended_actions: list[str]
    blocked_actions: list[str]
    verification_plan: list[str]
    stop_policy: dict[str, Any]
    memory_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True)
class SelectedRepairAction:
    action_id: str
    score: float
    reasons: list[str]
    definition: dict[str, Any]
    memory_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


FAILURE_ACTION_RULES: dict[str, tuple[list[str], list[str], str]] = {
    "array_dimension_type_error": (
        [ACTION_FIX_ARRAY_SHAPE, ACTION_REWRITE_INDEX_ORDER],
        ["ADD_ARRAY_PARTITION"],
        "Compiler evidence indicates array rank, scalar/array, or index-order misuse.",
    ),
    "signature_or_top_mismatch": (
        [ACTION_ALIGN_SIGNATURE, ACTION_ADD_MISSING_INCLUDE],
        [],
        "Top function name, prototype, or call site appears inconsistent.",
    ),
    "missing_include_or_type": (
        [ACTION_ADD_MISSING_INCLUDE, ACTION_ALIGN_SIGNATURE],
        [],
        "Compilation evidence points to a missing include or undeclared type.",
    ),
    "data_file_runtime_error": (
        [ACTION_FIX_DATA_FILE_PATH],
        [],
        "CSIM runtime cannot find a required testbench data file.",
    ),
    "functional_mismatch": (
        [ACTION_LLM_SEMANTIC_REPAIR],
        [],
        "The design compiles but the functional result does not match the testbench contract.",
    ),
    "numeric_tolerance_mismatch": (
        [ACTION_LLM_SEMANTIC_REPAIR],
        [],
        "The design compiles but numerical tolerance or precision behavior is inconsistent.",
    ),
    "synth_interface_error": (
        [ACTION_SYNTH_INTERFACE_REPAIR, ACTION_FIX_STATIC_LOOP_BOUND],
        [],
        "Synthesis failed around interface, port, bundle, or unsupported construct handling.",
    ),
    "synth_resource_or_loop_error": (
        [ACTION_SYNTH_INTERFACE_REPAIR, ACTION_FIX_STATIC_LOOP_BOUND],
        [],
        "CSIM passed but synthesis failed, likely due to loop/resource/HLS-compatibility structure.",
    ),
    "dataflow_deadlock_or_fifo": (
        [ACTION_SYNTH_INTERFACE_REPAIR, ACTION_FIX_STATIC_LOOP_BOUND],
        [],
        "Tool evidence mentions dataflow, FIFO, stream, stall, or deadlock.",
    ),
    "timeout": (
        [ACTION_FIX_STATIC_LOOP_BOUND, ACTION_SYNTH_INTERFACE_REPAIR],
        [ACTION_LLM_SEMANTIC_REPAIR],
        "Tool execution timed out; avoid freeform semantic repair until structure is constrained.",
    ),
}


def action_ids_for_failure_class(failure_class: str) -> list[str]:
    return list(FAILURE_ACTION_RULES.get(failure_class, ([ACTION_LLM_SEMANTIC_REPAIR], [], "unknown failure"))[0])


def _evidence_from_capsule(capsule: dict[str, Any]) -> list[str]:
    evidence = [*map(str, capsule.get("key_errors", [])[:4]), *map(str, capsule.get("signal_lines", [])[:6])]
    evidence = [line.strip() for line in evidence if line and str(line).strip()]
    return list(dict.fromkeys(evidence))[:8]


def _affected_symbols(evidence: list[str]) -> list[str]:
    joined = "\n".join(evidence)
    symbols: list[str] = []
    for pattern in [
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[",
        r"undefined reference to [`']?([A-Za-z_][A-Za-z0-9_]*)",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s+was not declared",
        r"use of undeclared identifier [`']?([A-Za-z_][A-Za-z0-9_]*)",
        r"no matching function.*?([A-Za-z_][A-Za-z0-9_]*)",
    ]:
        for match in re.finditer(pattern, joined):
            value = match.group(1)
            if value not in {"error", "warning", "note", "candidate", "function"}:
                symbols.append(value)
    return list(dict.fromkeys(symbols))[:12]


def _assertion_id(case_name: str, stage: str, attempt: int, failure_class: str, evidence: list[str]) -> str:
    payload = json.dumps([case_name, stage, attempt, failure_class, evidence[:4]], ensure_ascii=False)
    return "assert_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_diagnosis_assertion(
    *,
    case_name: str,
    stage: str,
    task_mode: str,
    failure_capsule: dict[str, Any],
    attempt: int,
    memory_refs: list[str] | None = None,
    early_stop_similarity_threshold: float = 0.92,
) -> DiagnosisAssertion:
    failure_class = str(failure_capsule.get("failure_class") or failure_capsule.get("failure_type") or "unknown_hls_failure")
    recommended, blocked, claim = FAILURE_ACTION_RULES.get(
        failure_class,
        ([ACTION_LLM_SEMANTIC_REPAIR], [], "Failure class is unknown; use constrained semantic repair after deterministic checks."),
    )
    evidence = _evidence_from_capsule(failure_capsule)
    if not evidence:
        evidence = [str(failure_capsule.get("recommended_policy") or "No compact tool evidence was available.")]
    confidence = 0.55
    if failure_class != "unknown_hls_failure":
        confidence += 0.2
    if evidence and any(re.search(r"error|fatal|mismatch|failed|deadlock|fifo", item, re.I) for item in evidence):
        confidence += 0.15
    if failure_capsule.get("root_cause_hints"):
        confidence += 0.1
    confidence = min(confidence, 0.95)
    verification_plan = []
    for action_id in recommended:
        definition = REPAIR_ACTIONS.get(action_id)
        if definition:
            verification_plan.extend(definition.verification)
    verification_plan = list(dict.fromkeys(verification_plan or ["rerun CSIM", "rerun SYNTH only if CSIM passes"]))
    return DiagnosisAssertion(
        assertion_id=_assertion_id(case_name, stage, attempt, failure_class, evidence),
        case_name=case_name,
        stage=stage,
        task_mode=task_mode,
        failure_class=failure_class,
        claim=claim,
        evidence=evidence,
        confidence=round(confidence, 3),
        affected_symbols=_affected_symbols(evidence),
        recommended_actions=[action for action in recommended if action in REPAIR_ACTION_IDS],
        blocked_actions=blocked,
        verification_plan=verification_plan,
        stop_policy={
            "early_stop_similarity_threshold": early_stop_similarity_threshold,
            "stop_on_repeated_assertion": True,
            "freeform_repair_allowed": failure_class not in {"timeout"},
        },
        memory_refs=memory_refs or [],
    )


def select_repair_actions(
    assertion: DiagnosisAssertion,
    *,
    action_memory_hits: list[Any] | None = None,
    limit: int = 3,
) -> list[SelectedRepairAction]:
    hits = action_memory_hits or []
    blocked = set(assertion.blocked_actions)
    candidates: dict[str, tuple[float, list[str], list[str]]] = {}
    for action_id, definition in REPAIR_ACTIONS.items():
        if action_id in blocked:
            continue
        score = 0.0
        reasons: list[str] = []
        memory_refs: list[str] = []
        if assertion.failure_class in definition.applies_to:
            score += 2.0
            reasons.append(f"matches failure_class={assertion.failure_class}")
        if action_id in assertion.recommended_actions:
            score += 1.5
            reasons.append("recommended by diagnosis assertion")
        score += assertion.confidence * 0.5
        for hit in hits:
            if str(getattr(hit, "action_id", "")) != action_id:
                continue
            polarity = str(getattr(hit, "polarity", "positive"))
            hit_score = float(getattr(hit, "reuse_score", 0.0) or 0.0)
            ref = f"action_memory#{getattr(hit, 'memory_id', '?')}"
            memory_refs.append(ref)
            if polarity == "negative":
                score += min(hit_score, -4.0)
                reasons.append(f"negative memory penalty {ref}")
            else:
                score += max(hit_score, 1.0)
                reasons.append(f"positive memory boost {ref}")
        candidates[action_id] = (score, reasons, memory_refs)
    if assertion.recommended_actions:
        allowed = set(assertion.recommended_actions)
        filtered = {aid: value for aid, value in candidates.items() if aid in allowed}
        if filtered:
            candidates = filtered
    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
    selected: list[SelectedRepairAction] = []
    for action_id, (score, reasons, memory_refs) in ranked[:limit]:
        selected.append(
            SelectedRepairAction(
                action_id=action_id,
                score=round(score, 3),
                reasons=reasons or ["fallback action ranking"],
                definition=REPAIR_ACTIONS[action_id].to_dict(),
                memory_refs=memory_refs,
            )
        )
    return selected


def render_action_capsule(
    assertion: DiagnosisAssertion,
    selected_actions: list[SelectedRepairAction],
    *,
    action_memory_capsule: str = "",
) -> str:
    lines = [
        "## Diagnosis Assertion",
        f"- id: {assertion.assertion_id}",
        f"- class: {assertion.failure_class}",
        f"- claim: {assertion.claim}",
        f"- confidence: {assertion.confidence}",
        f"- affected_symbols: {', '.join(assertion.affected_symbols) if assertion.affected_symbols else 'unknown'}",
        "- evidence:",
    ]
    for item in assertion.evidence[:4]:
        lines.append(f"  - {item[:240]}")
    lines.append("- allowed repair actions:")
    for action in selected_actions:
        definition = action.definition
        lines.append(
            f"  - {action.action_id}: executor={definition.get('executor')}, risk={definition.get('risk')}, "
            f"allowed={'; '.join(definition.get('allowed_edits', [])[:2])}"
        )
        forbidden = "; ".join(definition.get("forbidden_edits", [])[:2])
        if forbidden:
            lines.append(f"    forbidden={forbidden}")
    if action_memory_capsule:
        lines.extend(["", "## Verified Action Memory", action_memory_capsule])
    return "\n".join(lines)


def candidate_score(candidate: dict[str, Any]) -> float:
    flags = candidate.get("flags") or {}
    total_tokens = float(candidate.get("total_tokens") or 0.0)
    diff_risk_penalty = float(candidate.get("diff_risk_penalty") or 0.0)
    negative_memory_penalty = float(candidate.get("negative_memory_penalty") or 0.0)
    score = 0.0
    if flags.get("can_synthesize"):
        score += 1000.0
    if flags.get("can_pass_testbench"):
        score += 300.0
    if flags.get("can_compile"):
        score += 100.0
    score -= total_tokens / 1000.0
    score -= diff_risk_penalty
    score -= negative_memory_penalty
    return round(score, 3)


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        data = dict(candidate)
        data["score"] = candidate_score(data)
        scored.append(data)
    return sorted(scored, key=lambda item: item.get("score", 0.0), reverse=True)[0]


def action_ids_from_selected(selected_actions: list[SelectedRepairAction]) -> list[str]:
    return [action.action_id for action in selected_actions]
