from pathlib import Path

from ccd_hls_agent.budget import BudgetLedger
from ccd_hls_agent.skills import route_skills
from ccd_hls_agent.task_modes import TaskMode, classify_task_mode
from ccd_hls_agent.token_report import build_token_report


class DummyResult:
    method = "ccd_hls_loop"
    case_name = "demo"
    sample_idx = 0
    prompt_tokens = 30
    completion_tokens = 12
    total_tokens = 42
    can_synthesize = True
    can_compile = True
    can_pass_testbench = True
    metrics = {
        "task_mode": "generate",
        "stage_records": [
            {"stage": "GENERATION", "metrics": {"prompt_tokens": 10, "completion_tokens": 5}},
            {"stage": "CSIM_REPAIR", "metrics": {"prompt_tokens": 20, "completion_tokens": 7}},
            {"stage": "CSIM", "metrics": {"attempt": 1}},
        ],
    }


def test_token_report_groups_llm_and_tool_stages():
    report = build_token_report(DummyResult())

    assert report["total_tokens"] == 42
    assert report["tokens_by_stage"]["GENERATION"] == 15
    assert report["tokens_by_stage"]["CSIM_REPAIR"] == 27
    assert report["llm_calls_by_stage"]["CSIM_REPAIR"] == 1
    assert report["tool_calls_by_stage"]["CSIM_REPAIR"] == 1
    assert report["tokens_per_success"] == 42


def test_token_report_counts_failed_llm_prompt_estimate():
    class FailedResult:
        method = "ccd_hls_loop"
        case_name = "demo"
        sample_idx = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        can_synthesize = False
        can_compile = False
        can_pass_testbench = False
        metrics = {
            "task_mode": "generate",
            "stage_records": [
                {"stage": "GENERATION", "status": "failed", "metrics": {"prompt_tokens_est": 123, "llm_calls_used": 1}},
            ],
        }

    report = build_token_report(FailedResult())

    assert report["total_prompt_tokens"] == 123
    assert report["total_tokens"] == 123
    assert report["tokens_by_stage"]["GENERATION"] == 123
    assert report["llm_calls_by_stage"]["GENERATION"] == 1
    assert report["tokens_per_success"] is None


def test_budget_ledger_consumes_and_rejects_budget():
    ledger = BudgetLedger.from_limits(llm_calls=1, csim_calls=0, synth_calls=1, cosim_calls=0)

    assert ledger.consume("llm_calls", stage="GENERATION", label="first")
    assert not ledger.consume("llm_calls", stage="CSIM_REPAIR", label="second")
    assert not ledger.consume("csim_calls", stage="CSIM", label="attempt_1")
    assert ledger.summary()["llm_calls"]["remaining"] == 0
    assert ledger.events[-1]["reason"] == "budget_exhausted_csim"


def test_task_mode_classifier_covers_repair_modes():
    assert classify_task_mode(generation_stub=True) == TaskMode.GENERATE
    assert classify_task_mode(generation_stub=False, csim_flags={"can_compile": False}) == TaskMode.REPAIR_COMPILE
    assert classify_task_mode(generation_stub=False, csim_flags={"can_compile": True, "can_pass_testbench": False}) == TaskMode.REPAIR_CSIM
    assert (
        classify_task_mode(
            generation_stub=False,
            csim_flags={"can_compile": True, "can_pass_testbench": True},
            synth_passed=False,
        )
        == TaskMode.REPAIR_SYNTH
    )
    assert (
        classify_task_mode(
            generation_stub=False,
            csim_flags={"can_compile": True, "can_pass_testbench": True},
            synth_passed=True,
        )
        == TaskMode.OPTIMIZE_PPA
    )


def test_skill_router_hits_expected_error_classes():
    signature_skills, _ = route_skills(
        task_mode=TaskMode.REPAIR_COMPILE,
        failure_capsule={"failure_type": "signature_or_compile_error", "key_errors": ["error: no matching function for call"]},
    )
    data_skills, _ = route_skills(
        task_mode=TaskMode.REPAIR_CSIM,
        failure_capsule={"failure_type": "runtime_data_file_missing", "key_errors": ["Couldn't open input data file"]},
    )
    fifo_skills, _ = route_skills(
        task_mode=TaskMode.REPAIR_COSIM,
        failure_capsule={"failure_type": "synthesis_error", "key_errors": ["dataflow deadlock due to FIFO depth"]},
    )

    assert signature_skills[0].skill_id == "hls_signature_repair"
    assert data_skills[0].skill_id == "hls_data_file_runtime"
    assert fifo_skills[0].skill_id == "hls_dataflow_fifo"
