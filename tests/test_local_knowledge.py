from pathlib import Path

from ccd_hls_agent.deterministic_repair import apply_deterministic_repair
from ccd_hls_agent.failure_analysis import build_failure_capsule
from ccd_hls_agent.hls_backends import ToolResult
from ccd_hls_agent.local_memory import HLSLocalMemory, error_signature_from_capsule, render_memory_capsule
from ccd_hls_agent.repair_actions import (
    ACTION_ADD_MISSING_INCLUDE,
    ACTION_FIX_ARRAY_SHAPE,
    ACTION_LLM_SEMANTIC_REPAIR,
    ACTION_REWRITE_INDEX_ORDER,
    build_diagnosis_assertion,
    select_best_candidate,
    select_repair_actions,
)
from ccd_hls_agent.skills import route_skills
from ccd_hls_agent.task_modes import TaskMode


def make_case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    case.mkdir()
    (case / "top.txt").write_text("mix_cols\n")
    (case / "mix_cols.h").write_text("#include <stdint.h>\nvoid mix_cols(uint8_t state[4][4]);\n")
    (case / "mix_cols_tb.cpp").write_text("int main(){ return 0; }\n")
    (case / "mix_cols.cpp").write_text(
        '#include "mix_cols.h"\n'
        "static uint8_t xtime(uint8_t x){ return x; }\n"
        "void mix_cols(uint8_t state[4][4]){\n"
        "  for(int i=0;i<4;i++){\n"
        "    uint8_t t = state[i][0] ^ state[i][1];\n"
        "    state[i][0] ^= xtime(t);\n"
        "  }\n"
        "}\n"
    )
    return case


def test_failure_taxonomy_classifies_array_dimension_error():
    tool = ToolResult(
        status="failed",
        return_code=1,
        stdout="",
        stderr="mix_cols.cpp:10: error: invalid operands to binary expression ('uint8_t[4]' and 'uint8_t[4]')\n",
        metrics={"can_compile": False, "can_pass_testbench": False},
        command="csim",
        duration_ms=1,
    )

    capsule = build_failure_capsule("CSIM", tool, token_budget=300)

    assert capsule["failure_class"] == "array_dimension_type_error"
    assert capsule["recommended_policy"] == "run_array_dimension_static_repair_then_llm_if_needed"


def test_skill_router_is_failure_conditioned():
    generation_skills, _ = route_skills(task_mode=TaskMode.GENERATE, selected_atoms=[])
    compile_skills, _ = route_skills(
        task_mode=TaskMode.REPAIR_COMPILE,
        failure_capsule={
            "failure_class": "array_dimension_type_error",
            "failure_type": "compile_error",
            "key_errors": ["invalid operands to binary expression ('uint8_t[4]' and 'uint8_t[4]')"],
        },
    )

    assert "hls_signature_repair" not in [skill.skill_id for skill in generation_skills]
    assert "hls_static_bounds" in [skill.skill_id for skill in compile_skills]


def test_deterministic_repair_rewrites_aes_state_indexing(tmp_path: Path):
    case = make_case(tmp_path)
    capsule = {
        "failure_class": "array_dimension_type_error",
        "signal_lines": ["error: invalid operands to binary expression ('uint8_t[4]' and 'uint8_t[4]')"],
        "key_errors": [],
    }

    result = apply_deterministic_repair(
        case_dir=case,
        kernel=case / "mix_cols.cpp",
        header=case / "mix_cols.h",
        tb=case / "mix_cols_tb.cpp",
        failure_capsule=capsule,
    )

    assert result.applied
    text = (case / "mix_cols.cpp").read_text()
    assert "state[0][i]" in text
    assert "state[i][0]" not in text


def test_local_memory_round_trip(tmp_path: Path):
    db = tmp_path / "memory.sqlite"
    memory = HLSLocalMemory(db)
    capsule = {
        "failure_class": "array_dimension_type_error",
        "signal_lines": ["invalid operands to binary expression uint8_t[4]"],
        "key_errors": [],
    }

    memory.add_event(
        case_name="mix_columns",
        case_family="aes",
        failure_class=capsule["failure_class"],
        error_signature=error_signature_from_capsule(capsule),
        attempted_fix="rewrite state[i][col] to state[col][i]",
        tool_result_before={"can_compile": False},
        tool_result_after={"can_compile": True},
        verified=True,
        artifact_uri="run",
    )
    hits = memory.search(case_family="aes", failure_class="array_dimension_type_error", error_signature="uint8_t[4]")

    assert hits
    assert hits[0].verified
    assert "state" in render_memory_capsule(hits)


def test_diagnosis_assertion_maps_array_error_to_actions():
    capsule = {
        "failure_class": "array_dimension_type_error",
        "signal_lines": ["error: invalid operands to binary expression ('uint8_t[4]' and 'uint8_t[4]')"],
        "key_errors": [],
        "root_cause_hints": ["array misuse"],
    }

    assertion = build_diagnosis_assertion(
        case_name="mix_columns",
        stage="CSIM",
        task_mode="repair_compile",
        failure_capsule=capsule,
        attempt=1,
    )

    assert assertion.failure_class == "array_dimension_type_error"
    assert ACTION_FIX_ARRAY_SHAPE in assertion.recommended_actions
    assert ACTION_REWRITE_INDEX_ORDER in assertion.recommended_actions
    assert "ADD_ARRAY_PARTITION" in assertion.blocked_actions


def test_action_memory_positive_and_negative_adjust_ranking(tmp_path: Path):
    memory = HLSLocalMemory(tmp_path / "memory.sqlite")
    capsule = {
        "failure_class": "signature_or_top_mismatch",
        "signal_lines": ["error: undefined reference to top_kernel"],
        "key_errors": [],
    }
    assertion = build_diagnosis_assertion(
        case_name="demo",
        stage="CSIM",
        task_mode="repair_compile",
        failure_capsule=capsule,
        attempt=1,
    )
    signature = error_signature_from_capsule(capsule)
    memory.add_action_event(
        case_name="demo",
        case_family="unit",
        diagnosis_claim=assertion.claim,
        failure_class=assertion.failure_class,
        error_signature=signature,
        action_id=ACTION_ADD_MISSING_INCLUDE,
        action_params={},
        polarity="positive",
        before_flags={"can_compile": False},
        after_flags={"can_compile": True},
        verified=True,
        reuse_conditions={},
        anti_reuse_conditions={},
        artifact_uri="run",
    )
    memory.add_action_event(
        case_name="demo",
        case_family="unit",
        diagnosis_claim=assertion.claim,
        failure_class=assertion.failure_class,
        error_signature=signature,
        action_id="ALIGN_SIGNATURE",
        action_params={},
        polarity="negative",
        before_flags={"can_compile": False},
        after_flags={"can_compile": False},
        verified=False,
        reuse_conditions={},
        anti_reuse_conditions={},
        artifact_uri="run",
    )

    hits = memory.search_action_memory(
        case_family="unit",
        failure_class=assertion.failure_class,
        error_signature=signature,
        action_ids=assertion.recommended_actions,
    )
    selected = select_repair_actions(assertion, action_memory_hits=hits, limit=2)

    assert selected[0].action_id == ACTION_ADD_MISSING_INCLUDE
    assert selected[-1].action_id == "ALIGN_SIGNATURE"


def test_candidate_rerank_prefers_synth_pass():
    best = select_best_candidate(
        [
            {"candidate_id": 0, "flags": {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False}},
            {"candidate_id": 1, "flags": {"can_compile": True, "can_pass_testbench": True, "can_synthesize": False}},
            {"candidate_id": 2, "flags": {"can_compile": True, "can_pass_testbench": True, "can_synthesize": True}},
        ]
    )

    assert best is not None
    assert best["candidate_id"] == 2


def test_default_unknown_action_is_semantic_repair():
    assertion = build_diagnosis_assertion(
        case_name="demo",
        stage="CSIM",
        task_mode="repair_csim",
        failure_capsule={"failure_class": "unknown_hls_failure", "signal_lines": ["ERROR: unknown"]},
        attempt=1,
    )

    assert select_repair_actions(assertion, limit=1)[0].action_id == ACTION_LLM_SEMANTIC_REPAIR
