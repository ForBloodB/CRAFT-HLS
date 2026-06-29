from pathlib import Path

from ccd_hls_agent.deterministic_repair import apply_deterministic_repair
from ccd_hls_agent.failure_analysis import build_failure_capsule
from ccd_hls_agent.hls_backends import ToolResult
from ccd_hls_agent.local_memory import HLSLocalMemory, error_signature_from_capsule, render_memory_capsule
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
