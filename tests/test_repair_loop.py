import asyncio
from pathlib import Path

import scripts.run_hls_eval_benchmark as runner
from ccd_hls_agent.failure_analysis import repeated_failure_early_stop
from ccd_hls_agent.hls_backends import ToolResult
from ccd_hls_agent.model_clients import ModelResult
from ccd_hls_agent.schemas import ModelConfig


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: str, system_prompt: str | None = None) -> ModelResult:
        self.calls += 1
        return ModelResult(
            content='<OUTPUT_CODE name="kernel.cpp">\n#include "kernel.h"\nvoid kernel() {}\n</OUTPUT_CODE>',
            prompt_tokens=10,
            completion_tokens=5,
            duration_ms=1,
            raw={},
        )


def make_generation_case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    case.mkdir()
    (case / "top.txt").write_text("kernel\n")
    (case / "kernel_description.md").write_text("Implement kernel.\n")
    (case / "hls_eval_config.toml").write_text("tags = ['unit']\n")
    (case / "kernel.h").write_text("void kernel();\n")
    (case / "kernel.cpp").write_text("void kernel() {}\n")
    (case / "kernel_tb.cpp").write_text("int main(){ kernel(); return 0; }\n")
    return case


def test_failure_capsule_extracts_signal_lines():
    tool_result = ToolResult(
        status="failed",
        return_code=1,
        stdout="running\nFAIL: mismatch at index 3\n",
        stderr="kernel.cpp:7: error: no matching function for call\n",
        metrics={"can_compile": False},
        command="fake_csim",
        duration_ms=12,
    )

    capsule = runner.build_failure_capsule("CSIM", tool_result, token_budget=200)

    assert capsule["stage"] == "CSIM"
    assert capsule["failure_type"] == "signature_or_compile_error"
    assert any("no matching function" in line for line in capsule["signal_lines"])


def test_failure_capsule_prioritizes_runtime_error_window():
    tool_result = ToolResult(
        status="failed",
        return_code=1,
        stdout=(
            "INFO: [HLS 200-2191] compiler setup\n"
            "Compiling kernel.cpp\n"
            "csim.exe: kernel_tb.cpp:27: int main(): Assertion `in_fd > 0 && \"Couldn't open input data file\"' failed.\n"
            "@E Simulation failed with unknown error: child killed: SIGABRT\n"
            "ERROR: [SIM 211-100] CSim failed with errors.\n"
        ),
        stderr="",
        metrics={"can_compile": False},
        command="fake_csim",
        duration_ms=12,
    )

    capsule = runner.build_failure_capsule("CSIM", tool_result, token_budget=240)

    assert capsule["failure_type"] == "runtime_data_file_missing"
    assert any("Couldn't open input data file" in window for window in capsule["key_errors"])
    assert not any("HLS 200-2191" == line for line in capsule["signal_lines"])


def test_stage_record_schema():
    records = []

    runner.add_stage_record(records, stage="CSIM", status="failed", message="compile failed", metrics={"x": 1})

    assert records[0]["stage"] == "CSIM"
    assert records[0]["status"] == "failed"
    assert records[0]["metrics"]["x"] == 1
    assert records[0]["created_at"]


def test_stage_runtime_files_copies_data_files(tmp_path: Path):
    case = make_generation_case(tmp_path)
    (case / "input.data").write_text("1 2 3\n")
    (case / "check.data").write_text("ok\n")
    build = tmp_path / "build"

    copied = runner.stage_runtime_files(case, build)

    assert (build / "input.data").read_text() == "1 2 3\n"
    assert (build / "check.data").read_text() == "ok\n"
    assert not (build / "kernel.cpp").exists()
    assert len(copied) == 2


def test_repair_prompt_includes_failure_history(tmp_path: Path):
    case = make_generation_case(tmp_path)
    prompt = runner.build_hls_repair_prompt(
        stage="CSIM",
        kernel=case / "kernel.cpp",
        header=case / "kernel.h",
        tb=case / "kernel_tb.cpp",
        description=case / "kernel_description.md",
        failure_capsule={"stage": "CSIM", "failure_type": "compile_error", "key_errors": ["first error"]},
        failure_history=[
            {"stage": "CSIM", "failure_type": "compile_error", "key_errors": ["old error"]},
            {"stage": "CSIM", "failure_type": "testbench_failure", "key_errors": ["new error"]},
        ],
        attempt=2,
        max_llm_calls=5,
    )

    assert "Failure History From Previous Loops" in prompt
    assert "old error" in prompt
    assert "new error" in prompt


def test_repeated_failure_early_stop_detects_same_error():
    capsules = [
        {
            "failure_type": "signature_or_compile_error",
            "key_errors": ["kernel.cpp:7: error: no matching function for call to foo(int)"],
        },
        {
            "failure_type": "signature_or_compile_error",
            "key_errors": ["../../kernel.cpp:11: error: no matching function for call to foo(int)"],
        },
    ]

    should_stop, score = repeated_failure_early_stop(capsules, threshold=0.80)

    assert should_stop
    assert score >= 0.80


def test_repeated_failure_early_stop_ignores_different_type():
    capsules = [
        {"failure_type": "signature_or_compile_error", "key_errors": ["no matching function"]},
        {"failure_type": "testbench_failure", "key_errors": ["mismatch at index 3"]},
    ]

    should_stop, score = repeated_failure_early_stop(capsules, threshold=0.80)

    assert not should_stop
    assert score == 0.0


def test_csim_failure_with_exhausted_llm_budget_skips_synth(tmp_path: Path, monkeypatch):
    case = make_generation_case(tmp_path)
    client = FakeClient()

    async def fake_csim_stage(*args, **kwargs):
        tool = ToolResult(
            status="failed",
            return_code=1,
            stdout="",
            stderr="kernel.cpp:3: error: compile failed\n",
            metrics={"can_compile": False, "can_pass_testbench": False},
            command="fake_csim",
            duration_ms=1,
        )
        return (
            {"can_compile": False, "can_pass_testbench": False, "can_synthesize": False},
            {"csim": tool.metrics, "csim_return_code": 1},
            tool,
        )

    async def fail_if_synth_called(*args, **kwargs):
        raise AssertionError("synth should not run after csim failure with exhausted LLM budget")

    monkeypatch.setattr(runner, "build_model_client", lambda model: client)
    monkeypatch.setattr(runner, "run_csim_stage", fake_csim_stage)
    monkeypatch.setattr(runner, "run_synth_stage", fail_if_synth_called)

    result = asyncio.run(
        runner.run_ccd_gen_v2_case(
            "exp_test",
            case,
            tmp_path / "run",
            ModelConfig(),
            "mock",
            None,
            0,
            6000,
            1,
            200,
        )
    )

    assert client.calls == 1
    assert result.can_parse
    assert not result.can_compile
    assert not result.can_synthesize
    assert result.tool_calls == 1
    assert result.metrics["terminal_stage"] == "CSIM"
    assert result.metrics["stopped_reason"] == "max_llm_calls_exhausted_after_csim_failure"
    assert result.metrics["llm_calls_used"] == 1
    assert result.metrics["failure_capsules"]
