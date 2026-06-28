import os
from pathlib import Path

from ccd_hls_agent import agent_tui as tui


def make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "result.json").write_text(
        """{
  "method": "ccd_hls_loop",
  "case_name": "demo",
  "can_parse": true,
  "can_compile": true,
  "can_pass_testbench": false,
  "can_synthesize": false,
  "prompt_tokens": 11,
  "completion_tokens": 7,
  "total_tokens": 18,
  "tool_calls": 1,
  "metrics": {
    "llm_calls_used": 1,
    "max_llm_calls": 2,
    "repair_rounds": 0,
    "terminal_stage": "CSIM",
    "stopped_reason": "test"
  }
}
""",
        encoding="utf-8",
    )
    (run / "stage_records.json").write_text(
        """[
  {
    "stage": "GENERATION",
    "status": "completed",
    "message": "generated",
    "metrics": {"prompt_tokens": 11},
    "artifacts": {"response": "response.txt"}
  }
]
""",
        encoding="utf-8",
    )
    (run / "response.txt").write_text("ok\n", encoding="utf-8")
    return run


def test_dashboard_helpers_load_without_textual(tmp_path: Path):
    run = make_run(tmp_path)

    data = tui.load_dashboard(run)
    rows = tui.stage_rows(data.stages)
    artifacts = tui.artifact_rows(tui.stage_artifacts(data.stages[1], data.run_dir, data.calls))

    assert data.result["case_name"] == "demo"
    assert rows[0][1] == "OK"
    assert rows[1][2] == "GENERATION"
    assert artifacts == [("1", "response", str(run / "response.txt"))]


def test_result_summary_contains_agent_metrics(tmp_path: Path):
    data = tui.load_dashboard(make_run(tmp_path))

    summary = tui.result_summary(data)

    assert "ccd_hls_loop" in summary
    assert "llm calls : 1 / 2" in summary
    assert "stopped   : test" in summary


def test_truncate_for_tui_keeps_head_and_tail():
    text = "a" * 20 + "b" * 20 + "c" * 20

    out = tui.truncate_for_tui(text, limit=20)

    assert out.startswith("a" * 10)
    assert out.endswith("c" * 10)
    assert "truncated" in out


def test_recent_runs_are_sorted_and_ignore_empty_dirs(tmp_path: Path):
    root = tmp_path / "experiments"
    old = make_run(root / "old_parent")
    new = make_run(root / "new_parent")
    (root / "empty").mkdir(parents=True)
    old_time = 1000
    new_time = 2000
    for marker in ["result.json", "stage_records.json"]:
        os.utime(old / marker, (old_time, old_time))
        os.utime(new / marker, (new_time, new_time))

    runs = tui.recent_runs(root)

    assert [run.run_dir for run in runs[:2]] == [new, old]


def test_launch_model_has_help_when_no_runs(tmp_path: Path):
    launch = tui.load_launch(tmp_path / "missing")

    assert launch.runs == []
    assert "HLS-agent run" in launch.help_text


def test_parser_keeps_view_and_run_subcommands():
    parser = tui.build_parser()

    view = parser.parse_args(["view", "some/run", "--snapshot"])
    run = parser.parse_args(["run", "--max-llm-calls", "3", "--llm-call-budget", "2", "--csim-budget", "0", "--no-view"])
    recent = parser.parse_args(["recent", "--limit", "5", "--snapshot"])
    doctor = parser.parse_args(["doctor"])
    contract = parser.parse_args(["contract", "review", "some/contract"])
    launch = parser.parse_args([])

    assert view.command == "view"
    assert view.snapshot is True
    assert run.command == "run"
    assert run.max_llm_calls == 3
    assert run.llm_call_budget == 2
    assert run.csim_budget == 0
    assert recent.command == "recent"
    assert recent.limit == 5
    assert doctor.command == "doctor"
    assert contract.command == "contract"
    assert contract.contract_command == "review"
    assert contract.contract_dir == Path("some/contract")
    assert launch.command is None


def test_doctor_detects_missing_model_key_and_vitis_tools(tmp_path: Path, monkeypatch):
    model_config = tmp_path / "model.json"
    model_config.write_text(
        """{
  "provider_type": "cloud_openai",
  "api_key_env": "MISSING_KEY",
  "model": "demo"
}
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("MISSING_KEY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no_tools"))

    checks = tui.doctor_checks(
        model_config=model_config,
        hls_eval_root=tmp_path / "hls-eval",
    )
    by_name = {check.name: check for check in checks}

    assert by_name["model api key"].status == "FAIL"
    assert by_name["v++"].status == "FAIL"
    assert by_name["vitis-run"].status == "FAIL"
    assert by_name["HLS-Eval root"].status == "FAIL"


def test_doctor_accepts_inline_local_model_key(tmp_path: Path, monkeypatch):
    model_config = tmp_path / "model.local.json"
    model_config.write_text(
        """{
  "provider_type": "cloud_openai",
  "api_key": "dummy-local-key",
  "model": "demo"
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", str(tmp_path / "no_tools"))

    checks = tui.doctor_checks(model_config=model_config, hls_eval_root=tmp_path / "hls-eval")
    by_name = {check.name: check for check in checks}

    assert by_name["model api key"].status == "OK"
