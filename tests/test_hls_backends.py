from pathlib import Path

from ccd_hls_agent.hls_backends import MockHLSBackend, VitisUnifiedHLSBackend, discover_tb_data_files


def test_mock_backend(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "top.txt").write_text("kernel\n")
    (case / "kernel.cpp").write_text("void kernel(){}\n")

    backend = MockHLSBackend()
    assert backend.check_health()["available"]
    info = backend.load_benchmark_case(case)
    assert info["top_function"] == "kernel"

    result = backend.run_synth(tmp_path, [case / "kernel.cpp"], {})
    assert result.return_code == 0
    assert result.metrics["synth_passed"] is True

    cosim = backend.run_cosim(tmp_path, [case / "kernel.cpp"], {})
    assert cosim.return_code == 0
    assert cosim.metrics["cosim_passed"] is True


def test_discover_tb_data_files_from_hls_eval_config(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "hls_eval_config.toml").write_text('tb_data = ["input.data", "check.data"]\n')
    (case / "kernel.cpp").write_text("void kernel(){}\n")
    (case / "kernel.h").write_text("void kernel();\n")
    (case / "kernel_tb.cpp").write_text("int main(){return 0;}\n")
    (case / "input.data").write_text("in\n")
    (case / "check.data").write_text("ok\n")
    (case / "kernel_description.md").write_text("desc\n")

    files = discover_tb_data_files([case / "kernel.cpp", case / "kernel.h", case / "kernel_tb.cpp"])

    assert [p.name for p in files] == ["check.data", "input.data"]


def test_vitis_config_includes_tb_data_files(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "hls_eval_config.toml").write_text('tb_data = ["input.data", "check.data"]\n')
    (case / "kernel.cpp").write_text("void kernel(){}\n")
    (case / "kernel.h").write_text("void kernel();\n")
    (case / "kernel_tb.cpp").write_text("int main(){return 0;}\n")
    (case / "input.data").write_text("in\n")
    (case / "check.data").write_text("ok\n")

    backend = VitisUnifiedHLSBackend()
    cfg = backend._write_config(
        tmp_path / "build",
        [case / "kernel.cpp", case / "kernel.h", case / "kernel_tb.cpp"],
        {"top_function": "kernel"},
        include_tb=True,
    )
    text = cfg.read_text()

    assert f"tb.file={(case / 'input.data').resolve()}" in text
    assert f"tb.file={(case / 'check.data').resolve()}" in text
