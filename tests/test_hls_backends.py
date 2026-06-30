from pathlib import Path

from ccd_hls_agent.hls_backends import HLSBackendConfig, MockHLSBackend, ToolResult, VitisUnifiedHLSBackend, discover_tb_data_files


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


def test_vitis_config_prefers_platform_over_part(tmp_path: Path):
    case = tmp_path / "case"
    case.mkdir()
    platform = tmp_path / "kv260.xpfm"
    platform.write_text("platform\n")
    (case / "kernel.cpp").write_text("void kernel(){}\n")

    backend = VitisUnifiedHLSBackend(HLSBackendConfig(part="xczu9eg-ffvb1156-2-e", platform=str(platform)))
    cfg = backend._write_config(
        tmp_path / "build",
        [case / "kernel.cpp"],
        {"top_function": "kernel"},
        include_tb=False,
    )
    text = cfg.read_text()

    assert f"platform={platform.resolve()}" in text
    assert "part=xczu9eg-ffvb1156-2-e" not in text


def test_vitis_backend_infers_tool_env_from_platform(tmp_path: Path, monkeypatch):
    install = tmp_path / "opt" / "2025.2" / "Vitis"
    bin_dir = install / "bin"
    platform = install / "base_platforms" / "xilinx_kv260" / "xilinx_kv260.xpfm"
    bin_dir.mkdir(parents=True)
    platform.parent.mkdir(parents=True)
    platform.write_text("platform\n")
    vitis_run = bin_dir / "vitis-run"
    vpp = bin_dir / "v++"
    vitis_run.write_text("#!/bin/sh\n")
    vpp.write_text("#!/bin/sh\n")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    backend = VitisUnifiedHLSBackend(HLSBackendConfig(platform=str(platform)))

    assert backend._vitis_run() == vitis_run.resolve()
    assert backend._vpp() == vpp.resolve()
    env_path = backend._tool_env()["PATH"].split(":")
    assert str(bin_dir.resolve()) in env_path


def test_vitis_csim_distinguishes_compile_from_testbench_failure(tmp_path: Path, monkeypatch):
    case = tmp_path / "case"
    case.mkdir()
    (case / "kernel.cpp").write_text("void kernel(){}\n")
    (case / "kernel.h").write_text("void kernel();\n")
    (case / "kernel_tb.cpp").write_text("int main(){return 1;}\n")

    backend = VitisUnifiedHLSBackend()
    monkeypatch.setattr(backend, "_vitis_run", lambda: Path("/bin/true"))

    def fake_run(*args, **kwargs):
        return ToolResult(
            status="failed",
            return_code=1,
            stdout="INFO: [SIM 211-2] *************** CSIM start ***************\nGenerating csim.exe\nMismatch at out[0]\n",
        )

    monkeypatch.setattr(backend, "_run", fake_run)

    result = backend.run_csim(tmp_path / "build", [case / "kernel.cpp", case / "kernel.h", case / "kernel_tb.cpp"], {})

    assert result.metrics["can_compile"] is True
    assert result.metrics["can_pass_testbench"] is False
    assert result.metrics["csim_passed"] is False
