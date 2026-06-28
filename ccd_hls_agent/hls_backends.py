from __future__ import annotations

import importlib.util
import os
import signal
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .utils import read_text, which_status, write_text


SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}
CASE_METADATA_FILES = {"hls_eval_config.toml", "kernel_description.md", "top.txt"}


def discover_tb_data_files(source_files: list[Path]) -> list[Path]:
    if not source_files:
        return []
    case_dir = source_files[0].parent
    config_path = case_dir / "hls_eval_config.toml"
    if config_path.exists():
        try:
            data = tomllib.loads(read_text(config_path))
        except tomllib.TOMLDecodeError:
            data = {}
        if "tb_data" in data:
            configured = []
            for item in data.get("tb_data", []) or []:
                fp = case_dir / str(item)
                if fp.is_file():
                    configured.append(fp)
            return sorted(configured)
    return sorted(
        item
        for item in case_dir.iterdir()
        if item.is_file()
        and item.suffix not in SOURCE_EXTENSIONS
        and item.name not in CASE_METADATA_FILES
        and not item.name.startswith("kernel_description")
        and not item.name.endswith(".tcl")
    )


@dataclass
class HLSBackendConfig:
    backend: str = "hls_eval"
    hls_eval_root: str | None = None
    vitis_root: str | None = None
    vitis_run_path: str | None = None
    vpp_path: str | None = None
    vitis_hls_path: str | None = None
    vivado_hls_path: str | None = None
    vivado_path: str | None = None
    hls_eval_command: str | None = None
    part: str = "xczu9eg-ffvb1156-2-e"
    clock_period_ns: float = 5.0
    timeout_seconds: float = 360.0


@dataclass
class ToolResult:
    status: str
    return_code: int
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    duration_ms: int = 0


class HLSBackend(Protocol):
    name: str

    def check_health(self) -> dict[str, Any]:
        ...

    def load_benchmark_case(self, path: Path) -> dict[str, Any]:
        ...

    def run_csim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        ...

    def run_synth(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        ...

    def run_cosim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        ...

    def parse_metrics(self, artifacts: dict[str, Path]) -> dict[str, Any]:
        ...


class HLSEvalBackend:
    name = "hls_eval"

    def __init__(self, config: HLSBackendConfig | None = None) -> None:
        self.config = config or HLSBackendConfig()

    def _ensure_hls_eval_importable(self) -> bool:
        if importlib.util.find_spec("hls_eval") is not None:
            return True
        candidates = []
        if self.config.hls_eval_root:
            candidates.append(Path(self.config.hls_eval_root))
        if os.environ.get("HLS_EVAL_ROOT"):
            candidates.append(Path(os.environ["HLS_EVAL_ROOT"]))
        candidates.extend([Path("external/hls-eval"), Path("../external/hls-eval")])
        for root in candidates:
            root = root.expanduser().resolve()
            if (root / "hls_eval").is_dir():
                sys.path.insert(0, str(root))
                return importlib.util.find_spec("hls_eval") is not None
        return False

    def check_health(self) -> dict[str, Any]:
        hls_eval_ok = self._ensure_hls_eval_importable()
        vitis_bin = self._find_vitis_bin()
        return {
            "available": hls_eval_ok and vitis_bin is not None,
            "hls_eval_available": hls_eval_ok,
            "vitis_hls_bin": str(vitis_bin) if vitis_bin else None,
            "detail": (
                "HLS-Eval 和 Vitis HLS 可用"
                if hls_eval_ok and vitis_bin
                else "需要安装 HLS-Eval 并配置 vitis_hls"
            ),
        }

    def _find_vitis_bin(self) -> Path | None:
        if self.config.vitis_hls_path:
            p = Path(self.config.vitis_hls_path)
            if p.name == "vitis_hls" and p.exists():
                return p
            candidate = p / "bin" / "vitis_hls"
            if candidate.exists():
                return candidate
        found = shutil.which("vitis_hls")
        return Path(found) if found else None

    def _find_vitis_root(self) -> Path | None:
        bin_path = self._find_vitis_bin()
        if bin_path is None:
            return None
        return bin_path.parent.parent

    def load_benchmark_case(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        if not self._ensure_hls_eval_importable():
            raise RuntimeError("HLS-Eval 未安装。请运行: pip install git+https://github.com/sharc-lab/hls-eval.git")
        from hls_eval.data import BenchmarkCase

        case = BenchmarkCase(path)
        return {
            "name": case.name,
            "top_function": case.top_fn,
            "kernel": str(case.kernel_fp),
            "testbench": str(case.tb_file),
            "source_files": [str(p) for p in case.source_files],
            "tags": case.tags_all,
        }

    def run_csim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        vitis_root = self._find_vitis_root()
        if vitis_root is None:
            return ToolResult(
                status="failed",
                return_code=127,
                stderr="vitis_hls not found; configure HLSBackendConfig.vitis_hls_path or PATH.",
                command="vitis_hls csim",
            )
        try:
            self._ensure_hls_eval_importable()
            from hls_eval.tools import VitisHLSCSimTool
        except Exception as exc:
            return ToolResult(status="failed", return_code=127, stderr=str(exc), command="import hls_eval.tools")

        t0 = time.monotonic()
        tool = VitisHLSCSimTool(vitis_root)
        tb_data_files = discover_tb_data_files(source_files)
        compile_data, run_data = tool.run(
            build_dir=workdir,
            source_files=source_files,
            aux_files=tb_data_files,
            hls_top_function=config.get("top_function"),
            hls_fpga_part=config.get("part", self.config.part),
            hls_clock_period_ns=config.get("clock_period_ns", self.config.clock_period_ns),
            timeout=config.get("timeout_seconds", self.config.timeout_seconds),
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        stdout = compile_data.data_execution.stdout
        stderr = compile_data.data_execution.stderr
        return_code = compile_data.data_execution.return_code
        status = "completed" if return_code == 0 else "failed"
        metrics = {
            "can_compile": compile_data.data_execution.return_code == 0,
            "compile_return_code": compile_data.data_execution.return_code,
            "compile_timeout": compile_data.data_execution.timeout,
            "tb_data_files": [str(p) for p in tb_data_files],
        }
        if run_data is not None:
            stdout += "\n" + run_data.data_execution.stdout
            stderr += "\n" + run_data.data_execution.stderr
            return_code = run_data.data_execution.return_code
            status = "completed" if return_code == 0 else "failed"
            metrics.update(
                {
                    "can_pass_testbench": run_data.data_execution.return_code == 0,
                    "csim_return_code": run_data.data_execution.return_code,
                    "csim_timeout": run_data.data_execution.timeout,
                }
            )
        else:
            metrics.update({"can_pass_testbench": False, "csim_return_code": None})
        return ToolResult(
            status=status,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            metrics={**metrics, "csim_passed": return_code == 0},
            command="VitisHLSCSimTool.run",
            duration_ms=duration_ms,
        )

    def run_synth(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        vitis_root = self._find_vitis_root()
        if vitis_root is None:
            return ToolResult(
                status="failed",
                return_code=127,
                stderr="vitis_hls not found; configure HLSBackendConfig.vitis_hls_path or PATH.",
                command="vitis_hls synth",
            )
        try:
            self._ensure_hls_eval_importable()
            from hls_eval.tools import VitisHLSSynthTool
        except Exception as exc:
            return ToolResult(status="failed", return_code=127, stderr=str(exc), command="import hls_eval.tools")

        t0 = time.monotonic()
        tool = VitisHLSSynthTool(vitis_root)
        result = tool.run(
            build_dir=workdir,
            source_files=source_files,
            hls_top_function=config.get("top_function"),
            hls_fpga_part=config.get("part", self.config.part),
            hls_clock_period_ns=config.get("clock_period_ns", self.config.clock_period_ns),
            timeout=config.get("timeout_seconds", self.config.timeout_seconds),
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        metrics = {"synth_passed": result.data_execution.return_code == 0}
        if result.data_tool is not None:
            metrics.update(getattr(result.data_tool, "__dict__", {}))
        return ToolResult(
            status="completed" if result.data_execution.return_code == 0 else "failed",
            return_code=result.data_execution.return_code,
            stdout=result.data_execution.stdout,
            stderr=result.data_execution.stderr,
            metrics=metrics,
            command="VitisHLSSynthTool.run",
            duration_ms=duration_ms,
        )

    def run_cosim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        return ToolResult(
            status="failed",
            return_code=127,
            stderr="HLS-Eval backend does not expose RTL cosim in this project; use --hls-backend vitis.",
            command="hls_eval cosim",
            metrics={"can_cosim": False, "unsupported": True},
        )

    def parse_metrics(self, artifacts: dict[str, Path]) -> dict[str, Any]:
        csynth = artifacts.get("csynth_xml")
        if csynth is None or not csynth.exists():
            return {}
        try:
            from hls_eval.vhls_report import DesignHLSSynthData

            data = DesignHLSSynthData.parse_from_synth_report_file(csynth)
            return getattr(data, "__dict__", {})
        except Exception as exc:
            return {"parse_error": str(exc)}


class CommandHLSBackend:
    name = "command"

    def __init__(self, config: HLSBackendConfig | None = None) -> None:
        self.config = config or HLSBackendConfig()

    def check_health(self) -> dict[str, Any]:
        commands = [which_status("vitis_hls"), which_status("vivado_hls"), which_status("vivado")]
        return {
            "available": any(item["available"] for item in commands),
            "commands": commands,
            "detail": "至少一个 Xilinx/Vivado 命令可用" if any(item["available"] for item in commands) else "未找到 vitis_hls/vivado_hls/vivado",
        }

    def load_benchmark_case(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        top = read_text(path / "top.txt").strip()
        source_files = [str(p) for p in path.glob("*") if p.suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}]
        return {"name": path.name, "top_function": top, "source_files": source_files, "tags": []}

    def _run_command(self, workdir: Path, command: str, timeout: float) -> ToolResult:
        t0 = time.monotonic()
        proc = subprocess.run(
            command,
            cwd=workdir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        return ToolResult(
            status="completed" if proc.returncode == 0 else "failed",
            return_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            command=command,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    def run_csim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        command = config.get("csim_command") or self.config.hls_eval_command
        if not command:
            return ToolResult(status="failed", return_code=127, stderr="No csim command configured.", command="")
        return self._run_command(workdir, command, config.get("timeout_seconds", self.config.timeout_seconds))

    def run_synth(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        command = config.get("synth_command") or self.config.hls_eval_command
        if not command:
            return ToolResult(status="failed", return_code=127, stderr="No synth command configured.", command="")
        return self._run_command(workdir, command, config.get("timeout_seconds", self.config.timeout_seconds))

    def run_cosim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        command = config.get("cosim_command") or self.config.hls_eval_command
        if not command:
            return ToolResult(status="failed", return_code=127, stderr="No cosim command configured.", command="")
        return self._run_command(workdir, command, config.get("timeout_seconds", self.config.timeout_seconds))

    def parse_metrics(self, artifacts: dict[str, Path]) -> dict[str, Any]:
        return {}


class VitisUnifiedHLSBackend:
    name = "vitis"

    def __init__(self, config: HLSBackendConfig | None = None) -> None:
        self.config = config or HLSBackendConfig()

    def _find_bin(self, name: str, configured: str | None = None) -> Path | None:
        if configured:
            p = Path(configured).expanduser()
            if p.exists():
                return p.resolve()
        root_candidates = []
        if self.config.vitis_root:
            root_candidates.append(Path(self.config.vitis_root))
        if os.environ.get("XILINX_VITIS"):
            root_candidates.append(Path(os.environ["XILINX_VITIS"]))
        root_candidates.append(Path("/data/tools/Xilinx/2025.2.1/Vitis"))
        for root in root_candidates:
            candidate = root.expanduser() / "bin" / name
            if candidate.exists():
                return candidate.resolve()
        found = shutil.which(name)
        return Path(found).resolve() if found else None

    def _vpp(self) -> Path | None:
        return self._find_bin("v++", self.config.vpp_path)

    def _vitis_run(self) -> Path | None:
        return self._find_bin("vitis-run", self.config.vitis_run_path)

    def check_health(self) -> dict[str, Any]:
        vpp = self._vpp()
        vitis_run = self._vitis_run()
        return {
            "available": vpp is not None and vitis_run is not None,
            "vpp": str(vpp) if vpp else None,
            "vitis_run": str(vitis_run) if vitis_run else None,
            "detail": "Vitis unified HLS CLI 可用" if vpp and vitis_run else "需要 v++ 和 vitis-run",
        }

    def load_benchmark_case(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        top = read_text(path / "top.txt").strip()
        source_files = [str(p) for p in path.glob("*") if p.suffix in {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}]
        tags = []
        config_text = read_text(path / "hls_eval_config.toml")
        if "tags" in config_text:
            tags = [item.strip().strip("'\"") for item in config_text.split("[", 1)[-1].split("]", 1)[0].split(",") if item.strip()]
        return {"name": path.name, "top_function": top, "source_files": source_files, "tags": tags}

    def _write_config(self, workdir: Path, source_files: list[Path], config: dict[str, Any], include_tb: bool) -> Path:
        workdir = workdir.resolve()
        cfg = workdir / ("hls_csim.cfg" if include_tb else "hls_synth.cfg")
        top = config.get("top_function")
        part = config.get("part", self.config.part)
        freqhz = config.get("freqhz", "200MHz")
        lines = [
            f"part={part}",
            f"freqhz={freqhz}",
            "",
            "[hls]",
        ]
        if top:
            lines.append(f"syn.top={top}")
        design_files = [p for p in source_files if not p.name.endswith("_tb.cpp")]
        tb_files = [p for p in source_files if p.name.endswith("_tb.cpp")]
        tb_data_files = discover_tb_data_files(source_files) if include_tb else []
        for fp in design_files:
            lines.append(f"syn.file={fp.resolve()}")
        for fp in tb_files:
            lines.append(f"tb.file={fp.resolve()}")
        for fp in tb_data_files:
            lines.append(f"tb.file={fp.resolve()}")
        lines.extend(
            [
                "flow_target=vivado",
                "syn.compile.unsafe_math_optimizations=true",
            ]
        )
        write_text(cfg, "\n".join(lines) + "\n")
        return cfg

    def _run(self, command: list[str], workdir: Path, timeout: float) -> ToolResult:
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                command,
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(2)
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                stdout, stderr = proc.communicate()
                return ToolResult(
                    status="failed",
                    return_code=-1,
                    stdout=stdout or "",
                    stderr=(stderr or "") + "\nTIMEOUT",
                    command=" ".join(map(str, command)),
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    metrics={"timeout": True},
                )
            return ToolResult(
                status="completed" if proc.returncode == 0 else "failed",
                return_code=proc.returncode or 0,
                stdout=stdout or "",
                stderr=stderr or "",
                command=" ".join(map(str, command)),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status="failed",
                return_code=-1,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nTIMEOUT",
                command=" ".join(map(str, command)),
                duration_ms=int((time.monotonic() - t0) * 1000),
                metrics={"timeout": True},
            )

    def run_csim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        vitis_run = self._vitis_run()
        if vitis_run is None:
            return ToolResult(status="failed", return_code=127, stderr="vitis-run not found", command="vitis-run")
        workdir.mkdir(parents=True, exist_ok=True)
        cfg = self._write_config(workdir, source_files, config, include_tb=True)
        result = self._run(
            [str(vitis_run), "--mode", "hls", "--csim", "--config", str(cfg.resolve()), "--work_dir", str((workdir / "component").resolve())],
            workdir,
            config.get("timeout_seconds", self.config.timeout_seconds),
        )
        compiled = result.return_code == 0 or "Generating csim.exe" in result.stdout or "CSim start" in result.stdout
        result.metrics.update(
            {
                "can_compile": compiled,
                "can_pass_testbench": result.return_code == 0,
                "csim_passed": result.return_code == 0,
                "tb_data_files": [str(p) for p in discover_tb_data_files(source_files)],
            }
        )
        return result

    def run_synth(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        vpp = self._vpp()
        if vpp is None:
            return ToolResult(status="failed", return_code=127, stderr="v++ not found", command="v++")
        workdir.mkdir(parents=True, exist_ok=True)
        cfg = self._write_config(workdir, source_files, config, include_tb=False)
        result = self._run(
            [str(vpp), "-c", "--mode", "hls", "--config", str(cfg.resolve()), "--work_dir", str((workdir / "component").resolve())],
            workdir,
            config.get("timeout_seconds", self.config.timeout_seconds),
        )
        metrics = self._parse_reports(workdir)
        result.metrics.update({"synth_passed": result.return_code == 0, "can_synthesize": result.return_code == 0, **metrics})
        return result

    def run_cosim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        vitis_run = self._vitis_run()
        if vitis_run is None:
            return ToolResult(status="failed", return_code=127, stderr="vitis-run not found", command="vitis-run")
        vpp = self._vpp()
        if vpp is None:
            return ToolResult(status="failed", return_code=127, stderr="v++ not found", command="v++")
        workdir.mkdir(parents=True, exist_ok=True)
        component_dir = workdir / "component"
        synth_cfg = self._write_config(workdir, source_files, config, include_tb=False)
        synth_result = self._run(
            [str(vpp), "-c", "--mode", "hls", "--config", str(synth_cfg.resolve()), "--work_dir", str(component_dir.resolve())],
            workdir,
            config.get("timeout_seconds", self.config.timeout_seconds),
        )
        if synth_result.return_code != 0:
            synth_result.metrics.update(
                {
                    "can_cosim": False,
                    "cosim_passed": False,
                    "synth_for_cosim_passed": False,
                    "synth_for_cosim_return_code": synth_result.return_code,
                    "tb_data_files": [str(p) for p in discover_tb_data_files(source_files)],
                }
            )
            return synth_result

        cosim_cfg = self._write_config(workdir, source_files, config, include_tb=True)
        cosim_result = self._run(
            [str(vitis_run), "--mode", "hls", "--cosim", "--config", str(cosim_cfg.resolve()), "--work_dir", str(component_dir.resolve())],
            workdir,
            config.get("timeout_seconds", self.config.timeout_seconds),
        )
        cosim_result.stdout = synth_result.stdout + "\n" + cosim_result.stdout
        cosim_result.stderr = synth_result.stderr + "\n" + cosim_result.stderr
        cosim_result.duration_ms += synth_result.duration_ms
        cosim_result.metrics.update(
            {
                "can_cosim": cosim_result.return_code == 0,
                "cosim_passed": cosim_result.return_code == 0,
                "synth_for_cosim_passed": True,
                "synth_for_cosim_return_code": synth_result.return_code,
                "synth_for_cosim_duration_ms": synth_result.duration_ms,
                "tb_data_files": [str(p) for p in discover_tb_data_files(source_files)],
            }
        )
        return cosim_result

    def _parse_reports(self, workdir: Path) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for report in workdir.rglob("csynth.xml"):
            metrics["csynth_xml"] = str(report)
            try:
                import xml.etree.ElementTree as ET

                root = ET.parse(report).getroot()
                timing = root.find(".//SummaryOfTimingAnalysis")
                latency = root.find(".//SummaryOfOverallLatency")
                resources = root.find(".//AreaEstimates/Resources")
                if timing is not None:
                    est = timing.findtext("EstimatedClockPeriod")
                    if est:
                        metrics["estimated_clock_period"] = float(est)
                if latency is not None:
                    for tag, key in [
                        ("Best-caseLatency", "latency_best_cycles"),
                        ("Average-caseLatency", "latency_average_cycles"),
                        ("Worst-caseLatency", "latency_worst_cycles"),
                    ]:
                        text = latency.findtext(tag)
                        if text and text.isdigit():
                            metrics[key] = int(text)
                if resources is not None:
                    for tag, key in [
                        ("LUT", "resources_lut_used"),
                        ("FF", "resources_ff_used"),
                        ("DSP", "resources_dsp_used"),
                        ("BRAM_18K", "resources_bram_used"),
                        ("URAM", "resources_uram_used"),
                    ]:
                        text = resources.findtext(tag)
                        if text and text.isdigit():
                            metrics[key] = int(text)
            except Exception as exc:
                metrics["report_parse_error"] = str(exc)
            break
        return metrics

    def parse_metrics(self, artifacts: dict[str, Path]) -> dict[str, Any]:
        return self._parse_reports(next(iter(artifacts.values())).parent if artifacts else Path("."))


class MockHLSBackend:
    name = "mock"

    def check_health(self) -> dict[str, Any]:
        return {"available": True, "detail": "Mock backend available for UI/development only."}

    def load_benchmark_case(self, path: Path) -> dict[str, Any]:
        source_files = [str(p) for p in path.glob("*") if p.suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}]
        return {
            "name": path.name,
            "top_function": read_text(path / "top.txt").strip() or None,
            "source_files": source_files,
            "tags": ["mock"],
        }

    def run_csim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        write_text(workdir / "mock_csim.log", "Mock csim passed.\n")
        return ToolResult(
            status="completed",
            return_code=0,
            stdout="Mock csim passed.",
            metrics={"csim_passed": True},
            command="mock_csim",
            duration_ms=1,
        )

    def run_synth(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        write_text(workdir / "mock_synth.log", "Mock synth passed. latency=100 II=1 LUT=64\n")
        return ToolResult(
            status="completed",
            return_code=0,
            stdout="Mock synth passed.",
            metrics={"synth_passed": True, "latency_best_cycles": 100, "II": 1, "resources_lut_used": 64},
            command="mock_synth",
            duration_ms=1,
        )

    def run_cosim(self, workdir: Path, source_files: list[Path], config: dict[str, Any]) -> ToolResult:
        write_text(workdir / "mock_cosim.log", "Mock cosim passed.\n")
        return ToolResult(
            status="completed",
            return_code=0,
            stdout="Mock cosim passed.",
            metrics={"cosim_passed": True, "can_cosim": True},
            command="mock_cosim",
            duration_ms=1,
        )

    def parse_metrics(self, artifacts: dict[str, Path]) -> dict[str, Any]:
        return {}


def build_hls_backend(kind: str, config: HLSBackendConfig | None = None) -> HLSBackend:
    if kind == "vitis":
        return VitisUnifiedHLSBackend(config)
    if kind == "command":
        return CommandHLSBackend(config)
    if kind == "mock":
        return MockHLSBackend()
    return HLSEvalBackend(config)
