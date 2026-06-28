from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from .utils import read_text


class TaskMode(StrEnum):
    GENERATE = "generate"
    REPAIR_COMPILE = "repair_compile"
    REPAIR_CSIM = "repair_csim"
    REPAIR_SYNTH = "repair_synth"
    REPAIR_COSIM = "repair_cosim"
    OPTIMIZE_PPA = "optimize_ppa"


def is_generation_stub(kernel: Path) -> bool:
    text = read_text(kernel)
    return "HLS-Eval generation mode: reference implementation hidden." in text


def classify_task_mode(
    *,
    generation_stub: bool,
    csim_flags: dict[str, Any] | None = None,
    synth_passed: bool | None = None,
    cosim_passed: bool | None = None,
) -> TaskMode:
    if generation_stub:
        return TaskMode.GENERATE
    if csim_flags is not None:
        can_compile = bool(csim_flags.get("can_compile"))
        can_pass = bool(csim_flags.get("can_pass_testbench"))
        if not can_compile:
            return TaskMode.REPAIR_COMPILE
        if not can_pass:
            return TaskMode.REPAIR_CSIM
    if synth_passed is False:
        return TaskMode.REPAIR_SYNTH
    if cosim_passed is False:
        return TaskMode.REPAIR_COSIM
    if csim_flags and csim_flags.get("can_compile") and csim_flags.get("can_pass_testbench") and synth_passed is True:
        return TaskMode.OPTIMIZE_PPA
    return TaskMode.GENERATE
