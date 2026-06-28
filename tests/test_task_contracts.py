from pathlib import Path

import pytest

from ccd_hls_agent.task_contracts import (
    DIAGNOSIS_TYPES,
    diagnose_request,
    ensure_contract_locked,
    lock_contract,
    prepare_hls_eval_contract,
    review_contract,
    summarize_resolution,
)


def test_diagnosis_classes_cover_expected_inputs():
    assert diagnose_request("void kernel(int *a); please optimize latency")["diagnosis_type"] == "A_FUNCTIONALLY_CORRECT_BUT_UNOPTIMIZED_BASELINE"
    assert diagnose_request("Vitis compile error: no matching function for call")["diagnosis_type"] == "B_FAILS_COMPILATION_OR_SYNTHESIS"
    assert diagnose_request("CSIM failed with mismatch at output[3]")["diagnosis_type"] == "C_COMPILES_BUT_FAILS_CSIM_COSIM_OR_HIDDEN_TESTS"
    assert diagnose_request("dataflow deadlock due to FIFO depth and invalid streaming behavior")["diagnosis_type"] == "D_STRUCTURAL_DEADLOCK_STREAMING_OR_RESOURCE_ISSUE"
    assert diagnose_request("Need an HLS module for a custom transform")["diagnosis_type"] == "E_OTHER_HLS_COMPILATION_PROBLEM"
    assert len(DIAGNOSIS_TYPES) == 5


def test_prepare_contract_writes_hls_eval_like_directory_with_todos(tmp_path: Path):
    contract_dir = tmp_path / "contract"

    meta = prepare_hls_eval_contract("Need an HLS kernel that adds two arrays.", contract_dir)

    assert (contract_dir / "kernel_description.md").exists()
    assert (contract_dir / "top.txt").exists()
    assert (contract_dir / "kernel.h").exists()
    assert (contract_dir / "kernel.cpp").exists()
    assert (contract_dir / "kernel_tb.cpp").exists()
    assert (contract_dir / "hls_eval_config.toml").exists()
    assert (contract_dir / "diagnosis.json").exists()
    assert (contract_dir / "contract_meta.json").exists()
    assert "Phase-0 Diagnosis" in (contract_dir / "kernel_description.md").read_text()
    assert "top.txt" in meta["missing_fields"]
    assert "TODO: complete top function signature" in (contract_dir / "kernel.h").read_text()


def test_lock_rejects_missing_fields_and_hash_detects_edits(tmp_path: Path):
    contract_dir = tmp_path / "contract"
    prepare_hls_eval_contract(
        "Implement void add(int *a, int *b, int *c); testbench expected behavior: c[i]=a[i]+b[i].",
        contract_dir,
        task_id="add",
    )
    review = review_contract(contract_dir)
    assert "testbench_or_expected_behavior" in review["missing_fields"]

    (contract_dir / "add_tb.cpp").write_text('#include "add.h"\nint main(){return 0;}\n')
    locked = lock_contract(contract_dir)
    assert locked["approved_by_user"] is True
    ensure_contract_locked(contract_dir)

    (contract_dir / "kernel_description.md").write_text((contract_dir / "kernel_description.md").read_text() + "\nChanged.\n")
    with pytest.raises(ValueError, match="changed after lock"):
        ensure_contract_locked(contract_dir)


def test_resolution_report_uses_initial_diagnosis_basis(tmp_path: Path):
    contract_dir = tmp_path / "contract"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prepare_hls_eval_contract(
        "CSIM failed with mismatch. void add(int *a, int *b, int *c); expected behavior: c[i]=a[i]+b[i].",
        contract_dir,
    )
    (run_dir / "result.json").write_text('{"can_pass_testbench": true, "can_synthesize": false}\n')

    summary = summarize_resolution(contract_dir, run_dir)

    assert summary["resolved"] is True
    assert "CSIM" in summary["basis"]
