from pathlib import Path

from ccd_hls_agent.ccd import atomize_static_scan, build_prompt, choose_frontier, scan_benchmark, score_atoms, select_context


def make_case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    case.mkdir()
    (case / "top.txt").write_text("kernel\n")
    (case / "kernel_description.md").write_text("Add two arrays element-wise.")
    (case / "hls_eval_config.toml").write_text("tags = ['unit']\n")
    (case / "kernel.cpp").write_text(
        """
void kernel(int A[16], int B[16], int C[16]) {
  for (int i = 0; i < 16; ++i) {
    #pragma HLS PIPELINE II=1
    C[i] = A[i] + B[i];
  }
}
"""
    )
    (case / "kernel_tb.cpp").write_text("int main(){return 0;}\n")
    return case


def test_scan_score_select_and_prompt(tmp_path: Path):
    case = make_case(tmp_path)
    scan = scan_benchmark(case)
    assert scan.top_function == "kernel"
    assert scan.metrics["loops"] == 1
    assert scan.metrics["arrays"] == 3

    atoms = atomize_static_scan("task_1", "run_1", case, scan)
    frontier = choose_frontier(scan, scan.blocker)
    scored = score_atoms(atoms, frontier)
    selected, dropped = select_context(scored, token_budget=500)

    assert selected
    assert len(selected) + len(dropped) == len(scored)
    assert selected[0].value_score > 0

    prompt = build_prompt(scan, selected, [], stage="LLM_CALL", frontier=frontier, token_budget=2000)
    assert "unified diff" in prompt
    assert "kernel" in prompt

