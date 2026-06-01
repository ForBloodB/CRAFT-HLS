from pathlib import Path

from scripts.run_hls_eval_benchmark import (
    build_ccd_hls_gen_v2_prompt,
    build_output_code_repair_prompt,
    build_hls_eval_zero_shot_prompt,
    has_diff_markers,
    source_has_diff_marker,
    write_kernel_output_code,
    write_kernel_output_code_with_local_repair,
)


def test_output_code_exact_writes_only_kernel(tmp_path: Path):
    kernel = tmp_path / "kernel.cpp"
    header = tmp_path / "kernel.h"
    tb = tmp_path / "kernel_tb.cpp"
    kernel.write_text("// stub\n")
    header.write_text("// header\n")
    tb.write_text("// tb\n")

    ok, message = write_kernel_output_code(
        '<OUTPUT_CODE name="kernel.cpp">\n#include "kernel.h"\nvoid kernel() {}\n</OUTPUT_CODE>',
        kernel,
    )

    assert ok, message
    assert '#include "kernel.h"' in kernel.read_text()
    assert header.read_text() == "// header\n"
    assert tb.read_text() == "// tb\n"
    assert not source_has_diff_marker(tmp_path)


def test_output_code_rejects_diff_markers(tmp_path: Path):
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text("// stub\n")

    ok, message = write_kernel_output_code(
        """```diff
--- a/kernel.cpp
+++ b/kernel.cpp
@@ -1 +1 @@
-// stub
+void kernel() {}
```""",
        kernel,
    )

    assert not ok
    assert "diff markers" in message
    assert kernel.read_text() == "// stub\n"
    assert has_diff_markers("--- a/kernel.cpp\n+++ b/kernel.cpp\n")


def test_missing_output_code_uses_local_markdown_repair(tmp_path: Path):
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text("// stub\n")

    ok, message, mode = write_kernel_output_code_with_local_repair(
        """Here is the implementation:

```cpp
#include "kernel.h"
void kernel() {
}
```
""",
        kernel,
    )

    assert ok, message
    assert mode == "local_secondary_parse_markdown_cpp_block"
    assert '#include "kernel.h"' in kernel.read_text()


def test_missing_output_code_without_cpp_requests_llm_repair(tmp_path: Path):
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text("// stub\n")

    ok, message, mode = write_kernel_output_code_with_local_repair("I cannot solve this.", kernel)

    assert not ok
    assert "No <OUTPUT_CODE> block found" in message
    assert mode is None
    assert kernel.read_text() == "// stub\n"


def test_format_repair_prompt_is_rewrap_only():
    prompt = build_output_code_repair_prompt("kernel.cpp", "void kernel() {}")

    assert '<OUTPUT_CODE name="kernel.cpp">' in prompt
    assert "not solving the HLS task again" in prompt
    assert "Do not output a unified diff" in prompt


def test_contract_templates_render_generation_prompts(tmp_path: Path):
    desc = tmp_path / "kernel_description.md"
    header = tmp_path / "kernel.h"
    tb = tmp_path / "kernel_tb.cpp"
    kernel = tmp_path / "kernel.cpp"
    desc.write_text("Implement kernel.\n")
    header.write_text("void kernel();\n")
    tb.write_text("int main(){ kernel(); return 0; }\n")
    kernel.write_text("// stub\n")

    zero = build_hls_eval_zero_shot_prompt(desc, tb, header)
    ccd, selected, _ = build_ccd_hls_gen_v2_prompt(desc, tb, header, kernel, [], token_budget=5000, baseline_prompt_tokens=100)

    assert '<OUTPUT_CODE name="kernel.cpp">' in zero
    assert '<OUTPUT_CODE name="kernel.cpp">' in ccd
    assert "No additional CCD context selected" in ccd
    assert selected == []
