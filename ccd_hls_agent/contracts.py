from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

from .failure_analysis import summarize_failure_history, truncate_estimated_tokens
from .utils import estimate_tokens, read_text


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def load_contract_template(name: str, template_dir: Path | None = None) -> str:
    root = template_dir or TEMPLATE_DIR
    return read_text(root / name)


def render_contract_template(name: str, template_dir: Path | None = None, **values: Any) -> str:
    return Template(load_contract_template(name, template_dir)).substitute(**values)


def _input_code(path: Path, *, token_budget: int | None = None) -> str:
    text = read_text(path)
    if token_budget is not None:
        text = truncate_estimated_tokens(text, token_budget)
    return f'<INPUT_CODE name="{path.name}">\n{text}\n</INPUT_CODE>'


def render_atom_lines(selected_atoms: list[Any]) -> str:
    lines = []
    for atom in selected_atoms:
        lines.append(
            f"- [{atom.kind} scope={atom.scope} value={atom.value_score:.3f} "
            f"certainty={atom.certainty_score:.3f}] {atom.summary}"
        )
    return "\n".join(lines) or "- No additional CCD context selected."


def build_ccd_hls_gen_v2_prompt(
    description: Path,
    tb: Path,
    header: Path,
    kernel: Path,
    selected_atoms: list[Any],
    *,
    token_budget: int,
    baseline_prompt_tokens: int,
    hls_skill_capsule: str = "- No HLS skills selected.",
    template_dir: Path | None = None,
) -> tuple[str, list[Any], int]:
    max_prompt_tokens = max(1, min(token_budget, int(baseline_prompt_tokens * 1.10)))

    def render(atom_subset: list[Any]) -> str:
        return render_contract_template(
            "ccd_hls_gen_v2.md",
            template_dir,
            kernel_name=kernel.name,
            ccd_context=render_atom_lines(atom_subset),
            hls_skill_capsule=hls_skill_capsule,
            description_input=_input_code(description),
            tb_input=_input_code(tb),
            header_input=_input_code(header),
        )

    kept = list(selected_atoms)
    while kept:
        prompt = render(kept)
        if estimate_tokens(prompt) <= max_prompt_tokens:
            return prompt, kept, max_prompt_tokens
        kept = kept[:-1]
    return render([]), [], max_prompt_tokens


def build_output_code_repair_prompt(kernel_name: str, original_response: str, template_dir: Path | None = None) -> str:
    return render_contract_template(
        "output_code_repair.md",
        template_dir,
        kernel_name=kernel_name,
        original_response=original_response,
    )


def build_hls_repair_prompt(
    *,
    stage: str,
    kernel: Path,
    header: Path,
    tb: Path,
    description: Path,
    failure_capsule: dict[str, Any],
    failure_history: list[dict[str, Any]],
    attempt: int,
    max_llm_calls: int,
    hls_skill_capsule: str = "- No HLS skills selected.",
    local_memory_capsule: str = "- No verified local memory matched this failure.",
    token_budget: int | None = None,
    template_dir: Path | None = None,
) -> str:
    budget_steps = [
        {"description": 1200, "header": 1600, "tb": 2200, "kernel": 2600, "history": 4, "memory": 900},
        {"description": 700, "header": 1400, "tb": 1400, "kernel": 2200, "history": 3, "memory": 600},
        {"description": 450, "header": 1200, "tb": 900, "kernel": 1800, "history": 2, "memory": 400},
        {"description": 300, "header": 1000, "tb": 600, "kernel": 1400, "history": 1, "memory": 260},
        {"description": 180, "header": 800, "tb": 360, "kernel": 1000, "history": 1, "memory": 160},
    ]

    def render_with_budget(step: dict[str, int]) -> str:
        memory_text = truncate_estimated_tokens(local_memory_capsule, step["memory"])
        return render_contract_template(
            "hls_repair.md",
            template_dir,
            stage=stage,
            kernel_name=kernel.name,
            attempt=attempt,
            max_llm_calls=max_llm_calls,
            failure_capsule_json=json.dumps(failure_capsule, ensure_ascii=False, indent=2),
            failure_history_json=json.dumps(
                summarize_failure_history(failure_history, limit=step["history"]),
                ensure_ascii=False,
                indent=2,
            ),
            hls_skill_capsule=hls_skill_capsule,
            local_memory_capsule=memory_text,
            description_input=_input_code(description, token_budget=step["description"]),
            header_input=_input_code(header, token_budget=step["header"]),
            tb_input=_input_code(tb, token_budget=step["tb"]),
            kernel_input=_input_code(kernel, token_budget=step["kernel"]),
        )

    prompt = render_with_budget(budget_steps[0])
    if token_budget is None:
        return prompt
    for step in budget_steps:
        prompt = render_with_budget(step)
        if estimate_tokens(prompt) <= token_budget:
            return prompt
    return prompt
