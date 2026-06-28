# CCD-HLS Agent

CCD-HLS Agent is an HLS-Eval / FPT 2026 research workspace. Roo Code is the
front-end agent for natural-language requirements, Phase-0 diagnosis, contract
review, and user approval. `HLS-agent` is the deterministic backend for
HLS-Eval-style runs, Vitis/tool execution, repair loops, token accounting,
budget ledgers, skill routing, and artifact replay.

The current default workflow is:

```text
natural-language request -> Roo diagnosis -> HLS-Eval-like contract
-> user review/lock -> CCD-HLS LOOP execution -> artifact-based report
```

## Install

```bash
python3 -m pip install -e ".[dev,tui]"
```

Optional HLS-Eval dependency:

```bash
python3 -m pip install "git+https://github.com/sharc-lab/hls-eval.git"
```

`HLS-agent` should then be available on `PATH`:

```bash
HLS-agent --help
```

## Model Config

This repo does not use `.env`. Put real API keys only in ignored local config
files:

```bash
cp configs/deepseek_v4_flash.json configs/deepseek_v4_flash.local.json
```

Example local file:

```json
{
  "profile_name": "cloud_deepseek_v4_flash",
  "provider_type": "cloud_openai",
  "base_url": "https://api.deepseek.com",
  "api_key": "YOUR_DEEPSEEK_KEY",
  "api_key_env": null,
  "model": "deepseek-v4-flash",
  "temperature": 0.2,
  "max_tokens": 4096,
  "timeout": 90.0
}
```

Tracked config files must keep `api_key` and `api_key_env` as `null` unless
they are harmless placeholders.

## Roo Code Workflow

Use Roo Code mode `hls-eval-agent`. The project mode and slash commands live in
`.roomodes` and `.roo/commands/`.

Typical command sequence:

```text
/hls-start
/hls-review-contract
/hls-lock-and-run
/hls-status
```

Roo must not run Vitis, CSIM, SYNTH, COSIM, or LLM code generation before the
contract is locked. The locked contract is an HLS-Eval-like directory:

```text
kernel_description.md
top.txt
<kernel>.h
<kernel>.cpp
<kernel>_tb.cpp
hls_eval_config.toml
diagnosis.json
contract_meta.json
```

`kernel_description.md` starts with `## Phase-0 Diagnosis`. The final report
must state the original diagnosis, whether it was resolved, and the evidence
from CSIM/SYNTH/COSIM/hidden-test/tool artifacts.

## HLS-agent Commands

Prepare and review a contract:

```bash
HLS-agent contract prepare \
  --request "Implement an AES add_round_key kernel for KV260" \
  --target-platform KV260 \
  --out workspaces/contracts/add_round_key

HLS-agent contract review workspaces/contracts/add_round_key
```

Lock and run a complete contract:

```bash
HLS-agent contract lock workspaces/contracts/add_round_key

HLS-agent contract run workspaces/contracts/add_round_key \
  --model-config configs/deepseek_v4_flash.local.json \
  --hls-backend vitis \
  --max-llm-calls 2 \
  --snapshot
```

Run one existing HLS-Eval case directly:

```bash
HLS-agent run \
  --case-path external/hls-eval/hls_eval_data/machsuite/md_knn \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/deepseek_v4_flash.local.json \
  --hls-backend vitis \
  --max-llm-calls 2 \
  --no-view
```

Inspect artifacts:

```bash
HLS-agent recent
HLS-agent view <run_dir> --snapshot
HLS-agent doctor
```

## Artifacts

Runtime outputs are local-only and ignored by Git:

```text
experiments/
workspaces/
external/
configs/*.local.json
configs/*_local.json
```

Important per-run artifacts include:

```text
result.json
stage_records.json
token_report.json
token_report.csv
budget_ledger.json
selected_skills.json
workflow_status.json
workflow_events.jsonl
```

Contract workflows additionally write:

```text
contract_stage_records.json
contract_token_report.json
workflow_token_summary.json
resolution_report.json
```

## Repository Layout

```text
ccd_hls_agent/     Core agent modules, CLI/TUI backend, skills, token/budget logic
scripts/           Compatibility wrappers and benchmark/cosim scripts
configs/           Non-secret model profiles
docs/              CCD-HLS v2/LOOP and FPT 2026 planning notes
tests/             Unit and integration-style tests
.roo/              Roo slash commands
.roomodes          Roo custom mode definition
```

`scripts/hls_tui.py` and `scripts/hls_agent_tui.py` are compatibility wrappers.
The canonical command is `HLS-agent`, implemented by `ccd_hls_agent.agent_tui`.

## Validate Before Commit

```bash
pytest -q
HLS-agent --help
git ls-files -co --exclude-standard -z | xargs -0 -r rg -n --hidden --no-heading 'sk-[A-Za-z0-9_-]{12,}|api_key\s*[:=]\s*"sk-|DEEPSEEK_API_KEY\s*=\s*sk-|OPENAI_API_KEY\s*=\s*sk-' || true
git status --short
```

The current intended commit scope is code and docs only. Do not commit real
keys, local configs, HLS-Eval external checkouts, run artifacts, or downloaded
paper PDFs.
