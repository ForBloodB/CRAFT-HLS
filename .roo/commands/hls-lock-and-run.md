# /hls-lock-and-run

Lock the contract and run the backend. This command is the only approval gate.

1. Run:

```bash
HLS-agent contract lock <contract_dir>
```

2. Then run:

```bash
HLS-agent contract run <contract_dir> --model-config configs/deepseek_v4_flash.local.json
```

3. Poll or inspect `workflow_status.json`, `workflow_events.jsonl`,
   `token_report.json`, `contract_token_report.json`, and
   `resolution_report.json`.

4. Final report must state whether the Phase-0 diagnosis was resolved and cite
   CSIM/SYNTH/COSIM/hidden-test/tool-log evidence.
