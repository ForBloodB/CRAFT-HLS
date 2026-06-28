# /hls-fill-contract

Fill TODO or blank contract fields with the LLM.

Rules:

- Do not overwrite user-written content.
- Keep uncertain fields as TODO.
- Keep the contract in HLS-Eval-like directory form.
- After editing, run:

```bash
HLS-agent contract review <contract_dir>
```

Then show missing fields and token status.
