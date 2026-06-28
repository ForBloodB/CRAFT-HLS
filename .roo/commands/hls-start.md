# /hls-start

Start the HLS-Eval contract workflow from the user's natural-language request.

1. Ask for the target platform only if it is missing; default to `KV260`.
2. Save the user's request to a temporary markdown file only if needed.
3. Run:

```bash
HLS-agent contract prepare --request "<USER_REQUEST>" --target-platform KV260 --out workspaces/contracts/<short_task_name>
```

4. Show the generated `diagnosis.json`, `contract_meta.json`, and the HLS-Eval-like files.
5. Do not run Vitis, CSIM, SYNTH, COSIM, or code generation yet.
