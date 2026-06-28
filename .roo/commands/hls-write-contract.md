# /hls-write-contract

Write or update the HLS-Eval-like contract directory.

Required files:

- `kernel_description.md`
- `top.txt`
- `<kernel>.h`
- `<kernel>.cpp`
- `<kernel>_tb.cpp`
- `hls_eval_config.toml`
- `diagnosis.json`
- `contract_meta.json`

Preserve user-written content. Unknown fields must remain as TODO or blank. Do
not lock the contract and do not run backend tools.
