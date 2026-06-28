# /hls-diagnose

Diagnose the current user request or contract.

Use the five fixed diagnosis classes:

- `A_FUNCTIONALLY_CORRECT_BUT_UNOPTIMIZED_BASELINE`
- `B_FAILS_COMPILATION_OR_SYNTHESIS`
- `C_COMPILES_BUT_FAILS_CSIM_COSIM_OR_HIDDEN_TESTS`
- `D_STRUCTURAL_DEADLOCK_STREAMING_OR_RESOURCE_ISSUE`
- `E_OTHER_HLS_COMPILATION_PROBLEM`

If a contract directory already exists, read `diagnosis.json` and `contract_meta.json`.
Otherwise run `/hls-start` first. Report evidence, missing fields, and confidence.
