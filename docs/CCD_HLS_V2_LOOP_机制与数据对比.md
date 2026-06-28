# CCD-HLS v2 与 CCD-HLS LOOP 机制及数据对比

生成时间：2026-06-01

本文比较四个当前保留的完整结果。注意：表中旧 zero-shot / agentic 数据来自本仓库历史适配流程；如果论文或最终 benchmark 要求“原仓库 runner 原封不动”，需要使用 HLS-Eval 原仓库 runner 重新生成 zero-shot / agentic 数据，再替换本表。

- HLS-Eval zero-shot：`experiments/full/full_deepseek_v4_flash_vitis_20260531_final`
- HLS-Eval agentic DeepSeek direct historical adapter：`experiments/full/hls_eval_agentic_deepseek_94x1_20260531`
- CCD-HLS v2：`experiments/full/full_ccd_hls_gen_v2_deepseek_20260531`
- CCD-HLS LOOP：`experiments/full/full_ccd_hls_gen_v2_repair_deepseek_20260531_163049`

说明：当前代码中的 LOOP 版本已经加入 repeated-failure early-stop 机制；但上面保留的 LOOP 全量结果是在 early-stop 加入前跑出的完整 94-case 日志驱动 repair-loop 结果。因此本文的“机制说明”按当前代码描述，“数据对比”按已有完整结果统计。若要量化 early-stop 对 token 和成功率的影响，需要用 `--methods ccd_hls_loop` 重新跑一版。

COSIM 结果来自独立验证目录：

- HLS-Eval agentic 与 CCD-HLS LOOP 全量 COSIM：`experiments/cosim/cosim_agentic_vs_loop_20260531_203541`
- HLS-Eval zero-shot 与 CCD-HLS v2 当前只有小样本 smoke/partial COSIM，不混入 94-case 主表。

## 1. CCD-HLS v2 使用的机制

CCD-HLS v2 是单轮生成式流程，核心目标是在不使用 agentic shell action 的前提下，尽量减少 prompt token，并提高输出格式稳定性。

流程如下：

```text
STATIC_SCAN
  -> CONTEXT_ATOMIZE
  -> CONTEXT_SCORE
  -> PROMPT_BUILD
  -> LLM_GENERATION
  -> OUTPUT_CODE_PARSE
  -> CSIM
  -> SYNTH
  -> DONE / FAILED
```

主要机制：

1. 结构化输出 contract

   模型必须输出：

   ```xml
   <OUTPUT_CODE name="kernel.cpp">
   ...
   </OUTPUT_CODE>
   ```

   不再要求 unified diff，也不要求 JSON metadata。这样降低了格式错误概率，也减少了 completion token。

2. 只允许写 kernel 文件

   v2 只会把 `<OUTPUT_CODE>` 中与 kernel 文件名完全匹配的内容写入 kernel `.cpp`，不允许修改 header/testbench。若输出 diff marker，如 `--- a/`、`+++ b/`、`@@`，会直接拒绝。

3. CCD context capsule

   对 benchmark 进行静态扫描后，将任务描述、代码结构、header/testbench 约束等信息转成 atom，再用 value/certainty 选择少量高价值 atom。默认最多 6 条，目标是让 prompt 不超过 zero-shot prompt 的约 `1.10x`。

4. 本地二次解析

   若模型没有返回 `<OUTPUT_CODE>`，会先尝试从 markdown C++ code block 或裸 C++ 源码中本地恢复。这个过程不调用 LLM。

5. 单轮评测

   生成后运行 CSIM；CSIM 通过后运行 SYNTH。若 CSIM 或 SYNTH 失败，v2 不进入日志驱动 repair loop，当前 case 直接失败。

## 2. CCD-HLS LOOP 使用的机制

CCD-HLS LOOP 在 v2 的基础上增加了有界多轮修复。它不是固定跑满 5 轮，而是“成功即停，失败才修”。

流程如下：

```text
GENERATION
  -> PARSE_VALIDATE
  -> FORMAT_REPAIR, optional
  -> CSIM
  -> CSIM_REPAIR, optional
  -> CSIM
  -> SYNTH
  -> SYNTH_REPAIR, optional
  -> CSIM
  -> SYNTH
  -> DONE / FAILED / EARLY_STOP
```

主要机制：

1. 有界 LLM 调用

   `--max-llm-calls 5` 表示每个 case 最多调用 5 次 LLM，包括初始生成、格式修复、CSIM repair、SYNTH repair。成功后直接进入下一阶段，不会用满预算。

2. 失败日志 capsule

   每次 CSIM/SYNTH 失败后，不把完整日志塞回 prompt，而是提取：

   - `failure_type`
   - `key_errors`
   - `signal_lines`
   - stdout/stderr tail
   - return code、工具命令、阶段 metrics

   这些内容写入 artifact，并作为下一次 repair prompt 的输入。

3. Failure history

   repair prompt 不只包含当前失败，还包含最近几次 failure capsule 的摘要，使模型能看到“上一次改完后错误有没有变化”。

4. 阶段短路

   - CSIM 通过后直接进入 SYNTH。
   - SYNTH 通过后标记 `DONE / synth_passed`，进入下一个 case。
   - CSIM 最终失败时不会继续 SYNTH。

5. Repeated-failure early-stop

   当前代码已经加入 early-stop：若连续两次 failure capsule 的 `failure_type + key_errors` 高度相似，默认相似度阈值为 `0.92`，则提前停止当前 case，记录：

   - `early_stop_triggered=true`
   - `early_stop_similarity`
   - `stopped_reason=early_stop_repeated_failure`
   - `EARLY_STOP` stage record

   命令行参数：

   ```bash
   --early-stop-similarity-threshold 0.92
   ```

   设置为 `0` 可关闭 early-stop。

## 3. Contract 模板位置

当前 contract 模板已经独立成可直接修改的 Markdown 文件：

- `ccd_hls_agent/templates/ccd_hls_gen_v2.md`
- `ccd_hls_agent/templates/output_code_repair.md`
- `ccd_hls_agent/templates/hls_repair.md`

其中 v2 主要使用 `ccd_hls_gen_v2.md`，LOOP 会额外使用 `output_code_repair.md` 和 `hls_repair.md`。HLS-Eval zero-shot 不再使用本地 Markdown contract，而是直接调用 `external/hls-eval/hls_eval/prompts.py` 中的 `build_prompt_gen_zero_shot`。

如果论文或正式对照要求“原仓库 runner 原封不动”，zero-shot 应使用 `external/hls-eval/hls_eval_experiments/hls_gen_zero_shot__main/exp.py`，agentic 应使用 `external/hls-eval/hls_eval_experiments/hls_gen_agent_miniswe/exp.py` 或 `hls_gen_agent_pi/exp.py`。本仓库保留的 DeepSeek direct agentic 脚本是适配器，不应和原仓库 runner 结果混称。

## 4. 指标口径

本文统计 94 个 HLS-Eval case，`samples=1`。

指标解释：

- CSIM compile：`can_compile`
- CSIM pass：`can_pass_testbench`
- SYNTH pass：`can_synthesize`
- COSIM：RTL cosim 独立运行；目前只有 HLS-Eval agentic 与 CCD-HLS LOOP 有 94-case 全量 COSIM 结果
- 平均 token：`avg_total_tokens = avg(prompt_tokens + completion_tokens)`
- 平均 LLM 调用轮次：每一轮模型调用算 1 次调用，而不是每个任务算 1 次。zero-shot 和 CCD-HLS v2 每个 case 固定 1 次；HLS-Eval agentic 使用 `agent_steps`；CCD-HLS LOOP 使用 `metrics.llm_calls_used`
- 平均 repair 轮次：v2 为 0；LOOP 使用 `metrics.repair_rounds`
- 平均 HLS 验证工具调用：为了让四个方法可比，本文不再使用原来的“平均验证轮次 / case”。新指标只统计实际调用外部 HLS 验证工具的次数，CSIM 记 1 次，SYNTH 记 1 次；对 agentic 只统计 `metrics.hls_tool_calls`，不把 agent 的 bash shell action 算作 HLS 验证调用。该指标当前不含 COSIM，因为 zero-shot/v2 尚无 94-case 全量 COSIM。
- 每次 LLM 调用平均 token：总 token / 实际 LLM 调用轮次数

## 5. 数据对比

| 指标 | HLS-Eval zero-shot | HLS-Eval agentic | CCD-HLS v2 | CCD-HLS LOOP |
|---|---:|---:|---:|---:|
| 样本数 | 94 | 94 | 94 | 94 |
| Parse 成功 | 77 / 94, 81.91% | 89 / 94, 94.68% | 79 / 94, 84.04% | 94 / 94, 100.00% |
| CSIM compile 成功 | 63 / 94, 67.02% | 75 / 94, 79.79% | 67 / 94, 71.28% | 84 / 94, 89.36% |
| CSIM pass testbench | 63 / 94, 67.02% | 75 / 94, 79.79% | 67 / 94, 71.28% | 84 / 94, 89.36% |
| SYNTH 成功 | 62 / 94, 65.96% | 74 / 94, 78.72% | 67 / 94, 71.28% | 84 / 94, 89.36% |
| COSIM 全量通过 | 未全量执行 | 73 / 94, 77.66% | 未全量执行 | 82 / 94, 87.23% |
| COSIM 通过 | 未全量执行 | 73 / 74, 98.65% | 未全量执行 | 82 / 84, 97.62% |
| COSIM 平均耗时 | 未全量执行 | 72.0 s | 未全量执行 | 84.8 s |
| 平均 prompt tokens / case | 1825.1 | 35437.5 | 1975.9 | 4696.4 |
| 平均 completion tokens / case | 1923.6 | 13862.4 | 1977.1 | 3885.5 |
| 平均 total tokens / case | 3748.7 | 49299.9 | 3953.1 | 8581.9 |
| 总 token | 352378 | 4634188 | 371588 | 806695 |
| 总 LLM 调用轮次数 | 94 | 749 | 94 | 160 |
| 平均 LLM 调用轮次 / case | 1.00 | 7.97 | 1.00 | 1.70 |
| 平均 repair/agent 循环轮次 / case | 0.00 | 7.97 | 0.00 | 0.46 |
| 总 HLS 验证工具调用（CSIM+SYNTH） | 140 | 164 | 146 | 225 |
| 平均 HLS 验证工具调用 / case（CSIM+SYNTH） | 1.49 | 1.74 | 1.55 | 2.39 |
| 每次 LLM 调用平均 token | 3748.7 | 6187.2 | 3953.1 | 5041.8 |

## 6. LOOP 的收益与代价

LOOP 相比其他方法的收益：

- 相比 HLS-Eval zero-shot，SYNTH 成功从 62/94 提升到 84/94，额外成功 22 个 case。
- 相比 HLS-Eval agentic，SYNTH 成功从 74/94 提升到 84/94，额外成功 10 个 case。
- 相比 CCD-HLS v2，SYNTH 成功从 67/94 提升到 84/94，额外成功 17 个 case。
- 在已完成的全量 COSIM 对照中，LOOP 的 COSIM all-case 通过数为 82/94，高于 agentic 的 73/94；二者在已尝试 COSIM 的设计上通过率都很高，agentic 为 98.65%，LOOP 为 97.62%。

LOOP 的 token/调用代价：

- 相比 HLS-Eval zero-shot，平均 total tokens 从 3748.7 增加到 8581.9，约为 2.29 倍。
- 相比 HLS-Eval agentic，平均 total tokens 从 49299.9 降到 8581.9，约减少 82.6%。
- 相比 CCD-HLS v2，平均 total tokens 从 3953.1 增加到 8581.9，约为 2.17 倍。
- LOOP 总 LLM 调用轮次数为 160，平均每个 case 1.70 次；agentic 总调用轮次数为 749，平均每个 case 7.97 次。
- LOOP 每次 LLM 调用平均 token 为 5041.8，低于 agentic 的 6187.2，但高于 zero-shot 和 v2 的单轮调用。
- LOOP 的平均 HLS 验证工具调用为 2.39 次/case，高于 zero-shot、agentic 和 v2，说明它用更多低成本工具验证换取了更高的最终通过率。

## 7. 结论

HLS-Eval zero-shot 是最朴素的单轮基线，当前代码直接使用原仓库的 `build_prompt_gen_zero_shot`，token 最少，但成功率最低。

HLS-Eval agentic 通过多步 shell action 提升了成功率，但平均每个 case 需要 7.97 次 LLM 调用轮次，平均 total tokens 最高。

CCD-HLS v2 是低成本单轮 CCD 基线：token 接近 zero-shot，成功率略高，但一旦格式、CSIM 或 SYNTH 失败，无法自我修复。

CCD-HLS LOOP 是面向准确率的日志驱动版本：通过 failure capsule 和 bounded repair loop 显著提高 CSIM/SYNTH 成功率。当前数据表明，LOOP 用 8581.9 平均 total tokens 和 1.70 次平均 LLM 调用轮次，将 SYNTH 成功率提高到 89.36%，高于 HLS-Eval zero-shot、HLS-Eval agentic 和 CCD-HLS v2。

原来的“平均验证轮次 / case”不适合作为四方法统一指标，因为 zero-shot/v2 的验证是单次 CSIM/SYNTH，LOOP 的 `attempt_count` 是日志驱动 repair 后的验证尝试轮，agentic 的 `agent_steps` 则是模型动作轮，不等价于 HLS 验证轮。本文已经改用“平均 HLS 验证工具调用 / case（CSIM+SYNTH）”，四个方法都可以按同一规则比较。

若研究目标是“更少 token 下的更高准确率”，下一步应重跑加入 early-stop 后的 LOOP，并比较：

- SYNTH 成功率是否保持接近 84/94；
- 平均 LLM 调用次数是否低于 1.70；
- 平均 total tokens 是否低于 8581.9；
- 失败 case 中 `early_stop_repeated_failure` 的占比。
