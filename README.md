# CRAFT-HLS

未来论文名建议：

```text
CRAFT-HLS: Contract-Refined Agentic Flow for Token-Efficient HLS Generation and Repair
```

中文可以写作：

```text
CRAFT-HLS：面向 HLS 生成与修复的合同精化、Token 高效智能体流程
```

这个名字把项目的三件核心事放在一起：先把自然语言需求精化成可审阅的 HLS-Eval 合同，再由 agent 执行生成、仿真、综合和修复，最后用 token report、budget ledger、skill router 和 artifacts 证明过程可复现、可比较、可写进论文。

仓库里的命令名仍叫 `HLS-agent`，这是稳定 CLI 入口；`CRAFT-HLS` 是项目名、方法名和未来论文名。

## 1. 项目做什么

CRAFT-HLS 面向 FPT 2026 / HLS-Eval Track A，目标是构建一个可复现的 HLS agent 工作流：

```text
用户自然语言需求
-> Roo Code 诊断
-> HLS-Eval-like 合同
-> 用户审阅并锁定合同
-> CCD-HLS LOOP 后端执行
-> 生成代码、CSIM、SYNTH、修复循环
-> token / budget / skill / workflow artifacts
-> 判断是否解决最初诊断的问题
```

当前核心能力：

- Roo Code 作为主交互入口，负责需求诊断、合同补全、用户确认。
- `HLS-agent` 作为确定性后端，负责运行 HLS-Eval case、调用模型、调用 HLS backend、保存 artifacts。
- CCD-HLS v2 / LOOP 作为内部生成与修复方法，支持有界 repair loop。
- M1-M3 已接入 stage-level token report、BudgetLedger、task mode、HLS skill router。
- 完全本地增强路径已接入 failure taxonomy、deterministic repair、本地 SQLite memory 和 tool-verified memory 更新。
- 默认目标平台按 KV260 组织合同字段；真实工具链仍由本机 Vitis/HLS-Eval 环境决定。

## 2. 目录结构

```text
ccd_hls_agent/     核心 Python 包：CLI/TUI、contract、workflow、token、budget、skill、HLS backend
scripts/           兼容入口、benchmark runner、cosim 验证脚本
configs/           不含密钥的模型配置模板
docs/              CCD-HLS v2/LOOP 机制、FPT 2026 规划文档
tests/             单元测试与集成风格测试
.roo/              Roo slash commands
.roomodes          Roo custom mode
references/        本地论文资料索引；PDF/HTML 默认不提交
```

本地运行产物不会提交：

```text
external/          HLS-Eval upstream checkout
experiments/       benchmark 和真实 Vitis run 输出
workspaces/        contract 工作区
configs/*.local.json
configs/*_local.json
```

## 3. 环境准备

推荐 Python 版本：`>=3.13`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,tui]"
```

确认 CLI 可用：

```bash
HLS-agent --help
```

如果要接入 HLS-Eval 数据集：

```bash
mkdir -p external
git clone https://github.com/sharc-lab/hls-eval.git external/hls-eval
python -m pip install -e external/hls-eval
```

如果要跑真实 Vitis backend，需要先加载 Vitis 环境，使下面命令可见：

```bash
which v++
which vitis-run
```

例如你的机器可能需要：

```bash
source /tools/Xilinx/Vitis/2025.2/settings64.sh
```

实际路径以本机安装为准。

后台实验建议显式传入 `--hls-platform`。当平台文件位于 Vitis 安装目录下时，backend 会从 `.xpfm` 路径推断 `Vitis/bin` 并注入子进程 `PATH`，避免 `nohup`/非交互 shell 中出现 `vitis-run not found`。

## 4. 模型配置

本项目不使用 `.env`。真实 key 只放在 ignored local config。

DeepSeek 示例：

```bash
cp configs/deepseek_v4_flash.json configs/deepseek_v4_flash.local.json
```

然后编辑 `configs/deepseek_v4_flash.local.json`：

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

本地 OpenAI-compatible 模型也可以使用同一结构，例如：

```json
{
  "profile_name": "local_qwen3_coder",
  "provider_type": "local_openai",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": null,
  "api_key_env": null,
  "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  "temperature": 0.2,
  "max_tokens": 4096,
  "context_window": 16384,
  "timeout": 180.0
}
```

建议保存为 `configs/qwen3_coder_30b.local.json`，该文件会被 Git 忽略。

检查配置：

```bash
HLS-agent doctor \
  --model-config configs/deepseek_v4_flash.local.json \
  --hls-eval-root external/hls-eval
```

## 5. 最小复现：不需要模型、不需要 Vitis

这一步用于确认仓库、CLI、合同层和测试都正常。

```bash
pytest -q
HLS-agent --help
```

准备一个 HLS-Eval-like 合同：

```bash
HLS-agent contract prepare \
  --request "Implement an AES add_round_key kernel for KV260. Top function add_round_key. Signature: void add_round_key(unsigned char state[16], const unsigned char round_key[16]); Expected behavior: xor each state byte with the matching round key byte." \
  --target-platform KV260 \
  --out workspaces/contracts/add_round_key_demo
```

查看合同摘要和合同阶段 token 统计：

```bash
HLS-agent contract review workspaces/contracts/add_round_key_demo
HLS-agent contract status workspaces/contracts/add_round_key_demo
```

这会生成：

```text
workspaces/contracts/add_round_key_demo/
  kernel_description.md
  top.txt
  add_round_key.h
  add_round_key.cpp
  add_round_key_tb.cpp
  hls_eval_config.toml
  diagnosis.json
  contract_meta.json
  contract_stage_records.json
  contract_token_report.json
  workflow_status.json
  workflow_events.jsonl
```

## 6. Agent 流程复现：需要模型，不需要 Vitis

如果你只想验证 agent 运行闭环、artifact 格式、token/budget/skill 输出，可以使用 `mock` HLS backend。注意：`mock` 只 mock HLS 工具，不 mock LLM，所以仍需要模型配置。

```bash
HLS-agent run \
  --case-path external/hls-eval/hls_eval_data/c2hlsc/add_round_key \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/deepseek_v4_flash.local.json \
  --hls-backend mock \
  --max-llm-calls 1 \
  --no-view \
  --snapshot
```

输出会写入 `experiments/`。查看最近结果：

```bash
HLS-agent recent
HLS-agent view <run_dir> --snapshot
```

## 7. 完全本地知识增强模式

完全本地模式不依赖 DeepSeek 或其他云 API。推荐组合是：

```text
本地 Qwen3/Qwen2.5
+ failure taxonomy
+ failure-conditioned skill router
+ deterministic repair
+ local SQLite memory
+ Vitis CSIM/SYNTH verification
```

默认开关：

```text
early-stop threshold: 0.92
deterministic repair: enabled
local memory: enabled
memory path: .hls_agent/memory/hls_memory.sqlite
skill token budget: 600
local model context window: 16384
```

本地 llama.cpp server 建议以 16K context 启动：

```bash
CTX=16384 PARALLEL=1 bash external/local_model_scripts/start_llama_server.sh 30b
```

如果 prompt 估算长度仍超过 `context_window - max_tokens`，runner 会在调用模型前自动压缩 repair 信息；压缩优先减少 failure history、日志摘录、testbench/kernel 摘录和 memory capsule，而不是直接让 llama.cpp 返回 `context size` 错误。

本地 Qwen3 运行示例：

```bash
HLS-agent run \
  --case-path external/hls-eval/hls_eval_data/c2hlsc/add_round_key \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/qwen3_coder_30b.local.json \
  --hls-backend vitis \
  --hls-platform /opt/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm \
  --max-llm-calls 5 \
  --early-stop-similarity-threshold 0.92 \
  --skill-token-budget 600 \
  --repair-log-token-budget 1200 \
  --memory-path .hls_agent/memory/hls_memory.sqlite \
  --no-view \
  --snapshot
```

如果不想在每条命令里写平台路径，也可以只在本机 shell 中设置：

```bash
export HLS_PLATFORM=/opt/2025.2/Vitis/base_platforms/xilinx_kv260_base_202520_1/xilinx_kv260_base_202520_1.xpfm
export HLS_PART=xck26-sfvc784-2LV-c
```

命令行显式传入的 `--hls-platform` / `--hls-part` 优先级高于环境变量。

消融实验可以关闭模块：

```bash
--disable-deterministic-repair
--disable-local-memory
--early-stop-similarity-threshold 0
```

新增 artifacts / metrics：

```text
failure_class
root_cause_hints
recommended_policy
local_memory_hits.json
DETERMINISTIC_REPAIR stage records
local_memory_positive_updates
local_memory_negative_updates
```

知识闭环：

```text
静态分析 + failure classifier
-> 检索 K0-K4 本地知识/memory
-> 注入短 Knowledge Capsule
-> deterministic repair 或 Qwen3 repair
-> Vitis 验证
-> positive/negative memory 更新
```

## 8. 完整复现：真实 DeepSeek + Vitis

完整复现需要：

- `external/hls-eval` 已 clone。
- Vitis 环境已加载，`v++` 和 `vitis-run` 可执行。
- `configs/deepseek_v4_flash.local.json` 已写入真实 key。

先跑一个小 case：

```bash
HLS-agent run \
  --case-path external/hls-eval/hls_eval_data/c2hlsc/add_round_key \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/deepseek_v4_flash.local.json \
  --hls-backend vitis \
  --max-llm-calls 2 \
  --no-view \
  --snapshot
```

成功时重点看：

```text
can_parse = true
can_compile = true
can_pass_testbench = true
can_synthesize = true
```

如果失败，仍然是有效实验结果。查看：

```bash
HLS-agent view <run_dir> --snapshot
```

重点 artifacts：

```text
result.json             最终结果布尔值、token 总数、metrics
stage_records.json      每个阶段的状态、message、artifact 路径
token_report.json       stage-level token 统计
token_report.csv        单 case token 表
budget_ledger.json      LLM/CSIM/SYNTH/COSIM budget 消耗
selected_skills.json    skill router 命中的 HLS skills
workflow_status.json    当前/最终 workflow 阶段
workflow_events.jsonl   阶段事件流
failure_capsules.json   压缩后的失败证据
```

## 9. Roo Code 使用方式

在 Roo Code 中选择项目 custom mode：

```text
hls-eval-agent
```

建议交互顺序：

```text
/hls-start
/hls-review-contract
/hls-fill-contract
/hls-lock-and-run
/hls-status
```

Roo 的职责：

- 读取用户自然语言功能需求和目标平台，默认 KV260。
- 做 Phase-0 diagnosis。
- 写出 HLS-Eval-like 合同目录。
- 要求用户审阅和修正合同。
- 只有用户执行 `/hls-lock-and-run` 后，才调用 `HLS-agent contract lock/run`。

合同主体文件：

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

`kernel_description.md` 第一节必须是：

```text
## Phase-0 Diagnosis
```

最终报告必须回答：

- 第一次诊断的问题是什么。
- 是否解决了这个问题。
- 依据是什么：CSIM、SYNTH、COSIM、hidden test、tool log 或 resource report。

## 10. 合同锁定后运行

如果合同已经完整，可以手动锁定：

```bash
HLS-agent contract lock workspaces/contracts/add_round_key_demo
```

锁定后运行：

```bash
HLS-agent contract run workspaces/contracts/add_round_key_demo \
  --model-config configs/deepseek_v4_flash.local.json \
  --hls-backend vitis \
  --max-llm-calls 2 \
  --snapshot
```

如果锁定后修改了合同文件，`contract run` 会拒绝执行。需要重新 review 和 lock。

## 11. 批量 benchmark

直接调用 runner：

```bash
python scripts/run_hls_eval_benchmark.py \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/deepseek_v4_flash.local.json \
  --methods ccd_hls_loop \
  --samples 1 \
  --hls-backend vitis \
  --max-llm-calls 2 \
  --case-filter "add_round_key$" \
  --out-dir experiments/full/add_round_key_smoke
```

批量跑更多 case 时，可以调整：

```text
--limit
--case-filter
--samples
--max-llm-calls
--hls-part
--hls-platform
--llm-call-budget
--csim-budget
--synth-budget
--cosim-budget
--skill-token-budget
--repair-log-token-budget
--early-stop-similarity-threshold
--disable-deterministic-repair
--disable-local-memory
--memory-path
```

## 12. 论文实验应该汇报什么

建议至少汇报：

- `can_parse`
- `can_compile`
- `can_pass_testbench`
- `can_synthesize`
- `total_tokens`
- `tokens_by_stage`
- `llm_calls_by_stage`
- `tool_calls_by_stage`
- `tokens_per_synth_success`
- `budget_summary`
- `selected_skills`
- `failure_class` 分布
- `memory hit rate`
- `positive memory reuse count`
- `failure class recovery rate`

推荐对比：

```text
HLS-Eval zero-shot
HLS-Eval agentic baseline
CCD-HLS v2
CCD-HLS LOOP
CRAFT-HLS with contract + skill + budget + token report
CRAFT-HLS Local with deterministic repair + verified memory
```

## 13. 开发与提交前检查

提交前运行：

```bash
pytest -q
HLS-agent --help
git diff --check
git ls-files -co --exclude-standard -z | xargs -0 -r rg -n --hidden --no-heading 'sk-[A-Za-z0-9_-]{12,}|api_key\s*[:=]\s*"sk-|DEEPSEEK_API_KEY\s*=\s*sk-|OPENAI_API_KEY\s*=\s*sk-' || true
git status --short
```

不要提交：

```text
真实 API key
configs/*.local.json
configs/*_local.json
external/
experiments/
workspaces/
references/papers/*.pdf
references/web/*.html
```

## 14. 当前命名约定

- 项目/论文名：`CRAFT-HLS`
- 未来论文题目：`CRAFT-HLS: Contract-Refined Agentic Flow for Token-Efficient HLS Generation and Repair`
- CLI 命令：`HLS-agent`
- Python 包名：`ccd_hls_agent`
- 核心后端方法：`ccd_hls_loop`
- 历史方法名：`CCD-HLS v2 / LOOP`
