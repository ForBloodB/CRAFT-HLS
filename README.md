# CCD-HLS Agent

这是一个面向 HLS-Eval / FPT 2026 的 **TUI-first** HLS Agent 实验工程。当前版本已经移除 Web 前端和 FastAPI 后端，只保留命令行/TUI 演示、benchmark runner、HLS backend、模型接口、context contract 与实验结果回放能力。

## 1. 当前状态

- 推荐入口：`scripts/hls_tui.py`
- 核心方法：`ccd_hls_loop`，即 CCD-HLS v2 + 有界日志驱动 repair loop。
- 模型接口：OpenAI-compatible，本地 Qwen-Coder 与云端 DeepSeek 等均通过同一 `ModelConfig` 调用。
- HLS 接口：`vitis` backend 适配 Vitis 2025.2 的 `v++ -c --mode hls` 和 `vitis-run --mode hls --csim/--cosim`。
- API key 只从 `.env` 环境变量读取，不写入 README、SQLite 或实验结果。

## 2. 安装与配置

运行/TUI 依赖：

```bash
python3 -m pip install -e ".[dev]"
```

如需真实接入 HLS-Eval：

```bash
python3 -m pip install "git+https://github.com/sharc-lab/hls-eval.git"
```

DeepSeek 配置：

```bash
cp .env.example .env
```

填写 `.env`：

```text
DEEPSEEK_API_KEY=你的APIKey
HLS_EVAL_ROOT=/home/distortionk/WorkSpace/VCS/HLS-Agent/external/hls-eval
```

模型 profile 文件：

```text
configs/deepseek_v4_flash.json
```

默认使用：

```json
{
  "profile_name": "cloud_deepseek_v4_flash",
  "provider_type": "cloud_openai",
  "base_url": "https://api.deepseek.com",
  "api_key_env": "DEEPSEEK_API_KEY",
  "model": "deepseek-v4-flash",
  "temperature": 0.2,
  "max_tokens": 4096,
  "timeout": 90.0
}
```

## 3. TUI 使用

查看已有 run：

```bash
python scripts/hls_tui.py view \
  experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0
```

非交互快照：

```bash
python scripts/hls_tui.py view \
  experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0 \
  --snapshot
```

运行一个最多 2 次 LLM/API 调用的单 case：

```bash
python scripts/hls_tui.py run \
  --case-path external/hls-eval/hls_eval_data/machsuite/md_knn \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/deepseek_v4_flash.json \
  --env-file .env \
  --hls-backend vitis \
  --max-llm-calls 2
```

TUI 快捷键：

```text
Up/Down      切换阶段
Left/Right   切换当前阶段 artifact
PgUp/PgDn    滚动内容
r            重新读取 run 目录
q            退出
```

“两轮循环”的当前定义是：**最多 2 次 LLM/API 调用**。第 1 次 generation 计入调用次数；第 2 次可能是 format repair、CSIM repair 或 SYNTH repair，取决于第 1 次之后失败在哪个阶段。

## 4. 已跑的 TUI Demo

### md_knn：CSIM repair 两次调用

结果目录：

```text
experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0
```

阶段链路：

```text
GENERATION -> PARSE_VALIDATE -> CSIM failed -> CSIM_REPAIR -> PARSE_VALIDATE failed
```

两次 API 输入输出：

```text
Call 1 / GENERATION
input : experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0/llm_call_01_generation_prompt.txt
output: experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0/llm_call_01_generation_response.txt

Call 2 / CSIM_REPAIR
input : experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0/llm_call_02_csim_repair_attempt_1_prompt.txt
output: experiments/tui_demos/tui_demo_md_knn_loop2_deepseek_20260601_091529/ccd_hls_loop/md_knn/sample_0/llm_call_02_csim_repair_attempt_1_response.txt
```

关键指标：

```text
llm_calls_used = 2
repair_rounds = 1
can_parse = true
can_compile = false
can_pass_testbench = false
can_synthesize = false
stopped_reason = PATCH_OR_OUTPUT_PARSE_FAILED: No <OUTPUT_CODE> block found.; local secondary parse found no standalone C/C++ source.
```

说明：第 2 次 prompt 已经包含 CSIM 失败后的 failure capsule，但 DeepSeek 本次第 2 次返回正文为空，所以停在 `PARSE_VALIDATE`。

### parallel_merge_sort：format repair 非空输出

结果目录：

```text
experiments/tui_demos/tui_demo_parallel_merge_sort_loop2_deepseek_20260601_091529/ccd_hls_loop/parallel_merge_sort/sample_0
```

该 case 的第 2 次调用是 `generation_format_repair`，输出非空 `<OUTPUT_CODE>`，随后进入 CSIM；由于 `--max-llm-calls 2` 已用完，CSIM 失败后不再继续 repair。

## 5. 全量 Benchmark 与对比

全量 HLS-Eval zero-shot / agentic / CCD-HLS v2 / CCD-HLS LOOP 结果已整理到：

```text
experiments/full/
```

主要目录：

```text
experiments/full/full_deepseek_v4_flash_vitis_20260531_final
experiments/full/hls_eval_agentic_deepseek_94x1_20260531
experiments/full/full_ccd_hls_gen_v2_deepseek_20260531
experiments/full/full_ccd_hls_gen_v2_repair_deepseek_20260531_163049
```

全量运行示例：

```bash
python scripts/run_hls_eval_benchmark.py \
  --hls-eval-root external/hls-eval \
  --data-dir external/hls-eval/hls_eval_data \
  --model-config configs/deepseek_v4_flash.json \
  --methods ccd_hls_loop \
  --samples 1 \
  --hls-backend vitis \
  --max-llm-calls 5 \
  --out-dir experiments/full/my_ccd_hls_loop_run
```

COSIM 验证脚本：

```bash
python scripts/run_cosim_validation.py \
  --runs hls_eval_agentic,ccd_hls_loop \
  --out-dir experiments/cosim/my_cosim_run \
  --timeout-seconds 900 \
  --resume
```

## 6. 当前代码结构

```text
ccd_hls_agent/
  ccd.py                 上下文扫描、atomize、value/certainty scoring、context selection
  contracts.py           prompt contract 渲染入口
  failure_analysis.py    HLS 日志压缩、failure capsule、重复失败 early-stop
  hls_backends.py        HLS-Eval / Vitis / command / mock backend
  model_clients.py       本地/云端 OpenAI-compatible 模型客户端
  schemas.py             ModelConfig、Stage、AtomRecord 等核心数据结构
  json_utils.py          JSON 安全序列化
  utils.py               文件、token 估算、时间、env 等通用工具
  templates/             可编辑 prompt 模板

scripts/
  hls_tui.py                         TUI 演示与 artifact 回放入口
  run_hls_eval_benchmark.py          zero-shot / CCD-HLS v2 / CCD-HLS LOOP benchmark runner
  run_hls_eval_agentic_deepseek.py   HLS-Eval agentic DeepSeek 直连 runner
  run_cosim_validation.py            对已生成结果做 RTL cosim 验证

configs/
  deepseek_v4_flash.json             DeepSeek profile
  deepseek_v4_flash.example.json     示例 profile

docs/
  CCD_HLS_V2_LOOP_机制与数据对比.md   方法机制与实验数据对比
  FPT2026_HLS_Context_Agent_Design.md 研究设计文档

experiments/
  full/                              四组全量主结果
  cosim/                             cosim 验证结果
  tui_demos/                         TUI 两调用演示结果
  archive/                           旧 smoke/targeted/中间实验归档
```

## 7. 解耦状态

当前代码已经按功能边界拆开：

- TUI 只负责展示和启动单 case，不直接实现 HLS 逻辑。
- Benchmark runner 负责实验流程和 artifact 落盘。
- HLS backend 负责工具调用，和模型调用解耦。
- Model client 只负责 OpenAI-compatible API，不关心 HLS 阶段。
- Prompt 模板放在 `ccd_hls_agent/templates/`，可独立修改。
- Failure capsule 生成在 `failure_analysis.py`，可独立替换压缩策略。

仍可继续改进的点：

- `scripts/run_hls_eval_benchmark.py` 仍然偏大，后续可拆成 runner core、case IO、evaluation loop 三个模块。
- `schemas.py` 目前保留在单文件中，后续可拆成 `model_config.py` 和 `records.py`。
- 当前没有 Web 服务；交互统一走 TUI 和 artifact 文件。
