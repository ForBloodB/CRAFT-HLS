# FPT 2026 Track A: CCD-HLS 参赛规划

生成日期：2026-06-27

本文面向 FPT 2026 Design Competition Track A: Budgeted End-to-End LLM4HLS Agent，说明当前项目已经满足了哪些题目要求，还缺什么，以及我们接下来如何围绕 token efficiency、skill/data library 和 HLS 优化能力组织参赛工作。

## 1. 题目要求摘要

比赛页面：<https://fpt2026.uark.edu/fpt26-design-competition/>

本地快照：`references/web/fpt2026_design_competition_20260627.html`

关键日期均为 23:59 AoE：

| 阶段 | 日期 | 当前距离 |
|---|---:|---:|
| Registration Deadline | 2026-07-07 | 10 天 |
| Submission Deadline | 2026-08-07 | 41 天 |
| Shortlist Announcement | 2026-08-21 | 55 天 |

Track A 要求开发一个 autonomous agent，在受限 tool invocation budget 下解决 HLS 任务。任务可能包括：

- 从正确但未优化的 C/C++ baseline 出发做 HLS 优化；
- 修复编译失败、综合失败的 HLS 设计；
- 修复能编译但 CSIM、COSIM 或 hidden tests 失败的设计；
- 修复 deadlock、invalid streaming、资源极差等结构问题；
- 处理其他 HLS compilation 问题。

agent 应展示完整 workflow：

- 解释 task specification 与 initial code；
- 生成或修改 HLS C/C++ code，包括 pragmas；
- 调用 evaluation interfaces；
- 解析 logs/reports 进行诊断；
- 优先解决 correctness，再做 PPA optimization；
- 在预算内终止。

提交材料要求 IEEE 双栏 PDF，正文不超过 2 页，appendix 不限。Track A 必须包含 agent workflow、agent operational process 和每个阶段 token consumption 的详细分析。最终评价维度包括 correctness、PPA metrics 和 problem difficulty；Final Stage 页面还给出 performance 80% 与 innovation 20% 的权重表述。

## 2. 当前项目已经做了什么

项目现状可以概括为：已经具备一个 budgeted、可回放、可统计 token 的 HLS-Eval agent 原型，但还没有完成面向比赛新任务形态的通用修复/优化 skill 系统，也缺少正式 PPA_OPT 与预算调度。

### 2.1 已实现系统能力

| 题目要求 | 当前项目状态 | 主要文件/结果 |
|---|---|---|
| 解释 task 与初始代码 | 已有 `scan_benchmark`、static atomize、top/header/testbench/description 输入 | `ccd_hls_agent/ccd.py` |
| 生成 HLS C/C++ implementation | 已有 CCD-HLS v2 单轮生成，输出 `<OUTPUT_CODE>` contract | `ccd_hls_agent/templates/ccd_hls_gen_v2.md` |
| 修改/修复候选代码 | 已有 FORMAT_REPAIR、CSIM_REPAIR、SYNTH_REPAIR 有界循环 | `scripts/run_hls_eval_benchmark.py` |
| 调用 HLS evaluation interfaces | 已支持 HLS-Eval backend 与 Vitis 2025.2 unified CLI；CSIM/SYNTH/COSIM backend 已有接口 | `ccd_hls_agent/hls_backends.py` |
| 解析 logs/reports | 已有 failure capsule、signal lines、error windows、report metrics parse | `ccd_hls_agent/failure_analysis.py` |
| correctness before PPA | 当前 LOOP 严格按 parse -> CSIM -> SYNTH，不做未通过 correctness 的 PPA | `docs/CCD_HLS_V2_LOOP_机制与数据对比.md` |
| 预算内终止 | 已有 `--max-llm-calls`、early-stop repeated failure、token 统计 | `scripts/run_hls_eval_benchmark.py` |
| token consumption 分析 | 已记录 prompt/completion/total tokens、每轮 prompt/response artifact | `experiments/full/*/summary.md` |
| 可演示/可回放 | 已有 TUI-first run/view/recent/doctor | `README.md`, `ccd_hls_agent/agent_tui.py` |

### 2.2 已有实验结果

现有 94-case HLS-Eval full benchmark 数据显示：

| 方法 | SYNTH 成功 | 平均 total tokens/case | 平均 LLM calls/case |
|---|---:|---:|---:|
| HLS-Eval zero-shot | 62/94, 65.96% | 3748.7 | 1.00 |
| HLS-Eval agentic | 74/94, 78.72% | 49299.9 | 7.97 |
| CCD-HLS v2 | 67/94, 71.28% | 3953.1 | 1.00 |
| CCD-HLS LOOP | 84/94, 89.36% | 8581.9 | 1.70 |

这说明当前 LOOP 已经抓住了比赛方向：它比 HLS-Eval agentic 更省 token，同时成功率更高。下一步不是把 prompt 写得更长，而是把更多诊断、检索、算子选择和历史压缩移到 deterministic 层。

## 3. 与比赛要求的主要差距

| 缺口 | 为什么重要 | 计划补齐方式 |
|---|---|---|
| 任务形态仍偏 HLS-Eval generation mode | 比赛可能给 baseline、坏代码、hidden tests、stream/deadlock、资源约束 | 增加 `task_mode` 分类：generate / repair_compile / repair_csim / repair_synth / repair_cosim / optimize_ppa |
| 工具预算模型不够细 | 题目可能限制 csim/cosim/synth 或 unified credit，不只是 LLM calls | 增加 `BudgetLedger`，分别记录 LLM、CSIM、COSIM、SYNTH、static tool、wall time |
| PPA_OPT 尚未正式闭环 | 评价包含 PPA metrics，当前 correctness 强但优化阶段弱 | 加入 correctness pass 后的 bounded pragma/code transform search |
| skill/data library 未落地 | 你的第二条 token 思路需要可检索的现成算子与方案 | 建 `hls_skills/` 与 `operator_memory/`，离线提炼论文和成功 case |
| log/report parser 还偏 regex | synthesis bottleneck、II/resource/timing 的因果定位不够 | 解析 `csynth.xml`、loop report、interface warning、dataflow warning，形成 HLS facts |
| RAG 仍是雏形 | 目前 `retrieve_patterns` 只按简单 term overlap | 增加 BM25/embedding 双路检索，检索对象为 skill、case diff、failure capsule、QoR record |
| token accounting 不够论文化 | 比赛要求每阶段 token consumption | 输出 `token_report.json/csv`，按 generation/format/csim repair/synth repair/ppa 分组 |
| hidden tests/COSIM 策略不足 | 比赛可能用 hidden functional tests 和 cosim | CSIM 通过后抽样运行 COSIM；失败时进入 COSIM_REPAIR；把 public test 覆盖不足作为风险 |
| problem difficulty 建模缺失 | 评价含 problem difficulty | 给每个 case 计算 difficulty tags：LOC、loops、arrays、stream、dataflow、pragma count、baseline failure type |

## 4. 我们的参赛主线

### 主线 A：通用 token 节省

你的当前“增量输入输出”已经是主线 A 的第一版：每一轮只把当前 code snapshot、failure capsule 和少量 failure history 给 LLM，而不是把完整轨迹和完整日志全部塞回 prompt。

下一步把它升级成可论文陈述的 Token-Efficient Workflow：

1. Stage-scoped prompting
   generation、format repair、CSIM repair、SYNTH repair、COSIM repair、PPA_OPT 使用不同模板和 token budget。

2. Failure capsule 结构化
   原始 stdout/stderr 冷存，只把 `failure_type`、`key_errors`、`signal_lines`、metrics tail 和相关 code scope 放入 prompt。

3. Context atom value scoring
   继续使用 `value_score / token_estimate`，但加入 tool evidence、scope graph distance、same-error historical success、supersededness。

4. Repeated failure early-stop
   当前已有相似 failure early-stop；要重新跑全量，量化它对成功率和 token 的影响。

5. AgentDiet baseline
   用 AgentDiet 风格 trajectory reduction 做通用 baseline，再证明 CCD-HLS 的 HLS-causal atom selection 更稳。

6. Completion token 控制
   继续使用单一 `<OUTPUT_CODE>` contract，禁止解释、JSON metadata、diff；repair prompt 明确“只返回完整 kernel”。

目标：保持或提升 84/94 SYNTH 成功，同时把 CCD-HLS LOOP 的平均 total tokens 从 8581.9 压到 7000 以下。

### 主线 B：skill 与数据供给

目标不是让 LLM 在 prompt 里“临场学习所有 HLS 知识”，而是给 agent 一个按需调用的 HLS skill/data library。

skill 类型：

| skill | 来源 | 默认注入内容 | 触发条件 |
|---|---|---|---|
| `hls_signature_repair` | HLS-Eval failures | header/top/testbench contract rule | compile error: not declared/no matching/undefined reference |
| `hls_data_file_runtime` | HLS-Eval failures | tb data file copy/check rule | couldn't open input data file |
| `hls_static_bounds` | HLS-Eval/ScaleHLS | static array/loop bound rule | dynamic allocation, variable-size arrays, unknown loop bounds |
| `hls_loop_pipeline_unroll` | HLSPilot, AutoDSE | pipeline/unroll basic rule and risk | SYNTH pass + loop bottleneck |
| `hls_array_partition` | HLSPilot, AutoDSE | memory port conflict -> partition rule | II limited by memory access |
| `hls_dataflow_stream` | Prometheus, HLS literature | dataflow legality, FIFO/deadlock guard | streaming/dataflow warning or task-level pipeline |
| `prometheus_tiling_fusion` | Prometheus | tiling/fusion/permutation/overlap/concurrency checklist | compute-memory bottleneck after correctness |
| `autodse_bottleneck` | AutoDSE | bottleneck-guided coordinate search | PPA_OPT candidate ranking |
| `variable_loop_bounds` | Choi & Cong 2018 | variable loop-bound transformation rule | loops with runtime bound and poor utilization |
| `rag_successful_case` | 本项目 experiments | closest successful code/failure/fix diff | same benchmark family/error class |

skill 设计采用 SkillReducer 的 progressive disclosure：

- routing description 小于 80 tokens；
- 默认 skill body 小于 300 tokens；
- 长案例、论文摘录、完整 code diff 只在命中时加载；
- 每条 skill 保存 `source_uri`、`applicability`、`anti_patterns`、`expected_effect`、`risk`。

## 5. 论文资料如何转成可用数据

本地论文目录：`references/papers/`

| 论文线 | 转化为 agent 能用的内容 |
|---|---|
| HLS-Eval / HLSFactory | benchmark metadata、case family、failure taxonomy、operator examples |
| ScaleHLS | function/loop/array/interface 多层 scope graph，支持更好的 context atom |
| Cong Hao GNN performance prediction | PPA proxy features：loops、arrays、pragma、estimated latency/resources |
| Prometheus | PPA_OPT 的高层 transform skill：fusion、tiling、permutation、overlap、concurrency |
| AutoDSE | bottleneck-guided coordinate optimizer，用于少量高价值 pragma 搜索 |
| Variable loop-bound DSE | runtime loop bounds 下避免盲目 pipeline/unroll 的代码重写策略 |
| HLSyn | pragma-rich QoR 数据，可训练/校准 PPA candidate ranker |
| RALAD | HLS RAG baseline：检索代码样例和优化 pattern，而非塞完整论文 |
| HLSPilot / ChatHLS / Agentic HLS | 对照多 agent/debug/directive tuning workflow |
| AgentDiet / SkillReducer | 通用 token 压缩与 skill token 压缩 baseline |

第一批离线提炼数据格式建议：

```json
{
  "skill_id": "prometheus_tiling_fusion",
  "route": "Use after CSIM/SYNTH pass when latency is dominated by repeated memory traffic or nested-loop computation.",
  "summary": "Consider tiling, loop permutation, task fusion, compute-communication overlap, and concurrent task execution as a coupled design space.",
  "applicability": ["nested loops", "array reuse", "memory bottleneck"],
  "anti_patterns": ["do not apply before functional correctness passes", "avoid changing public interface"],
  "expected_effect": {"latency": "down", "resource": "may increase"},
  "source_uri": "references/papers/prometheus_2025_pouget_lo_pouchet_cong.pdf"
}
```

## 6. 目标系统架构

```text
Task Ingest
  -> Task Mode Classifier
  -> Budget Ledger
  -> Static Analyzer / Scope Graph
  -> Tool Runner: CSIM / SYNTH / COSIM
  -> Report Parser / Failure Capsule Builder
  -> Skill Router
  -> Case/Operator Retriever
  -> Context Atom Scorer
  -> Prompt Assembler
  -> LLM Call
  -> Output Contract Validator
  -> Candidate Evaluator
  -> Memory Writer / Token Reporter
```

状态机：

```text
INIT
  -> STATIC_TRIAGE
  -> GENERATE_OR_REPAIR
  -> PARSE_VALIDATE
  -> CSIM_REPAIR*
  -> SYNTH_REPAIR*
  -> COSIM_REPAIR*
  -> PPA_OPT*
  -> FINAL_VALIDATE
  -> DONE / BUDGET_STOP / EARLY_STOP
```

优先级保持：

```text
parseability > compilability > public CSIM > synth > COSIM/hidden robustness > PPA
```

## 7. 工程里程碑

### M0：资料与规则冻结

交付物：

- `references/papers/README.md`
- `references/web/fpt2026_design_competition_20260627.html`
- 本文档

状态：已完成第一版。

### M1：可论文化 token report

交付物：

- `token_report.json/csv`
- 每 case 每阶段 token、LLM calls、tool calls、预算剩余；
- summary 增加 `tokens_by_stage`、`tokens_per_success`、`tool_calls_by_stage`。

验收：

- 能从 full benchmark 自动生成“每阶段 token consumption”表；
- 能支持比赛两页正文中的 workflow/token 图。

### M2：Task mode 与 BudgetLedger

交付物：

- task mode classifier；
- `BudgetLedger`；
- CLI 参数：`--llm-call-budget`、`--csim-budget`、`--synth-budget`、`--cosim-budget`、`--unified-credit-budget`。

验收：

- 可以模拟比赛不同 budget config；
- budget stop 原因可解释。

### M3：HLS skill/data library

交付物：

- `hls_skills/*.json` 或 `*.md`；
- `operator_memory/`：成功 case、失败 capsule、修复 diff、QoR facts；
- skill router 与检索器。

验收：

- repair prompt 默认不增长超过 15%；
- 命中 skill 的 case 能在 stage record 中看到 skill id、source、token cost。

### M4：Report parser 与 PPA proxy

交付物：

- 解析 `csynth.xml`、loop latency/II、resource、timing；
- memory port conflict、pipeline II violation、dataflow warning taxonomy；
- PPA candidate ranker。

验收：

- SYNTH pass 后能输出 top-3 PPA candidate；
- 每个 candidate 有 expected effect、risk、token cost、tool cost。

### M5：Bounded PPA_OPT

交付物：

- correctness pass 后最多 1-3 次 PPA transformation；
- candidate rollback；
- best-so-far selection by correctness + synth + latency/resource score。

验收：

- 在 HLS-Eval 94-case 中不降低 correctness；
- 对可优化 case 给出 latency/resource 改善统计。

### M6：正式实验矩阵

对照组：

1. HLS-Eval zero-shot；
2. HLS-Eval agentic；
3. CCD-HLS v2；
4. CCD-HLS LOOP；
5. CCD-HLS LOOP + early-stop rerun；
6. CCD-HLS + generic AgentDiet reduction；
7. CCD-HLS + skill/data library；
8. CCD-HLS + skill/data library + PPA_OPT。

指标：

- parse/compile/CSIM/SYNTH/COSIM pass；
- latency、estimated clock、LUT/FF/DSP/BRAM/URAM；
- total tokens、prompt tokens、completion tokens；
- tokens per successful design；
- LLM calls、CSIM/SYNTH/COSIM calls；
- budget stop、early stop、repair success rate；
- problem difficulty 分桶表现。

## 8. 两页论文建议结构

正文 2 页可按以下结构压缩：

1. Problem and workflow
   一张 FSM 图：STATIC_TRIAGE -> REPAIR -> VALIDATE -> PPA_OPT。

2. Method: Causal Context Diet + Skill Router
   说明 token 节省来自 failure capsule、context atom scoring、progressive skill disclosure。

3. Results
   表格对比 zero-shot / agentic / CCD-HLS LOOP / skill-enhanced CCD-HLS：success、PPA、tokens。

4. Ablation
   appendix 放 early-stop、skill router、RAG、PPA proxy 的消融。

5. Innovation
   强调不是单纯 prompt engineering，而是 HLS-causal context selection + budget ledger + skill/data library。

## 9. 当前最优先事项

1. 在 2026-07-07 前完成队伍注册材料和 Track A 方向确认。
2. 先做 M1 token report 和 M2 budget ledger，因为它们直接对应提交要求。
3. 然后做 M3 skill/data library，优先提炼 Prometheus、AutoDSE、HLS-Eval 成功/失败 case。
4. 最后做 M5 PPA_OPT；PPA 不应牺牲当前 84/94 的 correctness 基线。

## 10. 风险与应对

| 风险 | 应对 |
|---|---|
| 比赛任务不是 HLS-Eval generation，而是 repair/optimization | 增加 task mode classifier，保留原始 baseline，不强制 stub-out kernel |
| hidden tests 与 public test 不一致 | 代码生成约束更保守；优先遵守 spec/header/testbench；增加 metamorphic/static checks |
| PPA_OPT 破坏 correctness | best-so-far rollback，PPA only after CSIM/SYNTH/COSIM pass |
| skill 注入反而浪费 token | progressive disclosure，skill routing miss 时不加载正文 |
| LLM 输出不稳定 | 保持 `<OUTPUT_CODE>` contract、本地二次解析、format repair、temperature 低值 |
| Vitis 版本/工具链差异 | 在 artifact 中记录 tool version、part、clock、command；runner 支持 backend profile |

## 11. 一句话定位

CCD-HLS 的参赛定位是：在严格预算下，用 HLS-specific causal context selection 和可检索 skill/data library，把昂贵的“让 LLM 反复读日志和想优化策略”变成便宜、可解释、可统计的 deterministic workflow，只在最小必要上下文中调用 LLM 生成或修复代码。
