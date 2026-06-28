# FPT 2026 Track A 论文与资料清单

整理日期：2026-06-27

本目录保存面向 FPT 2026 Design Competition Track A 的本地论文资料索引。PDF 文件是 local-only 资料，默认被 `.gitignore` 排除，不随代码仓库提交；仓库只保留这份清单，便于后续按需重新下载或核对来源。

## 本地资料

| 文件 | 主题 | 对本项目的用途 |
|---|---|---|
| `hls-eval_2025_abikaram_hao.pdf` | HLS-Eval benchmark and framework | 比赛 Track A 的本地评测基线、任务接口、正确性指标来源 |
| `hlsfactory_2024_abikaram_hao.pdf` | HLS dataset construction framework | 构建可检索的 HLS 设计/pragma/QoR 数据池 |
| `scalehls_2021_ye_hao_chen.pdf` | MLIR-based scalable HLS framework | 提炼多层 IR、loop/array/function 分层分析思路 |
| `hls_perf_gnn_2022_wu_yang_xie_li_hao.pdf` | GNN-based HLS performance prediction | 后续 PPA proxy、QoR 预估与候选排序参考 |
| `hls_directives_llm4hls_2025_yao_zhao_sun_zhuo_yu.pdf` | LLM-based HLS directive optimization | 对照 LLM 作为 feature extractor / DSE agent 的 directive search 方法 |
| `prometheus_2025_pouget_lo_pouchet_cong.pdf` | Holistic FPGA accelerator optimization | 提炼 Prometheus 的 task fusion、tiling、loop permutation、compute-communication overlap、concurrent execution skill |
| `autodse_2021_sohrabizadeh_yu_cong.pdf` | Bottleneck-guided HLS DSE | 提炼 bottleneck-guided coordinate optimization，指导 PPA_OPT 阶段 |
| `hls_variable_loop_dse_2018_choi_cong.pdf` | Variable loop-bound HLS DSE | 提炼变量循环边界、低 PE utilization、directive 不足时的代码重写策略 |
| `hlsyn_2023_bai_sohrabizadeh_cong.pdf` | HLSyn benchmark | 提供 pragma-rich HLS 设计空间和 QoR 数据集参考 |
| `fifoadvisor_2025_abikaram_sarkar_basalama_cong_hao.pdf` | Automated FIFO sizing for HLS dataflow designs | 提炼 streaming/dataflow/FIFO depth/deadlock 相关 skill 和 PPA trade-off rule |
| `ralad_retrieval_augmented_hls_2024_xu_hu_huang.pdf` | Retrieval-augmented LLMs for HLS optimization | 作为 HLS-specific RAG baseline 与实现路线参考 |
| `hlspilot_2024_xiong_et_al.pdf` | LLM-based HLS generation | 提炼 C-to-HLS optimization strategy examples |
| `agentic_hls_reasoning_2025.pdf` | Agentic HLS with reasoning models | 对照 agentic DSE、ILP/tool feedback、reasoning model 使用方式 |
| `chathls_2025.pdf` | Multi-agent HLS debugging and directive tuning | 对照专业化 debugging agent、directive tuning agent 和 QoR-aware reasoning |
| `agentdiet_2025_xiao_gao_peng_xiong.pdf` | Agent trajectory reduction | 通用 token 节省 baseline：删除 useless/redundant/expired trajectory |
| `skillreducer_2026.pdf` | Skill token efficiency | skill progressive disclosure、routing description 压缩和 reference 按需加载参考 |

## 网页快照

| 文件 | 来源 | 用途 |
|---|---|---|
| `../web/fpt2026_design_competition_20260627.html` | <https://fpt2026.uark.edu/fpt26-design-competition/> | Track A 规则、任务形态、提交要求、截止日期快照 |
| `../web/hlsfactory_arxiv_2405_00820_20260627.html` | <https://arxiv.org/abs/2405.00820> | HLSFactory 在线元数据快照 |

## 后续可继续补充

| 资料方向 | 备注 |
|---|---|
| Stream-HLS / dataflow compiler papers | 与 FIFOAdvisor 联动，用于 dataflow/FIFO/deadlock skill |
| LightningSim / fast HLS simulation papers | 可作为低成本 tool feedback 与 FIFOAdvisor 的底层支撑 |
| More LLM4EDA/HLS repair papers | 用于补充 repair taxonomy 与 agent baseline |

## 资料转化原则

1. 论文不直接塞进 prompt。先离线提炼为短 skill、operator pattern、error taxonomy、pragma legality rule 和 QoR tradeoff rule。
2. skill 必须有短 routing description，正文采用 progressive disclosure：默认只加载 100-300 tokens 的 actionable rule，只有命中对应问题时再加载长案例。
3. 每条 skill 或检索样例都需要保存来源、适用条件、禁止条件、预期 PPA 影响和失败风险，进入 prompt 时以 compact atom 形式出现。
