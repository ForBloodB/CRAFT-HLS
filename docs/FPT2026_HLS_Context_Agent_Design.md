# FPT 2026 HLS-Eval Token-Efficient Agent Design

## 1. Goal

Design an HLS code-generation agent for the FPT 2026 Track A setting that solves HLS-Eval tasks with fewer LLM tokens by moving most context selection, log interpretation, history compression, and workflow control into deterministic computation.

The agent is optimized for:

- Correctness first: parseability, compilability, runnability, and synthesizability before PPA tuning.
- Low token cost: most calls should use compact prompts under 4K-8K tokens.
- Durable memory: every run writes structured facts, raw artifacts, selected context, dropped context, and token usage.
- Comparability with AgentDiet: support a baseline where generic trajectory reduction is used instead of HLS-specific causal value scoring.

Recommended working name: **CCD-HLS**, short for **Causal Context Diet for HLS Agents**.

## 2. Evidence and Motivation

This design is based on the following observations.

1. FPT 2026 Track A asks for an LLM agent that can generate HLS code under strict tool budgets, report its workflow, and report token consumption. The competition explicitly separates correctness from PPA optimization.
   Source: <https://fpt2026.uark.edu/fpt26-design-competition/>

2. HLS-Eval provides a natural benchmark for HLS code generation and editing, with Vitis HLS integration and staged metrics such as parseability, compilability, runnability, synthesizability, and pass@k.
   Sources: <https://arxiv.org/abs/2504.12268>, <https://github.com/sharc-lab/hls-eval>

3. AgentDiet reduces LLM-agent trajectory cost by removing useless, redundant, and expired trajectory content using sliding-window reflection. This is a strong generic baseline, but it does not model HLS-specific causality among code scopes, pragmas, tool reports, and QoR outcomes.
   Source: <https://arxiv.org/abs/2509.23586>

4. Program slicing provides the theoretical basis for preserving information that can affect a target program point. CCD-HLS adapts this idea to HLS decision frontiers: top function, failing loop, failing interface, bottleneck array, or current PPA objective.
   Source: <https://www.pls-lab.org/en/Program_slicing>

5. Qwen3-Coder-30B-A3B-Instruct is a practical local test model because it is a MoE coding model with 30.5B total parameters and about 3.3B activated parameters.
   Source: <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>

## 3. Core Difference from AgentDiet

AgentDiet compresses generic agent trajectories. CCD-HLS compresses HLS optimization histories using structured causality and tool evidence before an LLM call is made.

| Aspect | AgentDiet baseline | CCD-HLS proposed design |
|---|---|---|
| Compression target | Generic trajectory messages | HLS code, reports, logs, diffs, decisions, QoR facts |
| Main mechanism | Sliding-window reflection over previous trajectory | Deterministic parsing, graph distance, evidence scoring, value-per-token selection |
| Domain knowledge | Mostly task-agnostic | HLS-specific scopes: function, loop, array, interface, pragma, report metric |
| Certainty source | LLM/reflection summary | Tool report > compiler/csim error > static analysis > retrieved pattern > LLM hypothesis |
| Risk control | Summary quality depends on reflecting model | Raw artifacts are cold-stored; prompt facts carry evidence pointers |
| Reproducibility | Depends on trajectory reducer | Every selected/dropped atom has a numeric score and reason |
| Best use as baseline | Cost-reduction baseline | HLS-specialized method to beat or complement AgentDiet |

The comparison experiment should include:

- Full-history ReAct.
- AgentDiet-style trajectory reduction.
- CCD-HLS deterministic context scoring only.
- CCD-HLS scoring plus local RAG.
- CCD-HLS scoring plus RAG plus PPA proxy ranking.

## 4. System Architecture

```text
Task Ingest
  -> Stage Controller
  -> Static Analyzer
  -> Artifact Parser
  -> Context Atomizer
  -> Local Memory Store
  -> Local RAG Retriever
  -> Context Value Scorer
  -> Prompt Assembler
  -> Qwen3-Coder-30B-A3B
  -> Patch Validator
  -> HLS-Eval Runner
  -> Memory Writer
  -> Client Event Reporter
```

The LLM is only called by the Prompt Assembler. All previous blocks should run without LLM inference.

## 5. Agent Stages

Use a finite-state controller rather than a free-form planner.

```text
INIT
  Parse task, testbench, top signature, constraints.

STATIC_FIX
  Fix deterministic issues before calling Vitis HLS:
  missing includes, signature mismatch, unsupported constructs, obvious type errors.

CSIM_REPAIR
  Repair functional correctness based on compile/csim output.

SYNTH_REPAIR
  Repair synthesizability:
  unsupported language features, invalid pragmas, interface issues, loop bounds, memory issues.

COSIM_REPAIR
  Repair mismatch between C simulation and RTL/cosim if available.

PPA_OPT
  Only entered after correctness and synthesis pass.
  Try bounded pragma/code transformations.

DONE
  Emit final artifact, metrics, token report, and memory summary.
```

Stage priority:

```text
parseability > compilability > runnability > synthesizability > PPA
```

## 6. Deterministic Pre-LLM Execution Path

For every iteration, run this path before any LLM call.

### 6.1 Read Current State

Inputs:

- Current code snapshot.
- HLS-Eval task metadata.
- Latest tool result.
- Remaining tool budget.
- Remaining token budget.
- Current best candidate.

Outputs:

- `current_frontier`: the smallest decision target, such as `top/signature`, `loop:L3`, `array:A`, `interface:m_axi`, or `stage:PPA_latency`.

### 6.2 Parse and Normalize Artifacts

Parse:

- C/C++ code into function, loop, array, call, and pragma scopes.
- Compiler errors into `error_class`, `file`, `line`, `symbol`, and `message`.
- HLS logs into warning/error facts.
- Synthesis reports into latency, II, resource, timing, and pragma validity facts.
- Diffs into intervention facts.

No raw log should be passed to the LLM unless parsing fails or the error class is unknown.

### 6.3 Create Context Atoms

Every piece of reusable information becomes an atom.

```json
{
  "atom_id": "atom_000173",
  "task_id": "hls_eval_xxx",
  "run_id": "run_0009",
  "kind": "synth_bottleneck",
  "scope": "top/foo/loop_L3/array_A",
  "stage": "SYNTH_REPAIR",
  "summary": "Pipeline II=1 failed; actual II=4 due to memory port conflict on A.",
  "evidence_uri": "runs/run_0009/csynth.json#loop_L3",
  "code_hash": "sha256:...",
  "status": "active",
  "token_estimate": 28,
  "created_at_step": 9
}
```

Atom kinds:

```text
task_requirement
code_scope
testbench_fact
compile_error
csim_failure
synth_error
synth_warning
synth_metric
pragma_fact
intervention
decision
hypothesis
retrieved_pattern
final_success
```

### 6.4 Build HLS Context Graph

Nodes:

```text
task, code_snapshot, function, loop, array, interface, pragma,
tool_error, warning, metric, decision, intervention, hypothesis
```

Edges:

```text
contains(function, loop)
uses(loop, array)
calls(function, function)
annotates(pragma, loop/function/array/interface)
caused_by(metric/error, intervention)
supersedes(code_snapshot_new, code_snapshot_old)
invalidates(error/fact, hypothesis)
supports(report_fact, decision)
```

This graph is the basis for deterministic relevance calculation.

### 6.5 Score Context Value

Compute a value score for each atom before retrieval or prompting.

```text
Value(i) =
  0.24 * causal_relevance(i)
+ 0.20 * blocker_relevance(i)
+ 0.16 * evidence_strength(i)
+ 0.12 * expected_qor_impact(i)
+ 0.10 * retrieval_similarity(i)
+ 0.08 * uncertainty_need(i)
+ 0.06 * recency(i)
+ 0.04 * user_or_rule_priority(i)
- 0.16 * redundancy(i)
- 0.14 * supersededness(i)
- 0.08 * token_cost_norm(i)
```

Definitions:

```text
causal_relevance(i) = exp(- graph_distance(i.scope, current_frontier) / tau)

blocker_relevance(i) =
  1.0 if atom directly explains current failure
  0.7 if atom explains same error class in same task
  0.5 if atom explains same scope in old run
  0.2 otherwise

evidence_strength =
  0.95 for parsed synthesis report
  0.90 for compiler/csim exact error
  0.80 for deterministic static analysis
  0.70 for repeated historical success in same benchmark family
  0.60 for retrieved pattern
  0.35 for LLM hypothesis without tool evidence

expected_qor_impact =
  normalized absolute delta in latency, II, resource, or timing

uncertainty_need =
  1 - certainty

recency =
  exp(- age_steps / half_life)

redundancy =
  max semantic similarity to already selected atoms

supersededness =
  1.0 if atom refers only to obsolete code with no active causal path
  0.5 if atom refers to old code but same scope remains
  0.0 if atom matches current code hash or active scope
```

Certainty:

```text
Certainty(i) =
  0.35 * evidence_strength(i)
+ 0.25 * current_code_compatibility(i)
+ 0.20 * reproduction_count_norm(i)
+ 0.10 * parser_confidence(i)
+ 0.10 * status_stability(i)
- 0.40 * contradiction(i)
```

### 6.6 Compress by Certainty and Causality

Use a four-quadrant policy.

| Class | Condition | Prompt treatment |
|---|---|---|
| Critical unresolved | high causality, low certainty | Keep evidence excerpt or rerun deterministic parser/tool |
| Stable causal fact | high causality, high certainty | Keep compact structured fact |
| Archived known fact | low causality, high certainty | Keep only index or omit from prompt |
| Low-value noise | low causality, low certainty | Cold-store only |

### 6.7 Retrieve from Local RAG

RAG should retrieve patterns, not long documents.

```text
RAGScore =
  0.30 * BM25(error identifiers, symbols, pragma names)
+ 0.25 * embedding_similarity(code/error summary)
+ 0.20 * graph_scope_match
+ 0.15 * stage_match
+ 0.10 * historical_success
- 0.10 * token_cost_norm
```

Use two indexes:

- Sparse index: BM25 over symbols, error classes, pragma names, function names.
- Dense index: embeddings over compact summaries and code slices.

Prevent leakage:

- Do not index HLS-Eval reference implementations for test tasks.
- Mark memory source as `train`, `dev`, `self_generated`, or `external_doc`.
- During evaluation, only retrieve allowed sources.

### 6.8 Select Context Under Token Budget

Prompt context is selected by a bounded knapsack-like rule.

Always include:

- Task capsule.
- Current failing code slice.
- Current frontier.
- Latest blocker.
- Output contract.

Then greedily add atoms by:

```text
priority(i) = Value(i) / max(8, token_estimate(i))
```

Stop at stage-specific budgets:

```text
INIT generation:      6000-8000 prompt tokens
STATIC_FIX:           2500-4000 prompt tokens
CSIM_REPAIR:          3000-5000 prompt tokens
SYNTH_REPAIR:         3500-6000 prompt tokens
PPA_OPT:              4000-7000 prompt tokens
emergency debug:      max 12000 prompt tokens
```

## 7. LLM Prompt Contract

The LLM should never receive an open-ended "think and solve" prompt. It receives a narrow patch contract.

Template:

```text
You are editing HLS C/C++ for Vitis HLS.

Stage:
{stage}

Goal:
{goal}

Task capsule:
{task_capsule}

Current frontier:
{current_frontier}

Current code slice:
{code_slice}

Selected facts:
{selected_atoms_as_bullets}

Allowed actions:
{allowed_actions}

Forbidden actions:
{forbidden_actions}

Return only:
1. A unified diff.
2. A compact JSON metadata block.
No long explanation.
```

Expected metadata:

```json
{
  "stage": "SYNTH_REPAIR",
  "intended_fix": "partition array A to resolve loop_L3 memory port conflict",
  "touched_scopes": ["loop_L3", "array_A"],
  "expected_effect": {"II": "decrease", "BRAM": "increase"},
  "risk": "medium"
}
```

Generation settings for Qwen3-Coder-30B-A3B:

```text
temperature: 0.1-0.3 for repair
temperature: 0.2-0.5 for initial generation
max_new_tokens: 1024-4096
top_p: 0.8-0.95
```

## 8. Patch Validation Before Tool Calls

After LLM output:

1. Check diff parses.
2. Apply to a temporary workspace.
3. Run formatting if configured.
4. Re-run static analyzer.
5. Reject patch without HLS-Eval tool call if it:
   - Deletes required top function.
   - Changes testbench-visible signature incorrectly.
   - Introduces unsupported dynamic allocation.
   - Adds forbidden libraries.
   - Produces syntax errors detectable locally.

This saves both HLS tool budget and future token budget.

## 9. PPA Optimization Policy

Only enter PPA after correctness and synthesis pass.

Candidate actions:

```text
pipeline(loop)
unroll(loop, factor)
array_partition(array, cyclic/block/complete, factor)
dataflow(function)
inline(function)
bind_storage(array)
bind_op(operation)
loop_flatten(loop)
```

Deterministic candidate ranking:

```text
PPAActionScore(a) =
  0.30 * estimated_latency_gain(a)
+ 0.20 * bottleneck_match(a)
+ 0.15 * legality_score(a)
+ 0.15 * past_success_same_pattern(a)
+ 0.10 * resource_headroom(a)
+ 0.10 * low_risk_score(a)
- 0.20 * estimated_resource_cost(a)
- 0.10 * synthesis_budget_cost(a)
```

Run only top-k actions per round, where:

```text
k = min(3, remaining_synth_budget)
```

If two actions affect the same causal scope, test them separately before combining unless previous memory shows the pair is safe.

## 10. Memory Layout

Recommended first implementation:

```text
.hls_agent/
  memory.sqlite
  vectors/
    dense.faiss
    bm25.json
  runs/
    run_0001/
      code.cpp
      patch.diff
      compile.log
      csim.log
      csynth.rpt
      parsed_facts.json
      prompt.txt
      response.txt
      selected_atoms.json
      dropped_atoms.json
```

SQLite tables:

```sql
task(task_id, source, split, prompt_hash, top_function, created_at)
run(run_id, task_id, stage, code_hash, parent_run_id, status, created_at)
artifact(artifact_id, run_id, kind, uri, sha256)
atom(atom_id, task_id, run_id, kind, scope, stage, summary, evidence_uri,
     code_hash, status, token_estimate, value_score, certainty_score)
edge(src_id, dst_id, edge_type, weight)
decision(decision_id, run_id, action, touched_scopes, reason_atom_ids, result)
metric(run_id, metric_name, metric_value, scope)
llm_call(call_id, run_id, model, prompt_tokens, completion_tokens, selected_atom_ids)
```

## 11. Client Interaction Layer

The client should stream concise events. Users should see the agent's state without reading logs.

Event schema:

```json
{
  "step": 7,
  "stage": "SYNTH_REPAIR",
  "status": "context_selected",
  "message": "64 atoms scored; 9 selected; prompt estimate 3860 tokens.",
  "frontier": "loop_L3/array_A",
  "latest_blocker": "memory port conflict",
  "selected_context_count": 9,
  "dropped_context_count": 55,
  "prompt_tokens_est": 3860,
  "remaining_tool_budget": {"csim": 3, "synth": 2},
  "next_action": "call_llm_for_patch"
}
```

User-facing step messages:

```text
[1/8] Task ingest: top=foo, arrays=3, loops=5.
[2/8] Static scan: no deterministic signature fix available.
[3/8] Latest blocker: synthesis failed at loop_L3 due to memory port conflict.
[4/8] Context scoring: 64 atoms -> 9 selected, 55 cold-stored.
[5/8] RAG: retrieved 2 successful array partition patterns from allowed memory.
[6/8] LLM call: generating a minimal patch for loop_L3/array_A.
[7/8] Verification: csim passed, synthesis running.
[8/8] Memory: run_0010 written, token usage updated.
```

The client should also expose:

```text
current_stage
current_frontier
best_candidate
latest_blocker
selected_atoms_with_scores
dropped_atoms_summary
token_usage_by_stage
tool_usage_by_stage
next_action
confidence
```

## 12. End-to-End Loop Pseudocode

```python
def solve_task(task):
    state = init_state(task)
    memory.write_task(task)

    while not state.done and state.budget.remaining():
        emit("stage_started", state.summary())

        artifacts = load_current_artifacts(state)
        parsed = parse_artifacts(artifacts)
        static_result = run_static_analyzer(state.code, task)
        atoms = atomize(task, state, parsed, static_result)
        graph = update_context_graph(memory, atoms)
        memory.write_atoms(atoms, graph)

        state.frontier = choose_frontier(parsed, static_result, state)
        emit("frontier_selected", state.frontier)

        deterministic_patch = try_deterministic_patch(static_result, state)
        if deterministic_patch:
            result = validate_and_run(deterministic_patch, state)
            memory.write_run_result(result)
            state = update_state(result)
            continue

        candidates = retrieve_rag(memory, state.frontier, state.stage)
        scored_atoms = score_atoms(memory, atoms, candidates, state.frontier)
        selected, dropped = select_context(scored_atoms, state.token_budget)
        memory.write_context_selection(state.run_id, selected, dropped)
        emit("context_selected", context_summary(selected, dropped))

        prompt = assemble_prompt(task, state, selected, candidates)
        response = call_llm(prompt, model="Qwen3-Coder-30B-A3B-Instruct")
        memory.write_llm_call(prompt, response)

        patch = parse_patch(response)
        patch_status = validate_patch(patch, state)
        if not patch_status.accepted:
            memory.write_rejected_patch(patch_status)
            state = update_after_rejection(state, patch_status)
            continue

        result = run_hls_eval(patch, state)
        memory.write_run_result(result)
        state = update_state(result)

    return finalize(state, memory)
```

## 13. Evaluation Plan

Compare the following systems on the same HLS-Eval split and tool budget.

| System | Description |
|---|---|
| Full-History | Pass all previous messages/log summaries into the LLM until context limit |
| AgentDiet | Apply AgentDiet-style trajectory reduction to previous messages |
| RAG-Only | Retrieve local examples, no HLS-specific value scoring |
| CCD-HLS-NoRAG | Use deterministic atom scoring and compression only |
| CCD-HLS-RAG | Use atom scoring plus local RAG |
| CCD-HLS-RAG-PPA | Add deterministic PPA action ranking after correctness |

Primary metrics:

```text
parseability
compilability
runnability
synthesizability
pass@1 / pass@k
total prompt tokens
total completion tokens
LLM calls per solved task
tool calls per solved task
wall-clock time
```

PPA metrics:

```text
latency
II
LUT
FF
DSP
BRAM
URAM
timing slack
Pareto dominance count
```

Context metrics:

```text
compression ratio = selected_tokens / raw_history_tokens
selection precision = selected atoms later referenced by decisions / selected atoms
dropped regret = dropped atoms later needed for successful fix / dropped atoms
stable fact reuse = reused high-certainty atoms / total selected atoms
LLM avoidance rate = deterministic patches / total patches
```

Expected research claim:

```text
CCD-HLS should match or improve AgentDiet's token reduction while preserving more HLS-causal facts,
especially facts involving pragma decisions, memory bottlenecks, resource tradeoffs, and synthesis failures.
```

## 14. Implementation Milestones

Milestone 1: Minimal HLS-Eval runner integration.

- Run one task.
- Capture code, logs, reports, pass/fail stage.
- Store artifacts in `.hls_agent/runs`.

Milestone 2: Atom store and deterministic log parser.

- Parse compiler/csim/synth logs into atoms.
- Parse code scopes and pragmas.
- Write SQLite memory.

Milestone 3: Context scoring and prompt assembly.

- Implement value score.
- Implement four-quadrant compression.
- Emit `selected_atoms.json` and `dropped_atoms.json`.

Milestone 4: Qwen patch loop.

- Generate unified diffs.
- Validate patches before tool calls.
- Track token usage per stage.

Milestone 5: Local RAG.

- Add sparse and dense retrieval.
- Enforce no reference leakage.
- Store successful patterns.

Milestone 6: AgentDiet comparison.

- Implement a baseline reducer that operates only on trajectory messages.
- Run identical HLS-Eval tasks and budgets.
- Report solve rate, PPA, token cost, and context metrics.

Milestone 7: PPA optimizer.

- Add candidate pragma generation.
- Rank using deterministic PPAActionScore.
- Run bounded top-k synthesis experiments.

## 15. First-Version Defaults

Use these defaults unless experiments suggest otherwise.

```text
model: Qwen3-Coder-30B-A3B-Instruct
normal_prompt_budget: 6000 tokens
repair_prompt_budget: 4000 tokens
ppa_prompt_budget: 6000 tokens
emergency_prompt_budget: 12000 tokens
temperature_repair: 0.2
temperature_initial: 0.4
max_new_tokens_repair: 2048
max_new_tokens_initial: 4096
context_half_life_steps: 6
graph_distance_tau: 2.0
max_selected_atoms_repair: 12
max_rag_patterns: 3
```

## 16. What to Report in the Paper

The method section should present:

1. HLS context atomization.
2. HLS context graph.
3. Certainty and value scoring.
4. Token-budgeted context selection.
5. Local RAG with leakage control.
6. Deterministic-first execution loop.
7. User-visible workflow reporting.

The comparison section should emphasize:

- AgentDiet is a generic trajectory reducer.
- CCD-HLS is a domain-causal context selector.
- Both reduce tokens, but CCD-HLS is expected to preserve long-range HLS dependencies better, such as old pragma decisions that still affect current II/resource tradeoffs.

