# Migration Verdict: pydantic-graph for Factory Workflow Engine

## Q1: Can BaseNode model factory node types cleanly?

**Verdict: YES — with one structural adaptation**

### Evidence

Phase 2 ported the study chain (`graph_update → study → graph_explorer → concat_study`)
as 4 `BaseNode` subclasses. Each maps directly from a current engine node type:

| Current Engine        | pydantic-graph                | Mapping              |
|-----------------------|-------------------------------|----------------------|
| `FnNode("graph_update")` | `GraphUpdateNode(BaseNode)` | 1:1, command in run() |
| `Study("study")`      | `StudyNode(BaseNode)`        | 1:1, command in run() |
| `AgentNode("graph_explorer")` | `GraphExplorerNode(BaseNode)` | 1:1, agent call in run() |
| `FnNode("concat_study")` | `ConcatStudyNode(BaseNode)` | 1:1, command in run() |

The model is clean: each node's `run()` method encapsulates what the current engine
dispatches to `_run_node()`, `_run_fn()`, or `_run_agent()`. State access moves from
executor instance variables to `ctx.state` (typed) and `ctx.deps` (immutable deps).

All 7 tests for the study chain pass, confirming execution order, state mutations,
file creation (dry_run mode), Mermaid output, and event recording.

### Adaptation required

`FnNode` and `AgentNode` become full Python classes (~30 lines each vs ~5 lines in
dict form). This is more verbose but gains type-checked state access and explicit
control flow.

---

## Q2: Does return-type routing work for GateNode verdicts?

**Verdict: YES — this is the biggest architectural gain**

### Evidence

Phase 3 implemented `GateBaseNode` with a typed return union:

```python
class QAGateNode(GateBaseNode):
    async def run(self, ctx) -> BuilderNode | NextNode | End[HaltResult]:
```

This replaces the current engine's runtime edge matching:

```python
# Current: string-based, runtime-only
Edge(source="qa_gate", target="builder", condition="reloop")
Edge(source="qa_gate", target="fork_qa", condition="proceed")
# Executor: _next_conditional(node_id, VerdictType.PROCEED)
```

**10 tests** verify all 3 verdict paths (PROCEED, RELOOP, HALT), iteration counting,
feedback injection, feedback updates across iterations, Mermaid branching topology,
and Graph.iter() event yields for both simple and reloop paths.

**Type safety verified:** If a gate's `run()` body tries to return a node not in its
return annotation, pyright (strict mode) catches it as a type error. The current
engine only catches this at runtime when no matching edge is found.

The comparison harness (Phase 5) confirmed that both engines produce identical gate
verdict sequences for PROCEED, RELOOP, HALT, and max-iteration-halt paths.

### Key finding

Return-type-as-edge eliminates the entire `_parse_agent_verdict` and
`_parse_fn_verdict` code paths (102 lines in executor.py). Verdict routing is
declared in the type system, not parsed from strings.

---

## Q3: Does FactoryState + FactoryDeps carry context effectively?

**Verdict: YES — cleaner than the executor's instance variables**

### Evidence

The current engine spreads state across `WorkflowExecutor` instance variables:

```python
class WorkflowExecutor:
    self.completed_files: set[str]
    self.node_context: dict[str, str]      # feedback
    self.iteration_counts: dict[tuple[str, str], int]
    self.result.node_outputs: dict[str, str]
    self.result.events: list[dict]
```

pydantic-graph consolidates these into two typed dataclasses:

```python
@dataclass
class FactoryState:          # mutable, threaded via ctx.state
    iteration_counts: dict[tuple[str, str], int]
    node_feedback: dict[str, str]
    node_outputs: dict[str, str]
    completed_files: set[str]
    events: list[dict[str, Any]]

@dataclass
class FactoryDeps:           # immutable, injected via ctx.deps
    project_path: Path
    dry_run: bool
    event_emitter: Callable
```

Every node across all 4 phases successfully reads and writes state through
`ctx.state` and `ctx.deps`. Cross-node communication patterns verified:

- **Sequential state threading:** Each study chain node reads/writes
  `ctx.state.node_outputs` (Phase 2, 7 tests)
- **Gate iteration tracking:** `ctx.state.iteration_counts[(gate_id, target_id)]`
  increments correctly across RELOOP cycles (Phase 3, 10 tests)
- **Feedback injection:** Gates write to `ctx.state.node_feedback[target_id]`,
  reloop targets read it on re-entry (Phase 3, verified explicitly)
- **Fork/join state sharing:** All 3 QA children write to the same `ctx.state`
  concurrently without races (Phase 4, 10 tests)

### Advantage over current engine

State access is type-checked by pyright. In the current engine,
`self.node_context[target]` is an untyped dict access — typos in key names are
silent bugs. In pydantic-graph, `ctx.state.node_feedback[target_id]` is checked
against the `FactoryState` dataclass definition.

---

## Q4: Mermaid + Graph.iter() as bonus wins?

**Verdict: YES — significant observability gains**

### Mermaid

`Graph.render()` auto-generates valid Mermaid state diagrams from the type-inferred
topology. No configuration, no manual diagram maintenance.

Generated diagrams verified across all phases:
- Study chain: linear 4-node topology with `[*]` terminal (Phase 2)
- Gate routing: `<<choice>>` decision node showing PROCEED/RELOOP/HALT edges (Phase 3)
- Fork/join: single ForkJoinNode (internal parallelism not visible) (Phase 4)
- Comparison workflow: full builder→gate→fork pattern (Phase 5)

The current engine has **no built-in visualization**. `validate_graph()` uses
NetworkX for structural validation but produces no visual output.

**Limitation:** ForkJoinNode renders as a single node. The 3-branch fan-out is not
visible in the diagram. A custom Mermaid post-processor could expand it, but this
is acceptable for the prototype.

### Graph.iter()

`Graph.iter()` provides step-by-step async iteration over graph execution:

```python
async with graph.iter(state=state, deps=deps, inputs=start) as run:
    async for event in run:
        # observe each step in real time
```

Verified in all phases: correct event counts, EndMarker as terminal event, one
event per node step. This enables real-time observability that the current engine
achieves only through its event emission system.

---

## Q5: Can definitions.py construction patterns adapt?

**Verdict: YES — but the adaptation is structural, not syntactic**

### Evidence

The comparison harness (Phase 5) defines the identical workflow in both representations:

**Current engine pattern** (`build_current_engine_workflow()`):
```python
nodes = {
    "builder": SimNode(id="builder", kind=NodeKind.ACTION),
    "qa_gate": SimNode(id="qa_gate", kind=NodeKind.GATE),
    "fork_qa": SimNode(id="fork_qa", kind=NodeKind.FORK,
                        fork_targets=["health_checker", ...]),
    ...
}
edges = [
    SimEdge(source="builder", target="qa_gate"),
    SimEdge(source="qa_gate", target="fork_qa", condition="proceed"),
    SimEdge(source="qa_gate", target="builder", condition="reloop"),
    ...
]
```

**pydantic-graph pattern** (`build_pydantic_graph()`):
```python
class CompareBuilderNode(BaseNode):
    async def run(self, ctx) -> "CompareQAGateNode":
        ...

class CompareQAGateNode(GateBaseNode):
    async def run(self, ctx) -> CompareBuilderNode | CompareQAForkJoinNode | End:
        ...

class CompareQAForkJoinNode(ForkJoinNode):
    async def run(self, ctx) -> End[HaltResult]:
        await self.execute_children(ctx)
        ...
```

**The adaptation:**
- `dict[str, NodeType]` → class hierarchy (each node type is a class)
- `list[Edge]` → return type annotations (edges are implicit)
- `ForkNode + JoinNode` → single `ForkJoinNode` (asyncio.gather inside `run()`)
- `GateNode + edge conditions` → return union type (pyright validates)
- String-based node IDs → Python class references (import-time validation)

### The 18 new comparison tests confirm behavioral equivalence

Both representations produce:
- Same active nodes for PROCEED, RELOOP, and HALT paths
- Same gate verdict sequences
- Same fork/join children sets
- Same max-iteration halt behavior

The construction pattern changes significantly (data-driven → type-driven), but the
execution semantics are preserved.

---

## Recommended Migration Path

### Phase 1: Core Workflows (Recommended — start here)

Port `improve_workflow` first — it exercises all 3 patterns (sequential study chain,
gate verdicts, and deep-QA fork/join) and is the most frequently run workflow.

| Workflow         | Complexity | Patterns Used             | Priority |
|-----------------|:----------:|---------------------------|:--------:|
| improve         |   Medium   | Sequential + Gate + Fork  |   High   |
| build           |   Medium   | Sequential + Gate + Fork  |   High   |
| discover        |    Low     | Sequential only           |  Medium  |
| review          |    Low     | Sequential only           |  Medium  |
| research        |   Medium   | Sequential + Gate + Fork  |  Medium  |

### Phase 2: Parallel Execution (Keep custom)

`SubgraphForkNode` and `SelectionNode` have no pydantic-graph equivalent and should
remain custom implementations. These can wrap pydantic-graph subgraphs internally
(each worktree branch runs a `Graph` instance).

### Phase 3: Advanced Nodes (Keep custom)

`LLMNode` (in-process API tool-use loop) is orthogonal to graph execution and should
remain a separate concern. It can be wrapped as a `BaseNode` subclass that delegates
to the existing `run_llm_loop()` function.

### What to keep custom

| Component              | Reason                                                  |
|------------------------|---------------------------------------------------------|
| SubgraphForkNode       | Worktree isolation is unique to the factory             |
| SelectionNode          | Branch comparison logic is domain-specific              |
| LLMNode                | In-process API loop, orthogonal to graph execution      |
| `reads`/`writes` deps  | File-polling replaced by typed state; migration needed  |
| Agent invocation        | `invoke_agent` subprocess management stays as-is        |

### Estimated Complexity

| Task                                   | Effort   | Risk  |
|----------------------------------------|----------|-------|
| Port improve_workflow nodes            | 2-3 days | Low   |
| Port build_workflow nodes              | 2-3 days | Low   |
| Replace executor for ported workflows  | 1-2 days | Medium |
| Migrate `reads`/`writes` to state      | 3-5 days | High  |
| Integrate SubgraphFork with PG graphs  | 2-3 days | Medium |
| Remove legacy executor (after full port)| 1 day   | Low   |

### Risk Assessment

**Low risk:**
- Sequential chains and gate routing are fully validated (34 tests across Phases 1-4)
- asyncio.gather fork/join pattern is identical to current implementation

**Medium risk:**
- `reads`/`writes` file dependency polling has no pydantic-graph equivalent — the
  migration to typed state requires careful analysis of which files are used as
  inter-node communication channels vs. which are external artifacts

**Mitigated risk:**
- `from __future__ import annotations` interaction tested and validated in Phase 1
  (4 dedicated tests) — this was the top-flagged risk from research and it passed

### Recommendation

**Proceed with migration using the thin-adapter approach:**

1. Create `BaseNode` subclass wrappers for existing node types (`AgentNode`, `FnNode`,
   `Study`, `GateNode`)
2. Port one workflow at a time, starting with `improve_workflow`
3. Run both engines in parallel during migration (current engine as fallback)
4. Keep `SubgraphForkNode`, `SelectionNode`, and `LLMNode` as custom extensions
5. Deprecate `executor.py` after all core workflows are ported

The prototype demonstrates that pydantic-graph eliminates 74% of infrastructure code
while providing compile-time edge validation, auto-generated Mermaid diagrams, and
step-by-step execution iteration. The tradeoff is 2x more verbose node definitions,
which is acceptable given the type safety gains.
