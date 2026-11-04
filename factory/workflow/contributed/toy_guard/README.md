# Toy Guard Workflow

Deterministic end-to-end fixture for agent guardrails (INT-002). A two-node mode
(`draft` → `publish`) that is **composed as a `Package` and compiled**, so the
generated guard checker stage is actually expanded by
`factory.workflow.guardrails.expand_agent_guards`.

## Graph

```
draft (AgentNode, 2 guards, reloop budget 2)
  │
  ▼
guard_draft_reset (FnNode)          # fresh token + rules.json + stale-verdict cleanup
  │
  ▼
guard_draft_fork (ForkNode)
  ├── guard_draft_draft_md_must_state_status_ready (AgentNode checker)
  └── guard_draft_draft_md_must_not_contain_tbd     (AgentNode checker)
  │
  ▼
guard_draft_join (JoinNode)
  │
  ▼
guard_draft_gate (GateNode, fn evaluator)
  ├── PROCEED ──▶ publish (AgentNode)
  └── RELOOP  ──▶ draft
```

8 nodes / 6 edges: the two authored nodes plus the four fixed stage nodes
(`reset`, `fork`, `join`, `gate`) and two checkers. The gate command reconciles
the checker verdicts via
`python -m factory.workflow.guardrails_aggregate check`; the reset command
persists `{slug: rule}` via `... guardrails_aggregate reset`. Verdicts, the
iteration token and `violations.md` live under `.factory/guards/draft/`.

The two guards on `draft`, in attachment order:

1. `draft.md must state Status: ready`
2. `draft.md must not contain TBD`

## Boundaries — what is and is not expanded

`expand_agent_guards` runs **only** where a flat graph is produced through
`Package.compile()` (`factory/workflow/package.py`). Therefore:

- Hand-built `Workflow` objects that never call `compile()` are **not** expanded —
  including builtin modes in `definitions.py`, contributed workflow files that
  return a hand-built graph, and YAML overrides via
  `factory.skillopt.yaml_surface.yaml_to_workflow`. A `guards` list on such a
  node is inert data: it serializes, it does not enforce (ASM-004). This mode
  returns `package.compile()` precisely so it does not hit that boundary.
- `WorkflowRegistry` returns whatever `workflow()` returns, so a
  package-composed mode must return `…compile()` for `show` / `validate` /
  `export-skills` to see the stage. This mode does.
- Guards attached to an agent inside a `ForkNode` branch are **rejected** at
  expansion: branch targets are executed without their outgoing edges, so a
  spliced stage there would be silently dead.
- A guarded agent with `blocking=False` is **rejected** at expansion: the
  executor would schedule it in the background and immediately run reset /
  checkers / gate, so the stage could read `ABSENT` and false-pass.
- Guards on agents reached through a `DataNode` branch path or inside a
  `SubgraphForkNode` subgraph do get a stage emitted (expansion walks
  `workflow.nodes`), but running them in those contexts is **unverified and
  unsupported** in this PR (ASM-005). Do not attach guards there.

## Usage

```bash
factory workflow validate --file factory/workflow/contributed/toy_guard/workflow.py
```

`factory workflow show toy-guard` / `export-skills` observe the expanded stage
only where the install does not pre-populate `WorkflowRegistry` (the
`composed_modes` entry point self-registers before discovery — ESC-001). The
gates therefore exercise the real handlers **in-process** after
`WorkflowRegistry.reset()`; `validate --file` bypasses the registry entirely.

The mode is a test fixture: `tests/test_workflow_guardrails_e2e.py` drives it
through `WorkflowExecutor(agent_fn=...)` with stub agents, so it needs no model,
CLI, network or paid service. See `test_workflow.py` for the graph contract
checks collected by the normal pytest run.
