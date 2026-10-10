# ADR: task-setup uses a director topology with a mandatory outer loop feedback research phase

*Status: Proposed*
*Author: georgosgeorgos*
*Related: PR #1509, PR #1528*

---

## Decision

task-setup uses a **director topology** — CEO agents that dynamically spawn sub-agents based on domain complexity — and requires a **mandatory outer loop feedback research phase** before any code is generated. The research phase always includes a researcher focused specifically on what `VerifyResult.details` should contain to make the generated Task useful for evolutionary search.

---

## Context

Creating a new `Task` (the 4-hook interface: `instances()`, `setup()`, `prompt()`, `verify()`) is the entry point for applying the outer loop to a new domain. The correctness bar is low: if `verify()` returns a score between 0 and 1 and the CLI validates, the Task works. But a Task that works is not the same as a Task that is useful to the outer loop.

`VerifyResult.details` is the learning signal. The outer loop's reflection mechanism reads `details` dicts across the population and extracts patterns like:

> "High-scoring individuals had low blunder_count while low-scoring ones had high blunder_count"

These patterns drive mutation suggestions for the next generation. If `details` is sparse — `{"passed": True, "score": 0.7}` — the reflector has nothing to compare, suggestions are generic, and evolution stalls. The failure is silent: a sparse-details Task passes functional tests, validates, and runs without error. The first observable sign of the problem is a plateau in generation 0 or 1 with no improvement signal.

The other challenge is that Task design involves domain research a developer cannot be expected to carry in their heads. What are the right instances? What granularity of `details` enables meaningful comparison? What verification strategy fits the domain? These are non-obvious and benefit from structured investigation before implementation.

---

## Alternatives considered

**CLI wizard — prompted interactive input, no agent cost**

A structured questionnaire that asks the developer to fill in instance structure, scoring method, and details schema. Low cost, fast, predictable.

Rejected because the questions that matter most — "what numeric dimensions distinguish good solutions from bad ones in your domain?" — require domain research to answer well. A wizard front-loads the difficulty onto the developer at exactly the moment when they have the least information. It also can't adapt: a chess Task and a drug-simulation Task need fundamentally different research questions.

**Headless automated pipeline — no user gate, fixed researcher count**

Three fixed researchers (domain, verification, outer loop feedback) → synthesizer → builder → validate. No human approval step.

Rejected because the strategy synthesis step produces a Task specification that may have wrong assumptions about the domain, the scoring method, or the details schema. Building on a wrong spec wastes the full implementation round-trip. A user gate after strategy is cheap insurance against that. The director pattern is preferred over fixed researchers for the same reason as below.

---

## Why directors over a fixed researcher pipeline

A fixed pipeline with a predetermined number of researchers works when research dimensions are known in advance. For Task design they are not:

- A chess Task needs a tooling researcher (python-chess, Stockfish setup). A sorting Task doesn't.
- A drug discovery Task needs a dataset researcher (PDBbind, existing benchmarks). A code-generation Task doesn't.

Adding fixed nodes for every possible research dimension produces a bloated pipeline where most nodes are no-ops for most domains. Directors adapt: the Research Director reads the user's domain description and decides which dimensions need investigation. The mandatory dimensions (domain, verification, outer loop feedback) are always spawned. Optional dimensions are spawned when the domain warrants them.

---

## Why the outer loop feedback researcher is mandatory

Every task-setup run requires a researcher focused specifically on what `VerifyResult.details` should contain, even when the developer believes they know their domain.

The failure mode is silent and expensive to retrofit. A Task with sparse `details` passes all tests and validates cleanly. The outer loop runs, scores are computed, and generations proceed. The first sign of the problem is that improvement curves plateau immediately — the reflector is examining `{"passed": True, "score": 0.7}` across the population and has nothing to compare. Diagnosing this requires inspecting cycle records, tracing back to the Task definition, and rebuilding it. By then the outer loop has consumed a budget on an unlearnable objective.

Making the feedback researcher mandatory creates an explicit gate: you must think about observability before implementation. The concrete output — a set of recommended `details` keys with their types and learning value — becomes an input to the builder prompt, not an afterthought.

---

## Prompt design principles

**Generic framing for reflection mechanisms.** Prompts must not reference the current reflector algorithm by name. Write "reflection and improvement mechanisms" and "good solutions vs bad ones," not "contrastive analysis" or "top-K vs bottom-K." The invariant — rich details enable better learning — is stable. The implementation is not.

**Concrete example over abstract description.** Every prompt that explains `VerifyResult.details` design includes the chess-evolve example:

```python
# Good — numeric, comparable, per-instance
details = {
    "wins": 3, "draws": 1, "losses": 1,
    "blunder_count": 4,
    "avg_eval": -0.3,
    "game_results": [{"game": 1, "result": "win", "moves": 18}, ...],
}

# Anti-example — unlearnable
details = {"passed": True, "score": 0.6}
```

**Multiple numeric dimensions.** The reflection mechanism computes differences between populations. It needs numeric fields to do this. Boolean and string fields support error categorization but cannot drive numeric comparison. The builder prompt specifies that `details` must contain several numeric fields — how many is a heuristic that should be revisited as the reflector evolves; the chess example sets the expected level of richness.

**Per-instance granularity.** `details` should include per-instance breakdowns in addition to aggregates. This lets the reflector identify which specific instances drive score differences — and which mutation patterns help on hard instances specifically.

**Categorical error classification.** Include an `error_type` or `failure_category` field. "Failed on 3 instances" is less useful than "failed on 3 instances: 2 timeout, 1 wrong_output."

---

## Consequences

**Benefits:**
- Forces observability to be designed before implementation rather than retrofitted
- Directors adapt research depth to domain complexity — simple domains get fast runs, complex domains get thorough coverage
- User gate after strategy catches wrong assumptions before code is written

**Tradeoffs:**
- Director runs are not statically analyzable — sub-agents are spawned at runtime, which makes CI testing of the workflow itself impractical beyond topology checks
- Every run incurs the cost of 3+ researcher agents plus a strategy phase, even when the developer knows their domain well. Expect 20–40 minutes and meaningful token cost per run
- Developers building quick experimental Tasks will bypass the workflow entirely and write the Task directly. This is acceptable — the workflow is optimized for Tasks intended for multi-generation outer loop runs
- Prompt quality cannot be verified statically. The ADR's principles (concrete examples, multiple numeric dimensions, per-instance breakdowns) are criteria, not guarantees. Real validation requires running the workflow and inspecting generated Tasks

---

## Open questions

1. **Retroactive coverage.** `chess_evolve_task.py` predates this ADR. Does this ADR apply retroactively, and if so, what is the migration path?

2. **Numeric dimension threshold.** The chess example has 4 numeric keys (`wins`, `draws`, `losses`, `blunder_count`, `avg_eval`). Is this the right target? Should the outer loop feedback researcher be given an explicit floor, and if so, derived from what property of the reflector?

3. **User gate in headless mode.** `auto_approve=True` bypasses the strategy user gate. What guardrails, if any, should exist for headless task-setup runs?
