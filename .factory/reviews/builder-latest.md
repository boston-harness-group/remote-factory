# Builder Output — Bug 2 Cost Fix + Real Haiku Run

## Task A: Bug 2 (Per-Item Cost) — FIXED

### Root Cause
The previous implementation in `data_runtime.py` fabricated cost from `duration_ms`:
```python
branch_cost = branch_result.duration_ms / 1000.0  # WRONG: milliseconds ≠ dollars
```

### Fix Applied

**1. Removed duration fallback entirely** (`factory/workflow/data_runtime.py`)
- Never put a non-cost value in a cost field
- Cost now reads from `factory.events.sum_agent_costs()` which sums
  `total_cost_usd` from `agent.completed` events in `.factory/events.jsonl`
- Events are written per-worktree by `invoke_agent()` in `factory/agents/runner.py`
- If cost is 0.0 (no agent nodes in branch), logs a structured warning

**2. Real cost path traced:**
```
invoke_agent() → result.usage.total_cost_usd → _emit_safe("agent.completed") →
  events.jsonl → sum_agent_costs(project_path, since=start_time) → ItemResult.cost
```

**3. Test rewritten** (`tests/test_outer_loop/test_data_runtime_bugs.py`)
- `test_item_cost_from_agent_events`: AgentNode workflow + mock agent that writes
  events with known cost ($0.01). Asserts exact cost per item via `pytest.approx()`.
- `test_no_agent_branch_cost_is_zero`: FnNode-only branch → cost stays 0.0 (no fabrication).
- `test_cycle_record_sums_exact_costs`: CycleRecord sums exact per-item costs.

### Files Changed
| File | Change |
|------|--------|
| `factory/workflow/data_runtime.py` | Removed duration_ms fallback; added real cost from `sum_agent_costs()` |
| `tests/test_outer_loop/test_data_runtime_bugs.py` | Rewrote Bug 2 tests with exact cost assertions |

### Test Results
- `test_data_runtime_bugs.py`: **11 passed** (0.50s)
- `test_fork_e2e.py`: **22 passed, 11 xfailed** (2.00s) — NOT MODIFIED
- Full suite: **5908 passed**, 13 skipped, 11 xfailed
- `ruff check .`: All checks passed
- `mypy factory/`: Success, 0 issues in 254 files

---

## Task B: Real Haiku Outer-Loop Run

### Configuration
- **Model**: `claude-haiku-4-5-20251001`
- **Task**: DocQualityTask — 3 instances (2 train + 1 holdout)
- **Workflow**: plan → DataNode → work(AgentNode) → check → JoinNode → summarize
- **Budget**: 4 evaluations, population_size=2

### Results

| Metric | Value |
|--------|-------|
| best_score (train) | 0.580 |
| val_score (holdout) | 0.650 |
| overfit_flag | False |
| convergence_reason | budget_exhausted |
| generations_completed | 1 |
| total_evaluations | 4 |
| total_cost_usd | $0.2167 |
| duration | 256.5s |

### Train Item IDs
- `readme-cli` (train)
- `tutorial-scraping` (train)

### Val Items (holdout only)
- `api-reference` (val)

### Per-Item Cost > 0 from CycleRecord
| Individual | Item | Score | Cost | Status |
|------------|------|-------|------|--------|
| 4d9d1011 | readme-cli | 0.6500 | $0.0248 | ok |
| 4d9d1011 | tutorial-scraping | 0.5400 | $0.0261 | ok |
| c9af1c3e | readme-cli | 0.5600 | $0.0226 | ok |
| c9af1c3e | tutorial-scraping | 0.6300 | $0.0244 | ok |
| a549d4dd | readme-cli | 0.6500 | $0.0357 | ok |
| a549d4dd | tutorial-scraping | 0.5900 | $0.0328 | ok |
| ea407a2d | readme-cli | 0.6500 | $0.0256 | ok |
| ea407a2d | tutorial-scraping | 0.6300 | $0.0247 | ok |

All per-item costs are real USD from Claude CLI usage (not fabricated from duration).

### Reflector Prompt (verify_details present)
The contrastive reflector analyzed top-K vs bottom-K individuals and identified:
- **Completeness bottleneck**: max completeness score was 0.25 — agents generate surface-level
  docs that pass grammar/readability but lack substantive content
- **Prompt improvements generated**: 4 concrete suggestions for better content depth
- **Typed mutation suggestions**: 8 structured operator suggestions including
  `prompt_mutate`, `node_insert`, `knob_mutate`

### Git Worktree List (after run)
```
/tmp/doc-quality-project  6fa2c6e [master]
```
All evaluation worktrees cleaned up successfully.
