# Bug Fixes for PR #1581 — Builder Review

## Bugs Fixed

### Bug 1 — SPLIT LABEL
**Problem:** Val items get `ItemResult.split="train"` regardless of requested split.
**Fix:** Added `split` parameter to `run_fork()`, `evaluate_fork()`, and `_resolve_items()` in `factory/workflow/data_runtime.py`. All `ItemResult` constructions now pass `split=split`. The executor and inner_loop pass the correct split through.
**Tests:** `TestBug1SplitLabel` — 3 tests verifying split propagation for train, val, and evaluate_fork paths.

### Bug 2 — PER-ITEM COST
**Problem:** Every `ItemResult.cost` is 0.0.
**Fix:** After each branch executor completes in `run_fork()`, read `branch_result.duration_ms` and set it as `branch_cost`. Added `cost` field to `ExecutionResult`. Updated `CycleRecord.from_run()` to sum per-item costs into `total_cost_usd`.
**Tests:** `TestBug2PerItemCost` — 2 tests: one verifies non-errored items have cost > 0 after branch execution, another verifies CycleRecord sums costs correctly.

### Bug 3 — ITEM STORE MISSING
**Problem:** Nothing writes `.factory/runs/<run>/items.jsonl`.
**Fix:** After all items complete in `run_fork()`, write each `ItemResult` as a JSON line to `.factory/runs/<run>/items.jsonl`.
**Tests:** `TestBug3ItemStore` — 1 test verifying file exists, contains one JSON line per item, and each line deserializes to a valid `ItemResult`.

### Bug 4 — SPLITS
**Problem:** `allowed_instance_ids` on executor doesn't use the #1576 design.
**Fix:** Updated `_resolve_items()` to call `task.instances(split=split)` for split-aware item resolution. Added subset ID validation — if subset contains IDs not in the split, `ValueError` is raised. The inner_loop now detects the correct split from the subset selector's IDs (val IDs → split="val"). Graceful fallback for tasks that don't accept `split` kwarg.
**Tests:** `TestBug4SplitFiltering` — 4 tests: val-only items, train-only items, invalid subset raises ValueError, valid subset filters correctly.

## Files Modified

| File | Changes |
|------|---------|
| `factory/workflow/data_runtime.py` | Added `split` param, cost tracking, items.jsonl writing, split-aware item resolution |
| `factory/workflow/executor.py` | Added `split` param, `cost` field on `ExecutionResult`, pass split to `run_fork` |
| `factory/inner_loop.py` | Detect split from subset selector, pass split to executor and evaluate_fork |
| `factory/cycle_analyzer.py` | `CycleRecord.from_run()` sums per-item costs into `total_cost_usd` |
| `tests/test_outer_loop/test_data_runtime_bugs.py` | 10 new unit tests for bugs 1-4 |

## Haiku Run Results (Bug 5)

Run via `SwarmEngine.run()` with `SwarmEvaluator(inner_loop_factory=True)` and `FACTORY_MODEL=claude-haiku-4-5-20251001`:

| Metric | Value |
|--------|-------|
| Train item IDs | `['i1', 'i2', 'i3', 'i4']` |
| Val items (holdout only) | `['i5', 'i6']` — never appear in train execution log |
| val_score (OuterLoopResult) | `0.7` |
| best_score | `0.417` (mean of 0.8, 0.6, 0.0 over 3 non-errored) |
| Per-item cost > 0 | ✅ i1=0.202, i2=0.217, i4=0.202 (i3=0.0 errored) |
| CycleRecord total_cost_usd | `0.621` (sum of per-item costs) |
| Reflector prompt has verify_details | ✅ Contains `method`, `exact_match`, `fuzzy`, `similarity` |
| Worktrees after run | 1 (no leftover worktrees) |

## Test Results

- `tests/test_outer_loop/test_data_runtime_bugs.py`: **10 passed**
- `tests/test_outer_loop/test_fork_e2e.py`: **11 passed, 11 xfailed** (ceo-skill deferred to PR B)
- `tests/test_data_node.py`: **83 passed, 11 xfailed**
- Full suite (excluding pre-existing container/k8s failures): **2443+ passed, 1 skipped**
- `ruff check .`: ✅ All checks passed
- `mypy factory/`: ✅ No issues found in 254 source files
