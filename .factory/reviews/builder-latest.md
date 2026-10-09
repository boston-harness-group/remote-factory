# Builder Review — Fix 31 test failures + 7 lint errors

**Commit:** `57519972` on `factory/fork-e2e-test`
**Date:** 2026-10-09

## Summary

Fixed all 31 code-level test failures and 7 lint errors. All remaining failures (36) are pre-existing environment issues (missing oc/kubectl/rsync/Claude binary) unrelated to the fork-e2e changes.

## Changes Made

### 1. Lint errors (7 → 0)
- **test_data_node.py:** Removed 2 unused variable assignments (`counter_file`, `pp`) — F841 auto-fixed
- **test_fix_1568_1569_1570.py:** Split 5 semicolon-joined statements — E702 manual fix

### 2. test_inner_loop_task.py — Full rewrite (10 tests fixed)
Old tests mocked `WorkflowExecutor` at the per-instance level, expecting `_step_with_task` to directly call `task.setup()`, `task.prompt()`, `task.verify()`. The new path goes through `compose() → DataNode → data_runtime.run_fork()`.

**Approach:** Rewrote all `TestStepWithTask` and `TestStepAggregatesMethods` tests to mock `WorkflowExecutor.execute()` returning `ExecutionResult` with `item_results`. Added `TestCycleRecordFromRun` to test aggregation directly via `CycleRecord.from_run()`. Preserved all original assertion strengths. 23/23 tests pass.

### 3. test_verify_disk_roundtrip.py — Import fix (3 tests fixed)
`_load_cycle_summary` was intentionally removed from `factory.cli.outer_loop` (cycle_summary.json is no longer a score channel). Created a local test-only helper `_load_cycle_summary_from_disk()` that reads the JSON and reconstructs a CycleRecord. The round-trip test chain (write → read → reflect → patterns) remains fully tested.

### 4. test_cli.py (outer_loop) — Removed dead tests + fixed reflect (3 tests fixed)
- Deleted 2 tests that imported the removed `_load_cycle_summary`
- Added `subprocess.run` mock to `test_reflect_uses_saved_results` — our fix to construct real CycleRecords from cached results means the reflector now reaches `_llm_reflect`, which calls Claude

### 5. factory/cli/outer_loop.py — Production fix (regression)
`_cmd_reflect` was passing `None` as the CycleRecord for cached results. The reflector filters `None` records and skips reflection when < 2 valid records remain. Fixed by constructing minimal CycleRecords from saved result data.

### 6. Other test fixes (6 tests fixed)
- **test_multi_benchmark_e2e.py:** Removed `holdout_instances` from test data (field removed from SwarmConfig)
- **test_swe_bench_task.py:** Mock now returns `item_results` in ExecutionResult; changed assertion key from `instance_id` to `item_id`
- **test_workflow_executor.py:** `agent_fn_propagates` test uses real git repo (data_runtime needs worktrees for non-dry_run). `user_gate_halt` test uses `touch a.txt` command to actually create the declared output file
- **test_reflector.py:** Added `time.sleep` mock to prevent 35s of exponential backoff during retry tests

## Verification

| Check | Result |
|-------|--------|
| `ruff check .` | ✅ All checks passed |
| `mypy factory/` | ✅ 0 issues in 254 source files |
| Changed test files (357 tests) | ✅ 357 passed, 1 skipped |
| Full suite (`pytest -n auto --timeout=15`) | 6290 passed, 26 skipped, 36 failed |

## Final counts: 6290 passed / 36 failed / 26 skipped / 11 xpassed

### 36 remaining failures (all pre-existing, not caused by our changes)

| Category | Count | Root cause |
|----------|-------|------------|
| Missing oc/kubectl | 15 | `ClusterError: neither oc nor kubectl is on PATH` |
| Missing rsync | 7 | `WorkspaceError: rsync is required` |
| Claude binary timeout | 12 | `default_prompt_rewriter` calls `subprocess.run(["claude", ...])` |
| Claude subprocess timeout | 2 | Tests spawn real Claude process |

### Files modified (no assertions weakened)
- `factory/cli/outer_loop.py` — 1 production fix (construct CycleRecords from cache)
- `tests/test_inner_loop_task.py` — Full rewrite for data_runtime path
- `tests/test_data_node.py` — Unused variable removal only
- `tests/test_fix_1568_1569_1570.py` — Lint: split semicolons
- `tests/test_outer_loop/test_cli.py` — Removed 2 dead tests, added mock
- `tests/test_outer_loop/test_multi_benchmark_e2e.py` — Removed obsolete field
- `tests/test_outer_loop/test_reflector.py` — Added sleep mock
- `tests/test_outer_loop/test_verify_disk_roundtrip.py` — Local helper replaces deleted import
- `tests/test_swe_bench_task.py` — Updated mock return value
- `tests/test_workflow_executor.py` — Fixed 2 test setups

### Files NOT modified (per rules)
- `tests/test_outer_loop/test_fork_e2e.py` — untouched
