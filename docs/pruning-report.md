# Dead-Code Pruning Report

**Date:** 2026-08-17
**Branch:** factory/run-28fae87c
**Sources:** Three parallel researcher analyses (runner subsystem, workflow modes, tests/utilities)

---

## 1. Executive Summary

| Category | Confidence | Dead Lines (impl) | Dead Lines (tests) | Dead Lines (docs) | Total |
|----------|-----------|-------------------|--------------------|--------------------|-------|
| Dead runners (codex, bob, opencode) | HIGH | 888 | 1,960 | 78 | 2,926 |
| Dead agent prompt (evolver.md) | HIGH | 38 | 0 | 0 | 38 |
| Dead runner infrastructure (partial) | HIGH | ~250 | 0 | 0 | ~250 |
| Dead documentation (runner-v2-spec.md) | MEDIUM | 0 | 0 | 202 | 202 |
| Uncertain workflow modes | MEDIUM | 1,582 | ~1,507 | 1,289 | ~4,378 |
| Contributed benchmark workflows | LOW | 4,201 | included | 0 | 4,201 |
| **TOTAL** | | **~2,758** | **~3,467** | **~1,569** | **~7,795** |

**Supported modes (DO NOT REMOVE):** design, create, research, meta, discover, review, deep-qa, research-standalone, doc-generate, doc-update, spec-generate, spec-update, skill-refine, evolve.

**Actively used infrastructure (DO NOT REMOVE):** contained runtime, inner loop, outer loop, adversarial eval, all utility modules (analysis.py, checkpoint.py, report.py, clean_pr.py), MemPalace, all playbooks, refactory.py, cycle_analyzer.py.

**Key correction from research:** The initial research flagged `refactory.py` (141 lines) and `cycle_analyzer.py` (518 lines) as potentially dead. Verification proved both are **actively used** — refactory has a CLI command, agent prompt, and plugin registration; cycle_analyzer is imported by inner_loop.py, outer_loop/reflector.py, outer_loop/evaluator.py, outer_loop/featurebench_inner_loop.py, and cli/outer_loop.py.

---

## 2. Dead Runner Code

Only the Claude runner is supported going forward. Three runner implementations are dead.

### 2.1 Implementation Files (888 lines)

| File | Lines | Key Classes/Functions | Evidence |
|------|-------|----------------------|----------|
| `factory/runners/codex.py` | 219 | `CodexRunner`, `is_codex_dry_run()` | 0 production call sites |
| `factory/runners/bob.py` | 316 | `BobRunner`, `is_dry_run()` | 0 production call sites |
| `factory/runners/opencode.py` | 353 | `OpenCodeRunner`, `is_opencode_dry_run()` | 0 production call sites |

### 2.2 Runner Infrastructure (partial cleanup, ~250 lines)

| File | Total Lines | Dead Lines (est.) | What to Remove |
|------|------------|-------------------|----------------|
| `factory/runners/__init__.py` | 115 | ~50 | Imports for BobRunner/CodexRunner/OpenCodeRunner, `__all__` exports, `_RUNNERS` dict entries, `get_runner()` branches (lines 8-11, 21-24, 29-30, 37-39, 70-74) |
| `factory/runners/usage.py` | 179 | ~179 | All functions default `runner_name="bob"`. Generic tracking, but all callers are dead runners. If Claude doesn't use this module, the entire file is dead |
| `factory/ceo_completion.py` | — | ~2 | Comment reference to BobRunner at line 419 |

### 2.3 Codex Plugin System (~100 lines of dead code across active files)

| File | Total Lines | Dead Lines (est.) | What to Remove |
|------|------------|-------------------|----------------|
| `factory/agents/plugin.py` | 212 | ~60 | `_CODEX_PLUGIN_AGENTS_DIR_CANDIDATE` (line 18), `generate_codex_agent_toml()` (line 121), `check_codex_agents_in_sync()` (line 163) |
| `factory/cli/admin.py` | 471 | ~30 | Codex import (line 287), `--runner codex` branch (lines 303-314) |
| `scripts/sync_agents.py` | 65 | ~40 | Codex imports (line 19-21), `_CODEX_AGENTS_DIR` (line 26), codex sync logic (lines 34-35, 54) |
| `factory/cli/_parser_groups.py` | — | ~5 | Remove `codex` from `--runner` choices |

### 2.4 Dead Environment Variables

| Variable | Runner | References |
|----------|--------|------------|
| `FACTORY_BOB_DRY_RUN` | Bob | conftest.py (autouse), 17 test refs |
| `FACTORY_BOB_MAX_INVOCATIONS_PER_CYCLE` | Bob | usage.py, 9 test refs |
| `BOBSHELL_API_KEY` | Bob | 15 test refs |
| `FACTORY_CODEX_DRY_RUN` | Codex | 20 test refs |
| `CODEX_API_KEY` | Codex | CLAUDE.md, config examples |
| `FACTORY_OPENCODE_DRY_RUN` | OpenCode | 14 test refs |
| `FACTORY_OPENCODE_MAX_INVOCATIONS_PER_CYCLE` | OpenCode | 1 test ref |

### 2.5 Dead Config Entries

- `factory/user_config.py`: Bob/codex example profiles (lines 42, 54-59), env var mappings (lines 313-314)
- `tests/conftest.py`: `os.environ["FACTORY_BOB_DRY_RUN"] = "1"` (lines 13-15)
- `.factory/.bob_auth`: Auth persistence file (created by bob.py)
- `.factory/bob_usage.jsonl`, `.factory/opencode_usage.jsonl`: Usage log files

---

## 3. Dead Agent Prompts

### `factory/agents/prompts/evolver.md` (38 lines) — CONFIRMED DEAD

| Check | Result |
|-------|--------|
| Python references (`grep -r 'evolver' factory/ --include='*.py' \| grep -v test`) | **0 hits** |
| Markdown references (`grep -r 'evolver' factory/ --include='*.md'`) | **0 hits** |
| Test references (`grep -rn 'evolver' tests/ --include='*.py'`) | **0 hits** |
| Workflow definitions | **0 references** |
| Skill exports | **0 references** |

**All other agent prompts verified in use:** adversarial_tester.md (45 Python refs), archivist.md, builder.md, ceo.md, code_reviewer.md, failure_analyst.md, health_checker.md, researcher.md, strategist.md, refactory.md (active), skill_reviewer.md (2 refs), spec_annotator.md (3 refs), spec_patcher.md (1 ref).

---

## 4. Dead/Uncertain Workflow Modes

### 4.1 Workflow Inheritance Chain (CRITICAL)

```
build_workflow ──────────> design_workflow (ACTIVE — primary mode)
improve_workflow ────────> research_workflow (ACTIVE)
_study_subgraph ─────────> design_workflow, study_standalone_workflow
_deep_qa_subgraph ───────> build, improve, refine, create, parallel-improve, frontend-design
```

**`build_workflow` is NOT independently dead** — it is the structural base for `design_workflow` (the primary mode). Deleting it breaks design mode.

**`improve_workflow` is NOT independently dead** — it is the structural base for `research_workflow`. Deleting it breaks research mode.

These workflows can only be removed from the **registry** (so they can't be invoked directly via `--mode build` or `--mode improve`), but their **function definitions must remain** as library code for the active modes that extend them.

### 4.2 Mode Status Table

| Mode | Def Lines | SKILL.md Lines | Git Log Hits | Inheritance | Verdict |
|------|----------|---------------|-------------|-------------|---------|
| `build` | 313-574 (262) | 282 | 6 | Base for `design` | KEEP function, remove from registry |
| `improve` | 853-1100 (248) | 261 | 2 | Base for `research` | KEEP function, remove from registry |
| `founder` | 4433-4531 (99) | 81 | 0 | None | SAFE TO REMOVE entirely |
| `study` | 4545-4565 (21) | 68 | 2 | Uses `_study_subgraph` | SAFE TO REMOVE (subgraph shared) |
| `refine` | 1622-1813 (192) | 210 | 0 | Uses `_deep_qa_subgraph` | SAFE TO REMOVE entirely |
| `parallel-improve` | 4166-4432 (267) | 125 | 0 | None | SAFE TO REMOVE entirely |
| `frontend-design` | 2698-3252 (555) | 395 | 0 | None | NEEDS USER CONFIRMATION |
| `frontend-design-scan` | 3253-3422 (170) | 173 | 0 | None | NEEDS USER CONFIRMATION |
| `frontend-design-discover` | 3423-3650 (228) | 157 | 0 | None | NEEDS USER CONFIRMATION |

### 4.3 Subgraph Helper Safety

| Helper | Used By |
|--------|---------|
| `_study_subgraph()` | `design_workflow` (line 610), `study_standalone_workflow` (line 4553) |
| `_deep_qa_subgraph()` | `build_workflow` (line 475), `improve_workflow` (line 973), `research_workflow` (line 1169), `refine_workflow` (line 1707), `create_workflow` (line 1990), `parallel_improve_workflow` (line 4276) |

Both helpers are used by active modes. They MUST be retained even if all uncertain modes are deleted.

### 4.4 Safe-to-Remove Mode Summary

If user confirms, these modes can be fully deleted (function + registry entry + SKILL.md):

| Mode | Function Lines | SKILL.md Lines | Total |
|------|---------------|---------------|-------|
| `founder` | 99 | 81 | 180 |
| `study` (standalone) | 21 | 68 | 89 |
| `refine` | 192 | 210 | 402 |
| `parallel-improve` | 267 | 125 | 392 |
| **Subtotal** | **579** | **484** | **1,063** |

Registry-only removal (keep function, remove direct invocation):

| Mode | Registry Entry | SKILL.md Lines | Total Removable |
|------|---------------|---------------|----------------|
| `build` | 1 line | 282 | 283 |
| `improve` | 1 line | 261 | 262 |
| **Subtotal** | **2** | **543** | **545** |

Frontend modes (pending user confirmation):

| Mode | Function Lines | SKILL.md Lines | Total |
|------|---------------|---------------|-------|
| `frontend-design` | 555 | 395 | 950 |
| `frontend-design-scan` | 170 | 173 | 343 |
| `frontend-design-discover` | 228 | 157 | 385 |
| **Subtotal** | **953** | **725** | **1,678** |

---

## 5. Contributed Workflows

10 benchmark workflows under `factory/workflow/contributed/`:

| Mode | Python Lines | Last Commit | Files |
|------|-------------|-------------|-------|
| `devopsgym` | 430 | 2026-08-13 | `__init__.py`, `README.md`, `test_workflow.py`, `workflow.py` |
| `featurebench` | 380 | 2026-07-15 | Same |
| `legacybench` | 394 | 2026-07-16 | Same |
| `mini-swebench` | 298 | 2026-08-13 | Same |
| `outer_loop` | 193 | 2026-08-17 | Same |
| `programbench` | 628 | 2026-07-16 | Same |
| `salitrap` | 420 | 2026-08-03 | Same |
| `swebench` | 365 | 2026-07-15 | Same |
| `swebenchifyhard` | 339 | 2026-08-07 | Same |
| `terminalbench` | 380 | 2026-07-15 | Same |
| `tomswe` | 374 | 2026-07-21 | Same |
| **Total** | **4,201** | | |

**Recommendation:** KEEP as plugins. They are already isolated in the `contributed/` directory with a clean contract (`workflow.py` + `test_workflow.py` + `__init__.py`). Several have recent commits (devopsgym, mini-swebench: 2026-08-13; outer_loop: 2026-08-17). Removing them risks breaking external benchmark pipelines.

**Plugin preservation strategy:**
1. Document the contributed workflow contract in `factory/workflow/contributed/README.md`
2. Flag any workflows with zero test coverage
3. Keep the dynamic registration via `_get_builtin_registry()` lambda imports

---

## 6. Additional Dead Code Candidates

### 6.1 `factory/refactory.py` (141 lines) — **NOT DEAD** (corrected)

Initial research flagged this as dead. Verification shows extensive active usage:

| Check | Result |
|-------|--------|
| CLI command | `factory refactory` — registered in `factory/cli/_main.py` dispatch table |
| CLI parser | `factory/cli/_parser_groups.py` — dedicated subparser |
| Implementation | `factory/cli/ceo.py:cmd_refactory()` — imports from `factory.refactory` |
| Agent prompt | `factory/agents/prompts/refactory.md` (active, ~350 lines) |
| Plugin registration | Listed in `factory/agents/plugin.py` agent list |
| Podman integration | Listed in `factory/podman.py:SCORING_COMMANDS` |
| CLI filter | `_REFACTORY_AGENT_COMMANDS` filter in `factory/cli/_main.py` |

**Verdict: KEEP — actively used CLI feature.**

### 6.2 `factory/cycle_analyzer.py` (518 lines) — **NOT DEAD** (corrected)

Initial research flagged this as dead. Verification shows 5 active importers:

| Importer | Usage |
|----------|-------|
| `factory/inner_loop.py` | `from factory.cycle_analyzer import CycleAnalyzer, CycleRecord` |
| `factory/outer_loop/reflector.py` | `from factory.cycle_analyzer import CycleRecord` |
| `factory/outer_loop/evaluator.py` | `from factory.cycle_analyzer import CycleRecord` |
| `factory/outer_loop/featurebench_inner_loop.py` | `from factory.cycle_analyzer import CycleRecord` |
| `factory/cli/outer_loop.py` | `from factory.cycle_analyzer import CycleRecord as CR` |

**Verdict: KEEP — core infrastructure for inner loop and outer loop.**

### 6.3 `docs/codex-mcp.md` (78 lines) — DEAD

Documents the codex plugin system. Entire file is dead after codex runner removal.

### 6.4 `docs/runner-v2-spec.md` (202 lines) — MOSTLY DEAD

Multi-runner specification document. Contains 12 references to codex/bob/opencode. Most content is about the multi-runner architecture that's being deprecated. Could be either deleted entirely or rewritten to document the Claude-only runner architecture.

---

## 7. Verification Evidence

### 7.1 Runner Classes Outside Tests

```
$ grep -r 'CodexRunner\|BobRunner\|OpenCodeRunner' factory/ --include='*.py' | grep -v test
factory/ceo_completion.py:    a new cycle. The per-cycle limit is enforced within BobRunner during execution.
factory/runners/codex.py:"""CodexRunner — OpenAI Codex CLI backend implementation."""
factory/runners/codex.py:class CodexRunner:
factory/runners/__init__.py:from factory.runners.bob import BobRunner, is_dry_run
factory/runners/__init__.py:from factory.runners.codex import CodexRunner, is_codex_dry_run
factory/runners/__init__.py:from factory.runners.opencode import OpenCodeRunner, is_opencode_dry_run
factory/runners/__init__.py:    "BobRunner",
factory/runners/__init__.py:    "CodexRunner",
factory/runners/__init__.py:    "OpenCodeRunner",
factory/runners/__init__.py:    "bob": BobRunner,  # type: ignore[dict-item]
factory/runners/__init__.py:    "codex": CodexRunner,  # type: ignore[dict-item]
factory/runners/__init__.py:    "opencode": OpenCodeRunner,  # type: ignore[dict-item]
factory/runners/__init__.py:        return BobRunner(project_path=project_path)
factory/runners/__init__.py:        return OpenCodeRunner(project_path=project_path)
factory/runners/opencode.py:"""OpenCodeRunner — OpenCode v1.x (anomalyco/opencode) CLI backend."""
factory/runners/opencode.py:class OpenCodeRunner:
factory/runners/bob.py:"""BobRunner — Bob Shell CLI backend implementation."""
factory/runners/bob.py:class BobRunner:
```

**Analysis:** All references are self-referential (the runner files themselves) or registry entries (`__init__.py`). One comment in `ceo_completion.py`. Zero production call sites outside the runner subsystem.

### 7.2 Evolver References

```
$ grep -r 'evolver' factory/ --include='*.py' | grep -v test
(no output)
```

**Zero references.** Confirmed dead.

### 7.3 Refactory References (CORRECTION)

```
$ grep -r 'refactory' factory/ --include='*.py' | grep -v test | grep -v 'factory/refactory.py'
factory/plugins.py:    "notify", "plugins", "precheck", "profile", "refactory", "refine-begin",
factory/podman.py:SCORING_COMMANDS = frozenset({"ceo", "run", "eval", "improve", "workflow", "refactory", "baseline"})
factory/agents/plugin.py:    "builder", "archivist", "ceo", "strategist", "refactory",
factory/cli/_main.py:            "refactory",
factory/cli/_main.py:        refactory_filter = "--refactory-agent" in sys.argv
factory/cli/_main.py:                    if refactory_filter and cmd not in _REFACTORY_AGENT_COMMANDS:
factory/cli/_main.py:        if not refactory_filter:
factory/cli/_main.py:        "--refactory-agent",
factory/cli/_main.py:            return _cli.cmd_refactory(args)
factory/cli/_main.py:        "refactory": _cli.cmd_refactory,
factory/cli/_parser_groups.py:    p = sub.add_parser("refactory", help="Launch the re:factory persistent supervisor agent")
factory/cli/ceo.py:def cmd_refactory(args: argparse.Namespace) -> int:
factory/cli/ceo.py:    from factory.refactory import get_session_id, setup_workspace
factory/cli/ceo.py:    session_file = project_path / ".refactory" / "session.json"
factory/cli/ceo.py:    prompt = resolve_prompt("refactory")
factory/cli/ceo.py:        prefix="refactory-prompt-",
factory/cli/ceo.py:    mcp_config = project_path / ".refactory" / ".mcp.json"
factory/cli/__init__.py:    cmd_refactory as cmd_refactory,
```

**16 active references across 6 files.** NOT dead — actively used CLI feature.

### 7.4 Cycle Analyzer References

```
$ grep -r 'cycle_analyzer' factory/ --include='*.py' | grep -v test
factory/inner_loop.py:from factory.cycle_analyzer import CycleAnalyzer, CycleRecord
factory/outer_loop/reflector.py:from factory.cycle_analyzer import CycleRecord
factory/outer_loop/evaluator.py:from factory.cycle_analyzer import CycleRecord
factory/outer_loop/featurebench_inner_loop.py:from factory.cycle_analyzer import CycleRecord
factory/cli/outer_loop.py:    from factory.cycle_analyzer import CycleRecord as CR
```

**5 active importers.** NOT dead — core infrastructure.

### 7.5 Mode Git Log Analysis

```
$ git log --all --grep='--mode founder' --oneline | wc -l
       0

$ git log --all --grep='--mode study' --oneline | wc -l
       2

$ git log --all --grep='--mode refine' --oneline | wc -l
       0

$ git log --all --grep='--mode build' --oneline | wc -l
       6

$ git log --all --grep='--mode improve' --oneline | wc -l
       2

$ git log --all --grep='--mode parallel-improve' --oneline | wc -l
       0

$ git log --all --grep='--mode frontend-design' --oneline | wc -l
       0
```

### 7.6 Dead File Line Counts

```
$ wc -l factory/runners/codex.py factory/runners/bob.py factory/runners/opencode.py tests/test_codex_runner.py tests/test_opencode_runner.py
     219 factory/runners/codex.py
     316 factory/runners/bob.py
     353 factory/runners/opencode.py
     612 tests/test_codex_runner.py
     646 tests/test_opencode_runner.py
    2146 total
```

---

## 8. Dead Test Code

### 8.1 Dedicated Dead Test Files (delete entirely)

| File | Lines | Test Methods | Tests For |
|------|-------|-------------|-----------|
| `tests/test_codex_runner.py` | 612 | 33 | CodexRunner (dead) |
| `tests/test_opencode_runner.py` | 646 | 43 | OpenCodeRunner (dead) |
| `tests/test_parallel_improve.py` | 1,507 | 18+ | parallel-improve mode (dead if mode removed) |
| **Total** | **2,765** | **94+** | |

### 8.2 Partial Dead Test Classes (remove from shared files)

**`tests/test_runners.py`** (2,553 lines total):

| Class | Lines | Location | Tests For |
|-------|-------|----------|-----------|
| `TestBobRunner` | 201 | 209-409 | BobRunner (dead) |
| `TestBobAuthPreflight` | 65 | 554-618 | Bob auth (dead) |
| `TestBobInteractivePrompt` | 39 | 1689-1727 | Bob interactive (dead) |
| `TestBobMetaAuthCheck` | 54 | 1728-1781 | Bob meta auth (dead) |
| `TestBobBuildInteractiveCommand` | 75 | 2204-2278 | Bob build command (dead) |
| `TestOpenCodeInteractive` | 71 | 1618-1688 | OpenCode interactive (dead) |
| `TestOpenCodeBuildInteractiveCommand` | 65 | 2279-2343 | OpenCode build command (dead) |
| **Subtotal** | **570** | | |

**`tests/test_plugin_agents.py`** (331 lines total):

| Class | Lines | Location | Tests For |
|-------|-------|----------|-----------|
| `TestGenerateCodexAgentToml` | 53 | 220-272 | Codex TOML generation (dead) |
| `TestCheckCodexAgentsInSync` | 23 | 273-295 | Codex sync check (dead) |
| `TestCmdInstallCodex` | 36 | 296-331 | Codex install command (dead) |
| **Subtotal** | **112** | | |

**`tests/test_session_resume.py`** (658 lines total):

| Method | Lines | Location | Tests For |
|--------|-------|----------|-----------|
| `test_bob_does_not_support_session_resume` | ~10 | 100-110 | Bob session resume (dead) |
| `test_bob_always_allowed` | ~10 | 116-122 | Bob budget check (dead) |
| **Subtotal** | **~20** | | |

### 8.3 Dead Test Fixtures

| File | Fixture/Setup | What to Remove |
|------|--------------|----------------|
| `tests/conftest.py` | `os.environ["FACTORY_BOB_DRY_RUN"] = "1"` (lines 13-15) | Global env var setup for dead runner |

### 8.4 Test Files with Dead Runner References (cleanup needed)

These files have scattered dead runner references that need selective removal:

| File | Total Lines | Bob Refs | Codex Refs | OpenCode Refs |
|------|------------|----------|------------|---------------|
| `tests/test_runner_e2e.py` | 660 | 7 | 6 | 5 |
| `tests/test_user_config.py` | 769 | 20 | 3 | 0 |
| `tests/test_tmux_cli.py` | 658 | 2 | 1 | 0 |
| `tests/test_agents.py` | 753 | 1 | 1 | 1 |
| `tests/test_ceo_completion.py` | 1,378 | (BobRunner comment) | 0 | 0 |
| `tests/test_profile.py` | 296 | (check needed) | (check needed) | 0 |

### 8.5 Dead Test Code for Uncertain Modes

If modes are confirmed dead and removed, these tests also die:

| File | Lines | Relevant Dead Tests |
|------|-------|-------------------|
| `tests/test_workflow_definitions.py` | — | Tests for `build_workflow()`, `improve_workflow()`, `refine_workflow`, `study_standalone_workflow()`, `founder_workflow()` instantiation and validation |
| `tests/test_skill_export.py` | — | Tests for SKILL.md generation of dead modes |
| `tests/test_lazy_loading.py` | — | Tests for lazy-loaded dead mode entries |
| `tests/test_deprecation.py` | — | Tests for deprecated mode handling |
| `tests/test_workflow_deep_research.py` | — | Tests referencing dead research sub-patterns |

### 8.6 Dead Test Summary

| Category | Lines | Confidence |
|----------|-------|-----------|
| Dedicated dead runner test files | 1,258 | HIGH |
| Dead runner classes in shared test files | 570 | HIGH |
| Dead codex plugin test classes | 112 | HIGH |
| Dead bob session resume tests | ~20 | HIGH |
| Dead parallel-improve test file | 1,507 | MEDIUM (if mode removed) |
| **Total HIGH confidence** | **1,960** | |
| **Total if modes removed** | **3,467** | |

---

## 9. Safe Deletion Order

### Phase 1 — Runner Isolation (prerequisite, modifies active files)

Modify these files to decouple dead runners before deleting them:

| File | Action |
|------|--------|
| `factory/runners/__init__.py` | Remove imports for BobRunner, CodexRunner, OpenCodeRunner. Remove from `_RUNNERS` dict and `__all__`. Remove `is_codex_dry_run`, `is_opencode_dry_run` exports. Update `get_runner()` to error on "bob"/"codex"/"opencode" |
| `factory/cli/_parser_groups.py` | Remove `codex` from `--runner` choices in `install` command |
| `factory/user_config.py` | Remove bob/codex example profiles and env var mappings |
| `factory/ceo_completion.py` | Remove BobRunner comment at line 419 |
| `tests/conftest.py` | Remove `FACTORY_BOB_DRY_RUN=1` setup (lines 13-15) |

### Phase 2 — Runner Deletion (delete implementation files)

| File | Lines | Action |
|------|-------|--------|
| `factory/runners/codex.py` | 219 | DELETE |
| `factory/runners/bob.py` | 316 | DELETE |
| `factory/runners/opencode.py` | 353 | DELETE |
| `factory/runners/usage.py` | 179 | DELETE or refactor (verify if Claude uses it) |

### Phase 3 — Test Deletion (delete dead test files and classes)

| File | Action |
|------|--------|
| `tests/test_codex_runner.py` | DELETE entirely (612 lines) |
| `tests/test_opencode_runner.py` | DELETE entirely (646 lines) |
| `tests/test_runners.py` | Remove 5 Bob classes (434 lines) + 2 OpenCode classes (136 lines) |
| `tests/test_plugin_agents.py` | Remove 3 Codex classes (112 lines) |
| `tests/test_session_resume.py` | Remove 2 bob test methods (~20 lines) |
| `tests/test_runner_e2e.py` | Remove dead runner test parametrizations |
| `tests/test_user_config.py` | Remove bob/codex test cases (~20 lines) |
| `tests/test_tmux_cli.py` | Remove bob/codex references (~3 lines) |
| `tests/test_agents.py` | Remove bob/codex/opencode references (~3 lines) |

### Phase 4 — Plugin System Cleanup

| File | Action |
|------|--------|
| `factory/agents/plugin.py` | Remove `_CODEX_PLUGIN_AGENTS_DIR_CANDIDATE`, `generate_codex_agent_toml()`, `check_codex_agents_in_sync()` (~60 lines) |
| `factory/cli/admin.py` | Remove codex import and `--runner codex` branch (~30 lines) |
| `scripts/sync_agents.py` | Remove codex imports, `_CODEX_AGENTS_DIR`, codex sync logic (~40 lines) |

### Phase 5 — Dead Agent Prompts

| File | Action |
|------|--------|
| `factory/agents/prompts/evolver.md` | DELETE (38 lines) |

### Phase 6 — Documentation Cleanup

| File | Action |
|------|--------|
| `docs/codex-mcp.md` | DELETE entirely (78 lines) |
| `docs/runner-v2-spec.md` | DELETE or rewrite to Claude-only (202 lines) |
| `CLAUDE.md` | Rewrite Runners section — remove bob/codex/opencode specifics (19 refs) |
| `SPEC.md` | Update External Dependencies, Config Fields — remove dead runners (10 refs) |
| `docs/index.md` | Remove runner comparison content (9 refs) |
| `docs/configuration.md` | Remove bob example profiles (7 refs) |

### Phase 7 — Low-Confidence Candidates (SKIP — verified active)

~~`factory/refactory.py`~~ — **KEEP** (16 active references, CLI command)
~~`factory/cycle_analyzer.py`~~ — **KEEP** (5 active importers, core infrastructure)

### Phase 8 — Mode Consolidation (requires user confirmation)

After user confirms which modes are dead:

| Mode | Action |
|------|--------|
| `founder` | DELETE function from definitions.py, DELETE `skills/workflow-founder/SKILL.md`, remove registry entry |
| `study` (standalone) | DELETE function from definitions.py, DELETE `skills/workflow-study/SKILL.md`, remove registry entry. KEEP `_study_subgraph()` |
| `refine` | DELETE function from definitions.py, DELETE `skills/workflow-refine/SKILL.md`, remove registry entry |
| `parallel-improve` | DELETE function from definitions.py, DELETE `skills/workflow-parallel-improve/SKILL.md`, remove registry entry. DELETE `tests/test_parallel_improve.py` |
| `build` | KEEP function (base for design). Remove registry entry. DELETE `skills/workflow-build/SKILL.md` |
| `improve` | KEEP function (base for research). Remove registry entry. DELETE `skills/workflow-improve/SKILL.md` |
| `frontend-design` (3 modes) | Pending user confirmation |

### Phase 9 — Final Documentation Cleanup

- Update README.md runner/mode references
- Leave CHANGELOG.md historical entries intact
- Update `factory.md` smoke test if it references dead runners
- Run `factory workflow export-skills` to regenerate SKILL.md files after mode removal

---

## 10. Risk Assessment

| Category | Risk | Justification |
|----------|------|---------------|
| Dead runners (codex, bob, opencode) | LOW | Zero production call sites. Fully isolated behind runner abstraction. Claude-only path unaffected |
| Dead agent prompt (evolver.md) | LOW | Zero references anywhere in the codebase |
| Runner infrastructure cleanup | LOW | Registry entries and imports are mechanical cleanup |
| Codex plugin system | LOW | Only serves codex agent TOML generation. No Claude equivalent needed |
| Dead documentation | LOW | No functional impact |
| `founder` mode | LOW | 0 git log hits, terminal mode, no inheritance dependencies |
| `study` standalone mode | LOW | Subgraph helper is shared (retained), only standalone invocation removed |
| `refine` mode | LOW | 0 git log hits, no inheritance dependencies |
| `parallel-improve` mode | LOW | 0 git log hits, no inheritance dependencies |
| `build` mode (registry removal only) | MEDIUM | Function must be retained as base for `design_workflow`. Only the direct `--mode build` entry point is removed |
| `improve` mode (registry removal only) | MEDIUM | Function must be retained as base for `research_workflow`. Only the direct `--mode improve` entry point is removed |
| Frontend-design modes | MEDIUM | 0 git log hits but may be used by external projects. Needs user confirmation |
| Contributed benchmark workflows | MEDIUM | May be used by external benchmark pipelines. Plugin architecture already provides isolation |
| `factory/runners/usage.py` | MEDIUM | Generic tracking module. Need to verify if Claude runner uses it before deleting |
| `factory.md` smoke test | LOW | Currently references `BobAuth` in `-k` filter — needs update after bob removal |

---

## Appendix A: Cross-Reference Dependency Map

```
DEAD CODE                          ACTIVE CODE NEEDING CLEANUP
─────────────                      ───────────────────────────
factory/runners/codex.py ────────> factory/runners/__init__.py (imports, registry)
                          ────────> factory/agents/plugin.py (codex TOML generation)
                          ────────> factory/cli/admin.py (codex install branch)
                          ────────> factory/cli/_parser_groups.py (runner choices)
                          ────────> scripts/sync_agents.py (codex sync)

factory/runners/bob.py ──────────> factory/runners/__init__.py (imports, registry)
                        ──────────> factory/runners/usage.py (default runner_name)
                        ──────────> factory/ceo_completion.py (comment)

factory/runners/opencode.py ─────> factory/runners/__init__.py (imports, registry)

factory/agents/prompts/evolver.md  (no references — fully orphaned)

docs/codex-mcp.md ───────────────> (no code references — standalone doc)

DEAD TESTS                         DEAD PRODUCTION CODE
──────────                         ────────────────────
tests/test_codex_runner.py ──────> factory/runners/codex.py
tests/test_opencode_runner.py ───> factory/runners/opencode.py
tests/test_runners.py (Bob) ─────> factory/runners/bob.py
tests/test_runners.py (OC) ──────> factory/runners/opencode.py
tests/test_plugin_agents.py (Cx) > factory/agents/plugin.py (codex functions)
tests/test_session_resume.py (B) > factory/runners/bob.py
tests/test_parallel_improve.py ──> factory/workflow/definitions.py (parallel-improve)
```

## Appendix B: Documentation Files with Dead Runner References

| File | Codex Refs | Bob Refs | OpenCode Refs | Total | Action |
|------|-----------|----------|--------------|-------|--------|
| `CLAUDE.md` | — | — | — | 19 | Rewrite Runners section |
| `SPEC.md` | — | — | — | 10 | Update external deps |
| `docs/runner-v2-spec.md` | — | — | — | 12 | Delete or rewrite |
| `docs/index.md` | — | — | — | 9 | Remove runner comparison |
| `docs/codex-mcp.md` | 8 | 0 | 0 | 8 | Delete entirely |
| `docs/configuration.md` | 0 | 5 | 0 | 5 (bob) | Remove bob examples |

## Appendix C: `factory.md` Smoke Test Update

The current smoke test in `factory.md` line 86 contains:
```bash
uv run pytest tests/test_models.py tests/test_guards.py tests/test_runners.py -x -q --tb=short -k 'not (BobAuth or preflight_error_unchanged)'
```

After bob removal, the `-k 'not (BobAuth or preflight_error_unchanged)'` filter and possibly `tests/test_runners.py` itself need updating to reflect the simplified test surface.
