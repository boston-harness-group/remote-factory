# Builder Review — Split universal vs project-specific test infrastructure guidance

## Changes Made

### 1. Simplified universal Builder prompt (`factory/agents/prompts/builder.md`)
- **Before:** Two paragraphs under "Test infrastructure for connected scope" — one for remote-factory changes (naming FakeAgent, DummyTask, auto_write_outputs) and one for new projects (create project-specific fakes)
- **After:** Single line: "create or use project-specific fakes that replace external I/O. The fakes are a test deliverable — they belong in the project test infrastructure, not inline in each test."
- **Rationale:** The universal prompt should not contain remote-factory-specific tooling details. Those belong in the project override.

### 2. Created project-specific override (`.factory/agents/builder.md`)
- Lists the three reusable test infrastructure components: `FakeAgent`, `DummyTask`, `auto_write_outputs=False`
- Placed at `.factory/agents/builder.md` — the standard project-specific override path used by the agent runner's two-tier lookup

## Verification

- **Artifact check:** Both files exist with correct content
- **Smoke tests:** 165 passed (test_models, test_guards, test_runners)
- **Agent runner lookup:** Confirmed `factory/agents/runner.py` checks `.factory/agents/<role>.md` as the first-priority override path (line 64)
- **No test breakage:** Changes are prompt-only (markdown), no code logic affected
