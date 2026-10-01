# Builder Review — Remove duplicate fields from hypothesis template

## Issue
Remove 4 extra fields from the hypothesis template in `factory/agents/prompts/strategist.md` that duplicate what the acceptance criteria table already captures.

## Changes Made
- **File:** `factory/agents/prompts/strategist.md`
- **Lines removed (previously 139-142):**
  - `**Regression scenario:** <what input or condition produces the bug or gap>`
  - `**Expected outcome:** <what should happen instead>`
  - `**Required scope:** local (single function) | connection (value crosses a module boundary)`
  - `**Connection to exercise:** <required only when scope=connection — ...>`

## Verification
- Searched entire file for all 4 field names — confirmed they appeared only once (lines 139-142)
- After removal, template flows: Acceptance criteria → Execution step → Expected output → Why
- No other hypothesis templates or examples in the file contained these fields

## Acceptance Evidence

| Criterion | Scope | Evidence |
|---|---|---|
| 4 duplicate fields removed from hypothesis template | artifact | `git diff` shows exactly 4 lines removed from lines 139-142 |
| No other occurrences remain in file | artifact | `grep -n "Regression scenario\|Expected outcome\|Required scope\|Connection to exercise"` returns empty |
| Template structure is correct after removal | artifact | Lines 138→139 now flow from acceptance criteria table row directly to `**Execution step:**` |
| No unrelated changes | artifact | `git diff --stat` shows only `strategist.md` modified (4 deletions) |
