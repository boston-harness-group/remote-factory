# Builder Prompt Review — Behavioral Tests in Output & Exit Conditions

## Changes Made

1. **Output section**: Changed artifact count from "two" to "three" and added item 3 — behavioral tests for each `**Critical path:**` named in the issue.
2. **Exit conditions**: Both "Success (first run)" and "Success (reloop)" now include ", behavioral tests written for each named critical path".

## What Was NOT Changed

- The preamble section (lines 37–69) containing the detailed behavioral test rules, examples, and guidance — preserved as-is for reference.
- All other sections (Constraints, Guardrails, When Blocked) — untouched.

## Rationale

The behavioral test requirement was defined in the Task preamble but never surfaced in the Output deliverables or Exit conditions. Agents that skip to the Output section to check "am I done?" would miss the requirement entirely. These changes close that gap.
