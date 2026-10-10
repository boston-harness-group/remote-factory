# pg-factory: pydantic-graph Factory Engine Prototype

A standalone prototype that validates whether [pydantic-graph](https://github.com/pydantic/pydantic-graph) can replace the Factory's custom workflow execution engine.

## What this is

The Factory currently uses a custom graph execution engine (`primitives.py` + `executor.py`) with runtime edge matching, explicit `dict[str, Node] + list[Edge]` construction, and manual verdict routing. This prototype ports progressively harder subgraphs to pydantic-graph's `BaseNode` + `Graph` API to evaluate:

1. Can `BaseNode` model factory node types cleanly?
2. Does return-type-as-edge routing work for verdict/gate patterns?
3. Does `FactoryState` + `FactoryDeps` carry context through `GraphRunContext`?
4. Are `Graph.iter()` and Mermaid rendering useful bonus wins?
5. Can `definitions.py` construction patterns adapt?

## Setup

```bash
python3 -m pip install -e ".[dev]"
```

## Run tests

```bash
python3 -m pytest
```

## Type checking

```bash
python3 -m pyright
```

## Project structure

```
src/pg_factory/
  state.py      — FactoryState dataclass (mutable graph state)
  deps.py       — FactoryDeps dataclass (immutable dependencies)
  verdicts.py   — VerdictType enum + HaltResult for End returns
tests/
  conftest.py   — shared fixtures
  test_smoke.py — basic 2-node graph smoke test
  test_future_annotations.py — critical risk: from __future__ import annotations compatibility
```
