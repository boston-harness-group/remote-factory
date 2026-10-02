## Test Infrastructure

This project has reusable test infrastructure for connected scope tests:

- `factory.testing.FakeAgent` — contract-enforcing fake for agent calls (use instead of ad-hoc mocks)
- `factory.testing.DummyTask` — Task subclass with fixed instances and deterministic verify()
- `auto_write_outputs=False` on `WorkflowExecutor` — disables automatic file writes so you can test write contracts via FakeAgent

Use these for connected scope tests involving the workflow executor or agent pipeline.
