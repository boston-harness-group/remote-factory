#!/usr/bin/env bash
# Fake claude binary for E2E outer-loop wiring tests.
# Accepts all arguments that the real claude CLI would and emits valid
# stream-json output so the factory runner can parse the response.
#
# Env vars:
#   FAKE_CLAUDE_LOG — path to a file where every invocation is logged.

# ── Capture the invocation ──────────────────────────────────────────

if [ -n "${FAKE_CLAUDE_LOG:-}" ]; then
    {
        echo "---invocation---"
        echo "argv: $*"
        # Log the prompt text so tests can check which instance IDs the agent saw.
        # The prompt is the value after the -p flag.
        PROMPT=""
        while [ $# -gt 0 ]; do
            case "$1" in
                -p)
                    shift
                    PROMPT="$1"
                    ;;
            esac
            shift
        done
        echo "prompt: ${PROMPT}"
        echo "---end---"
    } >> "${FAKE_CLAUDE_LOG}"
fi

# ── Emit valid stream-json output ───────────────────────────────────

echo '{"type":"result","subtype":"success","result":"done","session_id":"fake","cost_usd":0.01,"duration_ms":100,"is_error":false,"num_turns":1,"total_input_tokens":10,"total_output_tokens":10}'
