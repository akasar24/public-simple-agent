# Test Harness

Tests and compares three memory implementations across five conversational scenarios.

## Setup
```bash
uv sync
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

## Running the harness
```bash
# full harness (takes ~15 min)
uv run pytest evals/ -v -s

# specific scenario
uv run pytest evals/ -v -s -k "basic_recall"

# specific memory type
uv run pytest evals/ -v -s -k "SummaryMemory"

# unit tests only
uv run pytest tests/ -v
```

## Scenarios

**basic_recall_short / basic_recall_long** -- user states their name and job early, filler messages push it back, agent is asked to recall. Short uses 5 filler messages, long uses 15. Tests whether memory retains early facts over time.

**recent_update** -- user states their name early, filler pushes it back, user then corrects themselves after the filler. Tests whether memory handles updates to previously stated facts.

**numerical_precision** -- user states an exact dollar amount early, long filler pushes it back, agent is asked for the exact figure. Tests whether memory preserves specific details vs compressing them loosely.

**preference_accumulation** -- user states three separate preferences across setup messages, filler in between, final question requires synthesizing all three. Tests multi-fact retention.

## Interpreting output

Each test prints:
```
[scenario][MemoryType] correctness=X/5, avg_chars=Y, avg_elapsed=Zs
```

- `correctness` -- how many of 5 runs the agent recalled the information correctly. Since LLMs are nondeterministic, we do this rather than simply capturing success vs failure.
- `avg_chars` -- average total character count of the context passed to the agent on the final question turn. This acts as a proxy for a token cost for the current context.
- `avg_total_elapsed` -- average total wall clock time across all agent invoke calls in the conversation, in seconds. This captures summarization overhead and per-call latency growth more honestly than timing the final call only.

## Memory implementations

| Implementation | Location |
|---|---|
| FullHistoryMemory | `src/memory/full_memory.py` |
| WindowMemory | `src/memory/window_memory.py` |
| SummaryMemory | `src/memory/summary_memory.py` |