# Memory Implementation Write-up

## Overview

The starter code uses full message history — every message ever sent gets re-passed
to the agent on every turn. This is correct but ultimately naive. We implement two alternatives
and compare all three across five conversational scenarios, evaluating correctness
and context size (used as a token cost proxy).

## Implementations

`FullMemory` passes the entire message history to the agent on every turn.
Perfect recall, zero information loss, but context grows linearly with conversation length.

`WindowMemory(window_size)` only passes the last `window_size` (configurable) messages as context. This is the cheapest token cost for a window,
which stays relatively flat regardless of conversation length, but loses early context once the window fills.
In this implementation, we still store 
full message history internally — it only trims on get_context. This means process memory grows the same as FullHistory, but token cost stays flat.

`SummaryMemory(threshold, keep_recent)` periodically compresses older messages into
a concise summary via a separate LLM call, then passes `[summary] + [recent messages]`
as context. Retains key facts from early conversation while keeping context size bounded.

## Results (see evals/README.md for test cases)

| Scenario | Full | Window (k=4) | Summary |
|---|---|---|---|
| `basic_recall_short` | 5/5, ~670 chars, ~6.0s | 0/5, ~320 chars, ~5.8s | 5/5, ~420 chars, ~6.2s |
| `basic_recall_long` | 5/5, ~1967 chars, ~13.9s | 0/5, ~339 chars, ~17.4s | 5/5, ~553 chars, ~15.1s |
| `contradiction` | 5/5, ~565 chars, ~6.8s | 5/5, ~223 chars, ~6.8s | 5/5, ~427 chars, ~6.3s |
| `numerical_precision` | 5/5, ~2004 chars, ~15.5s | 0/5, ~508 chars, ~17.5s | 5/5, ~412 chars, ~16.6s |
| `preference_accumulation` | ~4/5, ~826 chars, ~7.5s | 0/5, ~900 chars, ~9.8s | ~3/5, ~816 chars, ~7.5s |
## Analysis

`FullMemory` is the safest baseline but context grows linearly. At production scale this becomes a real
problem for both cost and context window limits.

`WindowMemory` is cheap and stays flat, but fails every recall scenario where relevant
information was introduced early. Its one win is `recent_update`, where the correction
arrives after filler and is always within the recent window. This ultimately points to a general principle
where window is best -- which is when recency matters the most. 

`SummaryMemory` is the best all-around strategy — it mostly matches `FullMemory` on correctness while
keeping context bounded. The tradeoff is an
extra LLM call every threshold messages and slight lossiness.

A few interesting notes:

* `SummaryMemory` has more overhead than `FullMemory` on shorter conversations, likely due to the extra API calls that it has to make. This makes this option only worth it at scale.
* `SummaryMemory` can also often have compounding errors. If the first summarization is missing info/is faulty, other summaries will compound those effects. This often results in the flakiest test results coming from this memory implementation.
* Even `FullMemory` doesn't always get it right on `preference_accumulation`. This is a
synthesis failure not a memory failure — the agent has all the information but doesn't
always surface all three preferences when it's not explicitly asked to. This is more of a prompting issue rather than a memory issue.
* `WindowMemory`'s char count is not always cheapest — in `preference_accumulation` 
  it actually exceeds `FullMemory`. When relevant context happens to be recent, 
  the window captures it fully, removing the usual cost advantage.
* `WindowMemory` is slower than `FullMemory` on long conversations despite passing fewer tokens — 17.4s vs 13.9s on `basic_recall_long`. Since each API call is cheap and fast, the bottleneck is network latency rather than token processing.
`FullMemory`'s larger context only meaningfully impacts latency at much greater scale than these test conversations.

## Tradeoff Summary

| | Recall             | Token Cost | Handles Updates | Extra LLM Calls |
|---|--------------------|---|---|---|
| FullHistory | ✅ Close to Perfect | ❌ Grows linearly | ✅ Yes | None |
| Window | ❌ Recency only     | ✅ Flat | ✅ If recent | None |
| Summary | ✅ Good             | ✅ Bounded | ⚠️ Lossy | Every N messages |

## Potential Extensions

`PersistentMemory` — same as `FullMemory` or `WindowMemory` but write messages to/read from a
file or database instead of directly in memory. The benefit here is that memory can survive across sessions,
which could be critical for a platform used constantly where users expect continuity
between conversations, and it can survive service restarts. The drawback is that reading from and writing
to memory is an extra I/O operation that you would need to do every set of messages that would add some time, and more
potential for things to break (I/O exceptions).

`StructuredMemory` — rather than storing raw messages, extract structured facts into a
key-value store (e.g. `{name: "Alice", budget: "$47,382", preferences: ["concise"]}`)
and inject them as context. This is more token-efficient than any message-based approach and
is easily queryable by the LLM, but it only retains what the extraction prompt explicitly captures --
nuance and implicit context are lost.