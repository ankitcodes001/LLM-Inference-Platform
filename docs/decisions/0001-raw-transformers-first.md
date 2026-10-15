# ADR 0001: Start with raw Transformers, not vLLM

Status: accepted
Date: 2026-10-15

## Context

The goal is to serve one open-source LLM to multiple concurrent clients. vLLM solves that
problem already, and starting there would produce a working system in an afternoon.

It would also produce no understanding. Continuous batching, PagedAttention and prefix
caching are answers. Reaching for them first means never seeing the questions.

## Decision

Build the inference path from raw Hugging Face Transformers and move to vLLM only after
every problem vLLM solves has been reproduced and measured locally.

The sequence is fixed:

```
manual generation loop  ->  KV cache  ->  naive serving  ->  static batching
  ->  own scheduler  ->  vLLM  ->  compare
```

Each phase must produce a measured problem before the next phase starts. A phase that
"works" but has no numbers is not finished.

## Consequences

Slower. Phases 1 through 5 build an engine that phase 6 deletes.

In exchange, every vLLM feature arrives as a solution to a problem already felt:

| Phase measures | vLLM feature it motivates |
|---|---|
| quadratic recompute without a cache | KV cache management |
| GPU idle under sequential requests | continuous batching |
| long requests blocking short ones in a static batch | continuous batching |
| memory fragmentation in the hand-written scheduler | PagedAttention |

The phase 6 comparison is only meaningful because phases 1-5 produced numbers on the same
hardware. That constraint is recorded in ADR 0002.

Deleting phases 1-5 code in phase 6 is expected, not waste. The measurements survive; they
are the deliverable.
