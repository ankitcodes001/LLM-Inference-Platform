# ADR 0002: Phases 1-5 run on Apple Silicon, phases 6+ on a CUDA box

Status: accepted
Date: 2026-10-15

## Context

Development happens on an Apple Silicon Mac with an MPS backend. vLLM has no usable Apple
GPU backend; on macOS it is CPU-only and experimental.

vLLM's CUDA builds also require compute capability 7.0 or higher. That rules out some free
GPU tiers:

| GPU | Compute capability | vLLM | bfloat16 | FlashAttention 2 |
|---|---|---|---|---|
| T4 | 7.5 | yes | no | no, falls back to xformers |
| P100 | 6.0 | no | no | no |
| L4, A10G, A100 | 8.0+ | yes | yes | yes |

## Decision

Phases 1 through 5 run locally on MPS. Phases 6 onward run on a Linux CUDA host.

Before phase 6 begins, the phase 3-5 benchmarks are re-run on that CUDA host. Both sets of
numbers are published.

## Consequences

Without the re-run, the phase 5 vs phase 6 comparison would be invalid: it would measure
MPS against CUDA, not a hand-written scheduler against vLLM. The re-run is what isolates
the variable.

Phase 1-5 numbers remain useful on their own. They show relative effects (cache on vs off,
batched vs sequential) that hold across hardware even when absolute latencies do not.

Secondary effect: `dtype` cannot be assumed. The local runs use bfloat16 on MPS; a T4 run
would have to use float16. When the hardware changes, dtype must be held constant across
the pair being compared, or recorded as a second variable.

If a shared free tier is used for the final numbers, that fact is stated in the README.
Shared GPUs produce noisy tail latencies, which makes p99 claims unreliable. Relative
comparisons stay valid; absolute figures do not.
