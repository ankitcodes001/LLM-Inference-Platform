# Architecture

How this system is put together, and why it changes shape phase by phase.

## Goal

Serve one locally hosted open-source LLM to multiple concurrent clients, and understand
every bottleneck along the way by measuring it rather than reading about it.

The project deliberately starts with raw Hugging Face Transformers instead of an optimized
serving engine. Each phase exposes a concrete, measured problem; that problem is the reason
the next phase exists.

## The one decision that matters: the seam

Across phases the serving layer is written once. The execution engine is rewritten
repeatedly:

| Phase | Engine |
|---|---|
| 1 | manual generation loop, no KV cache |
| 2 | manual loop, KV cache enabled |
| 4 | static batching |
| 5 | own scheduler, dynamic admission |
| 6 | vLLM |

So the architecture puts a seam between the two:

```
Clients
   |
Gateway          HTTP, validation, request ids, streaming
   |
   |  generate(prompt, config) -> GenerationResult        <- the seam
   |
Engine           swapped every phase
   |
Model / GPU
```

The gateway calls a variable, never a hardcoded engine. Replacing the engine is a
one-line change at startup, not a rewrite of every route.

### The interface is written later, on purpose

A `Protocol` with one implementation is speculative abstraction. Phases 1 and 2 have a
single engine (the same class with `use_cache` flipped), so no interface is written.

The second real implementation arrives in phase 4. The `Protocol` is introduced then.

Until then the discipline is weaker but free: every engine exposes the same method
signature, `generate(prompt, cfg) -> GenerationResult`.

## Components

```
inference/              depends on nothing
  config.py             ModelConfig, GenerationConfig
  loader.py             model + tokenizer load, device resolution
  sampling.py           logits -> token id (pure tensor functions)
  manual_loop.py        phase 1-2 engine
  batch_loop.py         phase 4 engine

scheduler/              depends on inference
  queue.py              bounded request queue
  scheduler.py          admission, active set, step loop

gateway/                depends on scheduler and inference
  schemas.py            request/response validation
  app.py                FastAPI routes

benchmarks/
  load_gen.py           concurrent client simulator
```

### Dependency rule

```
gateway  ->  scheduler  ->  inference

Never the reverse.
```

`inference/` must not know a server exists. This buys three things:

- `inference/` is testable without starting a server
- `sampling.py` is testable without loading a model (pure functions, CPU, milliseconds)
- phase 6 can delete `manual_loop.py` and `batch_loop.py` without touching anything above

## Phase 1 in detail

Phase 1 has no server. It is a library plus four scripts, exercised directly.

### Request flow

```
prompt
  -> apply chat template, tokenize            (once)
  -> loop, once per output token:
       forward pass over the full sequence    (no cache)
       take logits at the last position
       pick the next token id
       append it, check stop conditions
  -> decode the new token ids back to text
```

### Why the full sequence every step

Phase 1 sets `use_cache=False` deliberately. Every step re-processes all earlier tokens,
so total work grows quadratically with output length:

```
step 1   6 positions computed, 6 new
step 2   7 positions computed, 1 new
step 3   8 positions computed, 1 new
step 4   9 positions computed, 1 new
         30 position-computes total, 9 of them useful
```

Enabling the cache in phase 1 would leave phase 2 with nothing to compare against. The
waste is the baseline.

### File ownership

| File | Responsibility | Needs the model |
|---|---|---|
| `config.py` | settings only | no |
| `loader.py` | load model and tokenizer, resolve device and dtype | yes |
| `sampling.py` | logits tensor -> token id | no |
| `manual_loop.py` | drive the loop, record per-step telemetry | yes |

`sampling.py` never imports the model. That is the smallest seam in the project and the
reason its tests run in milliseconds.

### Testing strategy

Three layers, deliberately separate.

**Unit tests** cover `sampling.py` against hand-written logits with a vocabulary of four,
so every expected value can be verified by hand. No model, no GPU, ~50ms.

**Parity test** asserts that the manual greedy loop produces token ids identical to
`model.generate(do_sample=False)`.

Greedy is used because it is deterministic: sampling would make two runs differ and the
assertion meaningless. One assertion covers the whole chain, because any broken step
changes the output:

| Broken step | Symptom |
|---|---|
| chat template or tokenization | first token differs |
| wrong logits position, wrong argmax | first token differs |
| token append | second token differs |
| stop condition | output length differs |

The index at which outputs diverge localises the bug.

**Benchmarks** are not pass/fail. They emit numbers: TTFT, decode tokens/sec, total
latency, peak memory, and per-step latency across prompt lengths of 64, 256 and 1024
tokens.

TTFT and decode rate are reported separately because they scale differently. Without a
cache, per-step latency climbs as output grows, so decode rate degrades within a single
request. That curve is phase 1's most useful output.

### Exit criteria

Phase 2 does not begin until all four hold:

1. the parity test passes
2. benchmark numbers for three prompt lengths are recorded in `docs/benchmark-results.md`
3. the no-cache baseline is measured, not estimated
4. `docs/inference-internals.md` answers what the logits tensor is, why only the last
   position is used, and what each sampling knob does, with real output pasted in

## Phase evolution

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| `config.py` | created | - | + validated mirror | - | + queue limits |
| `loader.py` | created | - | - | - | - |
| `sampling.py` | created | - | - | + batch support | - |
| engine | manual | + cache | - | batched | - |
| `gateway/` | - | - | created | - | + admission |
| `scheduler/` | - | - | - | - | created |

`loader.py` and `sampling.py` are written once and survive to phase 6. A file that stops
changing is a file whose boundary was drawn correctly.

## Per-phase exit criteria

A phase ends when it produces a number, not when the code runs.

| Phase | Exit criteria | Number produced |
|---|---|---|
| 1 | manual greedy matches `model.generate()` token for token | tokens/sec without cache |
| 2 | cache on and off both measured | speedup ratio, memory delta |
| 3 | 1/5/10/25/50 concurrent clients | p50/p95/p99 per concurrency, GPU idle % |
| 4 | per-request vs batched execution | throughput gain, padding waste % |
| 5 | bounded queue, admission, timeout | queue wait, rejection rate, saturation point |

## Known risks

| Risk | Impact | Mitigation |
|---|---|---|
| MPS is not CUDA | phase 3-5 numbers will not transfer to a GPU box | re-run phase 3-5 benchmarks on the GPU box before phase 6; publish both |
| transformers v5 cache API | phase 2 code will not match v4 tutorials | read the installed source, not blog posts |
| MPS memory reporting is partial | phase 2 memory delta is fuzzy | log `torch.mps.current_allocated_memory()` and process RSS together |
| Python GIL | phase 3 concurrency may be illusory | measure both async and thread execution, report the difference |

The last one is phase 3's real lesson: what "concurrent request" means when there is one GPU.
