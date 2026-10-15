# LLM Inference Platform

Serving one locally hosted open-source LLM to multiple concurrent clients, built from raw
Hugging Face Transformers up to vLLM, with every step measured.

The point is not to wrap an LLM in FastAPI. It is to reproduce, locally and with numbers,
the problems that optimized inference engines exist to solve.

## Approach

```
manual generation loop
  -> KV cache
  -> naive concurrent serving
  -> static batching
  -> own scheduler
  -> vLLM
  -> compare
```

Each phase must expose a measured problem before the next phase starts. A phase that runs
but produces no numbers is not finished. See
[ADR 0001](docs/decisions/0001-raw-transformers-first.md).

## Status

| Phase | Topic | State |
|---|---|---|
| 1 | manual generation loop, no cache | in progress |
| 2 | KV cache | not started |
| 3 | naive concurrent serving | not started |
| 4 | static batching | not started |
| 5 | request scheduler | not started |
| 6 | vLLM | not started |

## Docs

- [Architecture](docs/architecture.md) — components, the gateway/engine seam, phase evolution
- [Decisions](docs/decisions/) — ADRs

## Setup

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
```

Phase 1 uses `Qwen/Qwen2.5-0.5B-Instruct`. Weights download on first run to
`~/.cache/huggingface`.

## Tests

```bash
pytest -m "not slow"    # unit tests, no model needed
pytest                  # everything, downloads and loads the model
```

## Hardware

Phases 1-5 run on Apple Silicon (MPS). Phases 6+ need a Linux CUDA host, because vLLM has
no Apple GPU backend. Phase 3-5 benchmarks are re-run on that host before the vLLM
comparison, so the comparison isolates the engine rather than the hardware. See
[ADR 0002](docs/decisions/0002-split-execution-environments.md).
