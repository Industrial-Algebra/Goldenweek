# Critique & Future Work

> **v0.1.0 snapshot** — honest self-assessment at first release. Not a
> marketing document; a record of what is actually true.

## What v0.1.0 delivers

- One trait (`GraphicsBackend`), one complete backend (Vulkan), opaque
  handles end-to-end, WGSL-first pipeline compilation with compile-time
  validation, the full 2×2 draw matrix, textures (M1), instanced draws
  + zero-copy import (M3).
- A verification culture with receipts: every merged PR carries a
  converged review loop with reproduced-RED findings; the validation
  baseline is documented and enforced.

## What is honestly weak

- **Metal does not exist yet.** The `metal` feature is a reservation;
  substrate-first sequencing (Zunesha-Metal first) is the plan, not the
  achievement.
- **One in-flight frame.** The synchronous present cycle caps
  throughput; multi-frame overlap is a real change (timeline
  semaphores, per-frame command pools), deliberately deferred.
- **Present-mode latency.** The 00067 validation observable is a
  present-vs-acquire race inherent to the current single-frame + FIFO
  mode; the timeline-semaphore refactor is the fix, pending.
- **Import is same-queue only.** Compute-queue → graphics-queue
  visibility (the full compute→render story) waits on Zunesha ADR 0005
  Phase 2 (M2).
- **Draw arity is 2×2.** The bindings list type is generalized; the
  draw API is not (≤1 vertex-rate + ≤1 instance-rate binding per
  pipeline). Indexed drawing and more bindings are future work.
- **No headless CI GPU.** GPU tests run on the dev rig
  (`GOLDENWEEK_TEST_DEVICE` pinning per ADR 0002); CI builds and unit
  tests everything, but layer-verified GPU runs are local.

## Addressed since ideation

- ~~Windowing~~ — never accepted (refusal, ADR 0001).
- ~~In-flight resource destruction~~ — retention registry (M1 loop, r9/r10).
- ~~Spec-invalid pipelines on lenient drivers~~ — exact-set match (M3
  loop, VERIFY stage).
- ~~Foreign-handle draws~~ — owner validation at every consume site.
- ~~Opaque import trust boundary~~ — substrate `owns_buffer` (zunesha 0.3).
