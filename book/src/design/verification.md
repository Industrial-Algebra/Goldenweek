# Verification

Goldenweek's verification stance is **structural correctness**: valid
surfaces, pipelines, and frames — measured against the Vulkan
validation layers, reproducibly. (Numerical exactness is Borsalino's
concern, by design.)

## The gate

Every change runs, on both feature configurations:

- full test suite (Vulkan tests serialised; headless-verification rigs
  per ADR 0002),
- **Vulkan validation layers with zero findings** — the accepted
  baseline is a documented pair (a present-mode latency race, 00067,
  flaky by nature) plus deliberately-leaked-buffer observables (05137)
  in teardown tests,
- clippy `-D warnings` (including `--all-features`),
- rustdoc `-D warnings`, fmt, both examples.

## What the layers actually caught

The discipline is not ceremonial. Layer-enabled runs found real defects
across the review loops — in-flight buffer destruction (00922),
in-flight pipeline destruction (00765), foreign-resource command
buffers (00059), concurrent queue host access (Threading), spec-invalid
pipelines that lenient drivers render anyway (Input-07904 — the
shader/bindings exact-set match exists because the layers flagged it).

## Review culture

Every PR runs a findings-only review loop to **demonstrated
convergence** (zero P1/P2 findings in a fresh round, with the previous
rounds' fixes attacked by the next round). Findings are reproduced RED
before adoption, dispositioned inline with provenance tags (original
defect vs fix-borne), and the loop-closure comment records the full
tally. The M1 loop converged after 13 rounds and 21 findings; the M3
loop after 2 — the difference was the M1 loop's consolidation work
(the caller-locks protocol and the retention registry), which M3's
surface inherited and the M3 round-2 audit verified held.
