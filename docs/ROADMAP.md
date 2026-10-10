# Goldenweek Roadmap

- **Status:** living document — sequenced by consumer need, per ADR 0001's
  reversibility clause (nothing lands without a concrete consumer).
- **Primary consumers today:** none published (Goldenweek is pre-0.1.0);
  the alignment target is **Miriami** (graphics-framework sibling, currently
  no-code ideation) and the Borsalino↔Miriami compute→render seam.

## Current state (2026-10-10)

Vulkan backend complete through increment 7: trait surface, Zunesha device
borrow, swapchain + frame lifecycle, WGSL→SPIR-V pipelines (naga), verified
draw + `read_pixels`, depth testing + blending (increment 6), swapchain
resize/recreate (increment 7). Metal is staged behind Zunesha-Metal
(substrate-first sequencing).

## Miriami alignment

Miriami (see its
[ideation synthesis](https://github.com/Industrial-Algebra/Miriami/blob/main/docs/architecture/00-ideation-synthesis.md))
is a functional-reactive, trans-graphical rendering framework: a retained
`Surface<Msg>` scene algebra lowered through layout/diff into backend-neutral
`DrawOp`s, committed via **Miriami's own `GraphicsBackend` trait**.

**The layering, stated once to prevent confusion:** two traits share the name.
Miriami's `GraphicsBackend` is the *DrawOp-commit* interface; Goldenweek's
`GraphicsBackend` is the *device-level* pipeline/draw interface. The
`miriami-native` crate is the adapter that implements the former on top of the
latter. Goldenweek never learns about DrawOps, scenes, or styles.

**What Miriami's ideation specifically asks of a Goldenweek backend:**

| Miriami need | Goldenweek answer |
|---|---|
| Glyph/image rendering delegated to the backend (Miriami refusal #4) | **M1** — sampled images + staging upload (glyph atlases, image DrawOps) |
| Grade-1 "living" fields (amari-automata / CA) rendered as GPU resources | **M2** — compute-written textures sampled by render (the image-side zero-copy seam) |
| Diffed DrawOp batches committed per frame | **M3** — instanced draws over Borsalino-filled vertex buffers (buffer-side zero-copy seam, exists since increment 5) |
| One backend per participant window, shared device | **M4** — multi-surface validation (N backends, one Zunesha device) |
| Native tier on macOS | Zunesha-Metal (in flight, separate session) → Goldenweek-Metal |
| Reactive per-frame commit (FIFO, deterministic core) | synchronous acquire→draw→present already matches; resize handled (increment 7) |

**What Goldenweek refuses for Miriami** (recording the boundary now):

- **No wasm / WebGPU target.** Raw-FFI Vulkan over Zunesha is the design
  (README § Design refusals); Miriami's wasm tiers run on canvas and
  cliffy-gpu, never on Goldenweek.
- **No scene graph, style algebra, widgets, or text *shaping*.** Those are
  Miriami's layers. Goldenweek accepts rasterized glyph atlases *as textures*
  (M1 upload path) — shaping and layout stay host/Miriami-side.
- **No numerical-exactness mandate.** Miriami's own principles doc reframes
  P2 as *exact algebra, approximate numerics* for GUIs — matching
  Goldenweek's existing "structural correctness, not numerical exactness"
  posture (Borsalino owns numerics).

## Milestone ladder

- **M1 — Textures & sampled images** *(next; the keystone)*
  Image creation, staging upload, sampled-image pipeline bindings. Serves
  glyph atlases, image DrawOps, and is the prerequisite for M2.
- **M2 — Compute-written textures (image-side zero-copy)**
  Borsalino-dispatched writes sampled directly by Goldenweek. **Depends on
  Zunesha cross-queue synchronization (candidate ADR 0004)** — compute-family
  writes and graphics-family samples of an EXCLUSIVE resource require
  ownership release/acquire. Until then, uploads (M1) are the only path.
- **M3 — Instanced & dynamic vertex batches**
  Instanced draws + vertex streams from compute-filled buffers (the existing
  `zunesha::Buffer` zero-copy seam). Serves DrawOp diff batches.
- **M4 — Multi-surface / multi-window**
  Validate N `VulkanBackend`s (one per surface) over one shared Zunesha
  device; per-surface frame state today is backend-owned, so this is
  validation + any fixes, not a redesign.
- **Metal backend** — after Zunesha-Metal lands (escape hatches:
  `raw_device()` / `raw_buffer()`), mirroring Borsalino Phase 3 sequencing.
- **0.1.0 release** — when the vertical slice above serves `miriami-native`'s
  first prototype: M1 + (M2 or M3) is the candidate bar, per the
  consumer-need rule.

## Sequencing notes

- The ladder is dependency-ordered (M1 → M2/M3 → M4), not date-ordered.
- M2's coupling to Zunesha ADR 0004 is deliberate: the cross-queue question
  is load-bearing for the whole compute→render interop story and belongs to
  the substrate, not to Goldenweek working around it.
- Miriami is ideation; if its `GraphicsBackend` trait shape changes before
  code exists, only the adapter note above is affected — the device-level
  surface Goldenweek exposes is independent of DrawOp representation.
