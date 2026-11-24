# Changelog

All notable changes to Goldenweek are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added — Instanced Batches & the Zero-Copy Import Seam (M3)
- **Generalized vertex bindings (breaking, pre-0.1.0):** `VertexLayout` is
  removed; `PipelineConfig::vertex_bindings: Vec<VertexBindingLayout>` where
  a binding is `{ stride, rate: VertexInputRate, attributes }`. Binding index
  is the list position, `VertexInputRate::{Vertex, Instance}` maps to the
  backend's input rate, and each attribute names its own binding. Every
  construction site migrated (both examples, all tests, all docs snippets).
- `compile_render_pipeline` validates the binding list **before any GPU
  work** and rejects with `InvalidArgument`: more than one `Vertex`-rate
  binding, more than one `Instance`-rate binding (the v0.1 draw arity), an
  attribute `location` repeated anywhere in the list (WGSL input locations
  are global, not per binding), or a binding that declares attributes with
  `stride == 0`. An empty list keeps the pre-M3 no-vertex-input meaning.
- **`draw_instanced`** and **`draw_instanced_textured`** — the full 2×2 draw
  matrix (plain/textured × single/instanced). Vertices bind at the pipeline's
  Vertex-rate binding, instances at its Instance-rate binding, then
  `vkCmdDraw(vertex_count, instance_count, 0, 0)`. Draw arity is enforced with
  `InvalidDraw`: a single-stream draw rejects a pipeline that declares an
  Instance-rate binding ("pipeline requires instances"), an instanced draw
  rejects a pipeline with no instance binding, and `instance_count == 0` is
  refused (mirroring the zero-vertex rule).
- **Retention covers both streams:** BOTH buffers and the pipeline register
  in the retention registry, and BOTH buffers get the `Arc::ptr_eq` owner
  validation — an instance buffer dropped mid-recording retires into the
  graveyard exactly like a vertex buffer. The CALLER-LOCKS contract is
  untouched: the new paths record, they never submit.
- **`import_zunesha_buffer`** — the zero-copy import seam: adopt an
  externally-created `zunesha::Buffer` (written by Borsalino's dispatch or
  zunesha's own staging) as a `GpuBuffer` with no Goldenweek-side upload and
  no copy, so a draw reads the same physical buffer. Same-graphics-queue
  writes only in v0.1 (queue FIFO ordering makes them visible to
  subsequently recorded draws); cross-family (compute-queue) writes need
  ownership transfer and stay out of scope (M2 / Zunesha ADR 0005 Phase 2).
  The import touches no queue. (The trait method is cfg-gated exactly like
  `pub mod vulkan` — `zunesha` is an optional, Linux/Windows-only
  dependency.)
- Hardware-verified on the headless Intel/Mesa rig:
  `draw_instanced_renders_per_instance_transforms` (three quads at their
  per-instance offsets, the gaps between them still the clear colour),
  `draw_instanced_renders_per_instance_colors` (red / green / blue),
  `draw_instanced_textured_samples_per_instance_uv` (each instance samples
  its own quadrant of a 2×2 atlas), `import_zunesha_buffer_draws_zero_copy`,
  the four `bindings_validation_*` rules, `draw_arity_mismatches`,
  `draw_instanced_rejects_zero_instances`, and
  `imported_buffer_drop_mid_recording_retires`.
- Refusals (documented): ONE vertex stream + ONE instance stream per draw
  (the bindings-list TYPE is the future multi-stream surface), no indexed
  draws, no instance offsets / first-instance, no cross-family writes.

### Added — Textures & Sampled Images (M1)
- **`Texture`**, **`TextureDesc`**, **`FilterMode`** — immutable RGBA8
  textures created and uploaded in one call (`create_texture`):
  tightly-packed `w * h * 4` payloads and non-zero extents validated with
  checked arithmetic (no panic on `u32::MAX` extents), with the image, view,
  sampler and descriptor pool torn down in dependency-safe order.
- **`draw_textured`** — the textured draw path: the texture is bound as
  **set 0**, sampled image at binding 0 and its sampler at binding 1 (the
  WGSL/naga-expressible form of a combined sampler — WGSL has no
  combined-image-sampler type).
- **Retention/leak contract** — a texture dropped while a recording that
  references it is still pending retires into the backend's graveyard and is
  destroyed at that frame's completion; a texture that outlives its backend
  leaks with a stderr note instead of issuing destroys on a dead device; a
  poisoned backend (unconfirmable completion) leaks rather than destroys
  unconfirmed GPU work. The tombstone is set under the same lock the public
  drop holds across its check-and-destroy, so the teardown race is closed.
- **Owner validation** — `draw_textured` rejects a texture created by a
  different backend (`Arc::ptr_eq` on the bindings registry): its descriptor
  set was allocated against another backend's set-layout object and its
  retention belongs to another registry.
- Pinned by hardware tests: `textured_quad_samples_exact_texels` (a 2×2
  atlas over a fullscreen quad, every quadrant asserted
  channel-order-aware), `draw_textured_rejects_non_open_phase`,
  `texture_creation_rejects_bad_payload`, plus the `textured` example.

### Added — Swapchain Resize/Recreate (Increment 7)
- **`VulkanBackend::recreate_swapchain(&mut self, requested)`** — tears
  down and rebuilds the swapchain + extent-sized depth buffer at a new
  size. The render pass is reused, so previously compiled pipelines stay
  valid across a resize (hardware-verified: pipeline compiled at 256×256
  renders green at 128×128).
- `requested` clamps to the surface limits on undefined-extent surfaces
  (headless); a defined current extent (real windows) always wins.
  `resolve_extent` unit-tested for all three behaviours.
- Recreating after a **dropped unpresented frame** is safe — the stale
  recording is discarded exactly as the next acquire would (mirrors the
  increment-5 dropped-frame guarantee).
- Refused with a new `GraphicsError::RecreateRefused` when the surface
  **format** changed (render pass + pipelines are format-bound; caller
  drops pipelines and recompiles). Failure semantics are two-windowed
  with the boundary at the `oldSwapchain` handoff CALL: failures before
  it (caps/format/extent) leave the old swapchain usable; the handoff is
  required for native windows and retires the old swapchain *whether or
  not creation succeeds* (WSI spec), so failures at or after it leave
  the backend swapchain-less (retired handle destroyed + cleared;
  replacement built under the `InitScratch` discipline) — drop the
  backend and construct a new one.
- Readback capability is tracked: surfaces without `TRANSFER_SRC`
  support get COLOR_ATTACHMENT-only swapchain images and
  `read_pixels` returns an error instead of issuing invalid transfers.
- Documented contract: no live `Frame` may outlive a recreate.

### Added — Depth Testing & Blending (Increment 6)
- **`DepthState`** (`PipelineConfig::depth`): fixed-function depth test —
  `CompareOp` (Never/Less/Equal/LessEqual/Greater/NotEqual/GreaterEqual/
  Always) + `write_enable`; defaults to `Less` + write.
- **`BlendMode`** (`PipelineConfig::blend`): additive color blending with
  `BlendFactor` pairs (alpha factors mirror color factors, v0.1); presets
  `alpha()`, `additive()`, `multiply()`.
- Both options default to `None` — existing pipelines compile and render
  identically (additive surface).
- The Vulkan render pass now carries a depth attachment (D32 → D24S8 → D16
  by support; the depth image is always allocated — documented v0.1
  simplification so any pipeline is subpass-compatible); `VulkanBackend::
  depth_format()` exposes the chosen format.
- Hardware-verified on a real device: depth — a near triangle drawn first
  survives a far overdraw (with a depth-off control frame asserting painter
  order); blend — 0.5-alpha magenta over the 0.1 gray clear reads back
  driver-dependent composites within ±1 (observed `[140,12,140,191]`; the
  clear byte itself converts to 25 or 26 depending on the device), with
  alpha-channel assertions pinning the mirrored-factor behaviour. New tests:
  `depth_test_rejects_far_overdraw`, `alpha_blend_composites_over_clear`,
  plus config-default unit tests.
- Constructor robustness (review round 1): an RAII `InitScratch` guard
  destroys every partially-created object if `VulkanBackend::new` fails
  after the swapchain — including the depth trio — so error paths leak
  nothing on the caller's device.

### Changed
- Zunesha dependency switched from path (`../Zunesha`) to the published
  crates.io version (`0.1`) — Goldenweek now builds and verifies against
  the released crate (13 tests + triangle example pass on RTX 5080
  against registry zunesha 0.1.0).

### Added — Backend-Agnostic Surface
- **`GraphicsBackend` trait** — the single graphics abstraction, mirroring
  Borsalino's `ComputeBackend` posture: construction is backend-specific
  (a `VulkanBackend` borrows a `zunesha` device), while the five runtime
  operations (`create_pipeline`, `create_buffer`, `acquire_frame`, `draw`,
  `present`) are backend-neutral.
- **Opaque-handle isolation** — `PipelineHandle`, `VertexBindingLayout`,
  `GpuBuffer`, `Frame` expose no backend types. Renderers target the trait and
  link no FFI.
- **No windowing (design refusal)** — accepts an external `SurfaceHandle`;
  Goldenweek never owns a window.
- **WGSL-first shading** — shaders are WGSL source, lowered via `naga`
  (parse → validate → SPIR-V) at pipeline creation. No offline shader
  compilation step in the API surface.
- **`read_pixels`** — public verification API: barriers the presented image
  and reads it back through Zunesha's staging path. Structural rendering
  verification (the numerical-exactness posture stays with Borsalino).

### Added — Vulkan Backend (`vulkan` feature, complete)
- **Shared-device integration** — `VulkanBackend<'a>` *borrows* a
  `zunesha::vulkan::VulkanDevice` (borrow-checked device lifetime; the device
  is shared with Borsalino per Zunesha ADR 0001).
- **Swapchain + frame lifecycle** — surface-support check, format/extent
  selection, FIFO present mode; `acquire_frame` / `draw` / `present` with
  acquire/render semaphore chaining and a per-frame fence (reset only
  immediately before the submit that signals it — dropped frames cannot
  deadlock the next acquire).
- **Render pass + graphics pipeline** — image views, render pass, and
  framebuffers built at construction; `compile_render_pipeline` via naga;
  fully-baked `PipelineConfig` with dynamic viewport/scissor.
- **Zero-copy compute→render interop** — `create_buffer` returns a
  `zunesha::Buffer`; the underlying `VkBuffer` is bound directly for vertex
  draws. A buffer Borsalino fills by compute can be drawn unchanged.
- **Verified rendering** — headless-surface test path rasterizes a triangle
  and asserts pixel values via `read_pixels` (magenta interior, clear-color
  exterior), stable across B8G8R8A8/R8G8B8A8 formats.

### Known Hardware Findings
- NVIDIA proprietary drivers do not support swapchains on
  `VK_EXT_headless_surface` (extension present but stubbed); headless
  graphics verification runs on Intel/Mesa. Windowed NVIDIA is unaffected.

### Added — Documentation
- **ADR 0002** — headless verification surfaces and device pinning: the
  empirical driver matrix (NVIDIA stubs, Intel/Mesa works, llvmpipe crashes)
  and the `GOLDENWEEK_TEST_DEVICE` convention, promoted from prose to a
  decision record (carried by four consecutive research pulses).
- Doc-sync pass: README quick-start now shows the real construction path
  (borrowed Zunesha device + `goldenweek::init`), and the backend tables in
  README and crate docs reflect the complete Vulkan backend.

### Added — `examples/triangle` (vulkan feature)
- The magenta-triangle readback verification, promoted to a runnable binary
  — pins the public API to something a consumer can target (report
  recommendation, carried by three pulses). Pins via `GOLDENWEEK_TEST_DEVICE`,
  skips gracefully on device-less hosts, `required-features = ["vulkan"]`.

### Changed — Spec-safe verification (found via the example, 2026-09-16)
- **`read_pixels(&frame)` now reads the *rendered* image, not the presented
  one** — it flushes the pending recording and fence-waits before the
  barrier/copy. Reading after `vkQueuePresentKHR` was spec-fragile: presented
  contents are undefined (WSI may clobber), and the current Mesa headless
  WSI does. Call between `draw` and `present`; repeat calls are idempotent.
- **`present` accepts both frame-loop phases** (`Open` — submits with
  semaphore chaining; `Flushed` — presents with no waits, the flush already
  fence-waited). Internal `RenderPhase` state machine (`Open` / `Flushed` /
  `Idle`) replaces the `recording: Option<u32>`.
- **`acquire_frame` records an explicit clear in two sub-rects** —
  full-surface clears (load-op or explicit) take the driver's fast-clear
  metadata path, which the current Mesa decompresses to float garbage on
  headless WSI swapchain images; sub-surface clears write real texels
  (ADR 0002, findings 1–2).
- `Frame::peek_index` is now public — verification consumers need the image
  index after `present` consumes the frame handle.

### Design Refusals (v0.1 scope)
- No `wgpu` — raw FFI (`ash`), pure Borsalino lineage.
- No scene graph, no material system, no async frames, no depth, no
  blending, no textures in v0.1.
