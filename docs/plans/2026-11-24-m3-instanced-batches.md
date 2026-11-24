# M3 — Instanced & Dynamic Vertex Batches

**Status:** contract (generation-dispatch ready)
**Date:** 2026-11-24
**Operator decisions (ask_user, this date):** (1) generalized bindings
list — `VertexLayout` becomes a per-binding list with input rates; (2)
full 2×2 draw matrix — `draw_instanced` AND `draw_instanced_textured`
ship in M3.
**Rides this branch:** the owed M1 CHANGELOG entry.

## Goal

Serve the DrawOp-diff consumer: instanced quads (per-instance transform
+ color) over static geometry, and glyph-atlas batches (instanced +
textured). Clear the 0.1.0 release bar (M1 + M3).

## API (verbatim — implement exactly)

```rust
/// Input rate of a vertex-input binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexInputRate {
    /// One element per vertex.
    Vertex,
    /// One element per instance.
    Instance,
}

/// One vertex-input buffer binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBindingLayout {
    /// Number of bytes between consecutive elements.
    pub stride: u32,
    /// Whether the binding advances per vertex or per instance.
    pub rate: VertexInputRate,
    /// Attributes read from this binding (offsets are within one element).
    pub attributes: Vec<VertexAttribute>,
}

// PipelineConfig: REPLACE `vertex_layout: VertexLayout` with
    /// Vertex-input bindings, in binding order (binding index = list
    /// position). v0.1 draw arity: AT MOST ONE Vertex-rate binding and
    /// AT MOST ONE Instance-rate binding (compile_render_pipeline
    /// rejects more with InvalidArgument); the list shape is the
    /// future multi-stream surface.
    pub vertex_bindings: Vec<VertexBindingLayout>,

// GraphicsBackend (and VulkanBackend + stub):
    /// Record a non-indexed instanced draw into `frame`.
    ///
    /// `vertices` is bound at the pipeline's Vertex-rate binding,
    /// `instances` at its Instance-rate binding; `vertex_count` and
    /// `instance_count` are the draw's first parameters.
    fn draw_instanced(
        &self,
        frame: &mut Frame,
        pipeline: &RenderPipeline,
        vertices: &GpuBuffer,
        instances: &GpuBuffer,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<()>;

    /// Record a non-indexed instanced textured draw into `frame`
    /// (the M1 texture set + the M3 instance binding).
    fn draw_instanced_textured(
        &self,
        frame: &mut Frame,
        pipeline: &RenderPipeline,
        vertices: &GpuBuffer,
        instances: &GpuBuffer,
        vertex_count: u32,
        instance_count: u32,
        texture: &Texture,
    ) -> Result<()>;

// The zero-copy import seam (v0.1 tier: same graphics queue):
    /// Adopt an externally-created `zunesha::Buffer` as a `GpuBuffer`
    /// WITHOUT any Goldenweek-side upload — the buffer's contents were
    /// written by another client of the shared device (Borsalino's
    /// dispatch, or zunesha's own staging). Zero copies, zero
    /// re-uploads: the draw reads the same physical buffer.
    ///
    /// Same-graphics-queue writes only in v0.1: queue FIFO ordering
    /// makes the writes visible to subsequently recorded draws.
    /// Cross-family (compute-queue) writes require ownership transfer
    /// — deliberately out of scope (M2 / Zunesha ADR 0005 Phase 2).
    fn import_zunesha_buffer(&self, buffer: zunesha::Buffer) -> Result<GpuBuffer>;
```

`VertexLayout` is REMOVED (pre-0.1.0 clean break — migrate every
construction site: both examples, all tests, all docs snippets, to the
bindings list).

## Validation (compile_render_pipeline, InvalidArgument)

- More than one `Vertex::`-rate binding, or more than one
  `Instance::`-rate binding → reject (draw-API arity, above).
- Attribute `location` duplicated ACROSS bindings → reject (WGSL input
  locations are global).
- Binding with `attributes` non-empty and `stride == 0` → reject.
- Empty `vertex_bindings` → keep the current no-vertex-input meaning
  (stride-0 pipelines today).

## Validation (draw time, InvalidDraw)

- `draw`/`draw_textured` on a pipeline WITH an Instance-rate binding →
  reject ("pipeline requires instances").
- `draw_instanced`/`draw_instanced_textured` on a pipeline WITHOUT an
  Instance-rate binding → reject ("pipeline has no instance binding").
- `instance_count == 0` → reject (mirrors vertex_count).

## Vulkan mapping notes

- Binding index = list position; `input_rate` from `rate`;
  `VertexInputAttributeDescription.binding` = the attribute's own
  binding's index (no longer hardcoded 0).
- At record time: `bind_vertex_buffers` twice — vertices at its
  binding index, instances at theirs — then
  `cmd_draw(vertex_count, instance_count, 0, 0)`.
- **Retention (the r9/r10 machinery applies verbatim):** BOTH buffers
  (vertices + instances) and the pipeline register in the retention
  registry; BOTH buffers get `Arc::ptr_eq` owner validation. An
  instance-buffer drop mid-recording retires to the graveyard exactly
  like a vertex buffer. The CALLER-LOCKS contract is untouched — no
  new queue-op sites beyond what draw already does.
- `import_zunesha_buffer`: boxes a `GpuBufferInner` around the GIVEN
  buffer (no upload, no copy), same `alive`/`retention` refs — the
  drop protocol is identical. The import does not touch the queue.

## Tests (TDD — RED first, all `#[serial]`, rig-pinned as usual)

1. `bindings_validation_*` — each InvalidArgument rule above (4 cases).
2. `instanced_pipeline_compiles` — WGSL with per-instance `@location`
   inputs compiles against a two-binding config.
3. `draw_instanced_renders_per_instance_transforms` — 3 instances,
   per-instance x-offset; readback shows 3 distinct quads at the
   offsets.
4. `draw_instanced_renders_per_instance_colors` — per-instance color;
   readback shows 3 distinct colors.
5. `draw_instanced_textured_samples_per_instance_uv` — 2×2 atlas,
   per-instance UV offset; readback shows each instance sampling a
   different texel quadrant.
6. `draw arity mismatches` — both InvalidDraw rules (2 cases).
7. `import_zunesha_buffer_draws_zero_copy` — buffer created via the
   shared `zunesha` device DIRECTLY (never through Goldenweek's
   upload), imported, drawn, verified by readback. This is the seam
   proof: Goldenweek performs no copy.
8. `imported_buffer_drop_mid_recording_retires` — imported instance
   buffer dropped between draw_instanced and present → graveyard, no
   00922 (mirrors the r9 regression shape).
9. Instance-count-0 rejection.
10. Stub backend: all four draw-family methods + import return
    `NoBackend`.
11. Existing tests + examples migrate to `vertex_bindings` and stay
    green (migration is part of the diff, not a follow-up).

## Non-goals (v0.1 refusals, documented)

- Multi-buffer draw arity (draw takes ONE vertex + ONE instance
  buffer; the bindings-list TYPE is the future surface).
- Indexed draws (unchanged from today).
- Instance offsets / first-instance.
- Cross-family (compute-queue) writes — M2 / ADR 0005 Phase 2.

## Gates

Both feature configs: test, clippy ×2, `RUSTDOCFLAGS="-D warnings"
cargo doc --no-deps` ×2, fmt, both examples; validation layers with
the settled baseline (00067 pair) only. GPU tests `#[serial]`,
headless Intel/Mesa rig, `GOLDENWEEK_TEST_DEVICE` overridable.

## CHANGELOG (same branch)

- Add the owed **M1 — Textures & sampled images** entry (create_texture,
  draw_textured, the retention/leak contract, owner validation — from
  the merged PR #13).
- Add the **M3** entry summarizing the above.
