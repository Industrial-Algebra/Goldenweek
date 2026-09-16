# Changelog

All notable changes to Goldenweek are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added — Backend-Agnostic Surface
- **`GraphicsBackend` trait** — the single graphics abstraction, mirroring
  Borsalino's `ComputeBackend` posture: construction is backend-specific
  (a `VulkanBackend` borrows a `zunesha` device), while the five runtime
  operations (`create_pipeline`, `create_buffer`, `acquire_frame`, `draw`,
  `present`) are backend-neutral.
- **Opaque-handle isolation** — `PipelineHandle`, `VertexLayout`,
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

### Design Refusals (v0.1 scope)
- No `wgpu` — raw FFI (`ash`), pure Borsalino lineage.
- No scene graph, no material system, no async frames, no depth, no
  blending, no textures in v0.1.
