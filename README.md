# Goldenweek

[![CI](https://github.com/Industrial-Algebra/Goldenweek/actions/workflows/ci.yml/badge.svg)](https://github.com/Industrial-Algebra/Goldenweek/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

Thin GPU **graphics** abstraction for the Industrial Algebra ecosystem.

> Render pipelines, frames, and presentation. Compute stays in
> [`borsalino`](https://crates.io/crates/borsalino).

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — purpose, the graphics surface, and relationship to Zunesha.
- **[docs/adr/0001 — Graphics surface and scope](docs/adr/0001-graphics-surface-and-scope.md)** — the v0.1 refusal list and rationale.

Goldenweek is the graphics sibling of
[Borsalino](https://github.com/Industrial-Algebra/Borsalino). Where Borsalino
is a minimal, synchronous **compute** abstraction over Metal and Vulkan,
Goldenweek applies the same discipline to **rendering**:

- Author WGSL vertex + fragment shaders.
- Compile a render pipeline.
- Acquire a frame, draw into it, present it to a surface.
- No windowing. No scene graph. No materials. No async runtime.

The graphics surface is necessarily stockier than compute's — presentation,
render passes, and pipeline state are irreducible — but the *philosophy* is
Borsalino's: opaque-handle backend isolation, WGSL-first via
[naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga), synchronous by default,
and a documented refusal list that defines the scope.

## Quick Start

```rust,ignore
// cfg(feature = "vulkan") — construction borrows the shared Zunesha device
// (ADR 0001: one device, shared with Borsalino's compute).
let device = zunesha::vulkan::VulkanDevice::init_with(
    zunesha::InitRequest::prefer_graphics(),
)?;

// Caller provides the platform surface (see SurfaceHandle docs) —
// Goldenweek never owns a window.
let gpu = goldenweek::init(&device, surface)?;

let pipeline = gpu.compile_render_pipeline(
    "vs_main", goldenweek::kernels::FLAT_TRIANGLE_VERT,
    "fs_main", goldenweek::kernels::FLAT_TRIANGLE_FRAG,
    &PipelineConfig::default(),
)?;

let vertices = gpu.create_buffer(&[
    0.0f32, 0.5,   // top
   -0.5f32, -0.5,  // bottom-left
    0.5f32, -0.5,  // bottom-right
])?;

let mut frame = gpu.acquire_frame()?;
gpu.draw(&mut frame, &pipeline, &vertices, 3)?;
gpu.present(frame)?;
```

## Relation to Borsalino

| | Borsalino | Goldenweek |
|---|---|---|
| **Domain** | GPU compute | GPU graphics |
| **Shader stages** | `@compute` | `@vertex` + `@fragment` |
| **Surface** | n/a (no presentation) | `SurfaceHandle` (caller-owned) |
| **Backends** | Metal (`objc` FFI), Vulkan (`ash`) | Metal (`objc` FFI), Vulkan (`ash`) |
| **Translation** | WGSL → MSL / SPIR-V via naga | WGSL → MSL / SPIR-V via naga |
| **Device model (v0.1)** | owns its own compute device | owns its own graphics+present device |
| **Default** | synchronous dispatch | synchronous frame loop |

The two are **parallel** device libraries, not layered. Compute→render
interop (particle systems, compute-then-draw) is a documented future concern
— see [Architecture](#architecture).

## Backends

| Backend | Platform | Feature | Status |
|---|---|---|---|
| Metal | macOS (Apple Silicon) | `metal` | 🚧 pending Zunesha-Metal (substrate-first) |
| Vulkan | Linux, Windows | `vulkan` | ✅ v0.1 — swapchain, pipeline, draw, verified readback |
| Stub | Any | (none) | ✅ `NoBackendStub` — safe fallback |

v0.1 ships the trait, opaque handle types, and the stub. Both backends hand-roll
their FFI (no `wgpu`) to match Borsalino's auditability — a deliberate choice
that accepts graphics' larger raw-FFI surface in exchange for full control and
character fidelity.

## Design refusals (v0.1)

Goldenweek follows IA Design Principle 4 (Architectural Refusal) — each refusal
collapses complexity and is reversible only via a documented ADR:

| # | Refusal | Rationale |
|---|---|---|
| 1 | **No windowing.** Accept an external `SurfaceHandle`. | Keeps Goldenweek a pure device library; the caller owns the window. |
| 2 | **No `wgpu` dependency.** Hand-roll Metal/Vulkan FFI. | Maximum auditability; true Borsalino lineage. |
| 3 | **No scene graph, no materials, no text/font.** | Those belong to Miriami (the framework layer). Goldenweek is the dumb device layer. |
| 4 | **No async-by-default.** Synchronous `acquire → draw → present`. | Matches Borsalino's dispatch model and the WASM-host story in Baedeker. |
| 5 | **No depth/stencil, blend, or multisample at v0.1.** | Flat-shaded triangles prove the architecture first. |
| 6 | **No textures at v0.1.** | Grow by proven need. |

## Architecture

```
GraphicsBackend trait
    │
    ├── (v0.2) MetalBackend     (metal.rs)
    │   ├── naga WGSL → MSL translation (vertex + fragment)
    │   └── objc_msgSend FFI
    │
    ├── (v0.2) VulkanBackend    (vulkan.rs)
    │   ├── naga WGSL → SPIR-V translation (vertex + fragment)
    │   └── ash FFI (Vulkan 1.3): swapchain + render pass + graphics pipeline
    │
    └── NoBackendStub           (lib.rs) — returns NoBackend for all ops
```

Opaque handle types (`RenderPipeline`, `GpuBuffer`, `Frame`) carry raw pointers
and backend-specific drop functions — no coupling between `lib.rs` and the
backend modules, exactly as in Borsalino.

### Ecosystem placement

```
                 ┌─────────────┐
                 │   Miriami   │  trans-graphical rendering framework (future)
                 └──────┬──────┘
                        │ GraphicsBackend (lowers onto)
              ┌─────────┴─────────┐
              │                   │
       ┌──────┴──────┐     ┌──────┴──────┐
       │  Baedeker   │     │ (native app)│
       │  (WASM host)│     └─────────────┘
       └──────┬──────┘
              │ adapter crate + host trait + WASM ABI
              │   (mirrors baedeker-borsalino / baedeker-gpu)
   ┌──────────┴──────────┐
   │                     │
┌──┴────────┐     ┌──────┴──────┐
│ Borsalino │     │  Goldenweek │
│ (compute) │     │  (graphics) │
└───────────┘     └─────────────┘
```

## License

Apache-2.0. Copyright (C) 2026 Industrial Algebra.

Contributors must sign the [CLA](https://github.com/Industrial-Algebra/.github/blob/main/CLA.md).
