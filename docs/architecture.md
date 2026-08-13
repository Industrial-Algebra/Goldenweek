# Goldenweek Architecture

> The **graphics** backend for the Industrial Algebra GPU stack — the graphics
> counterpart of [Borsalino](https://github.com/Industrial-Algebra/Borsalino)
> (compute). Named for Marianne (Miss Goldenweek).

Goldenweek turns a GPU device into something you can render to: it owns the
swapchain, the render pass, the graphics pipeline, and the frame lifecycle
(acquire → draw → present). It does **not** own the device — that is
[Zunesha](https://github.com/Industrial-Algebra/Zunesha)'s job. Goldenweek is a
*consumer* of a `zunesha::Device`, much as Borsalino is.

```
   Miriami (trans-graphical framework, future)
        │  lowers geometric projection onto rendering
        ▼
   Goldenweek ──┐  swapchain · render pass · pipeline · frame loop
        │       │
        │  holds a zunesha::Device, renders on its graphics queue
        ▼       │
   Zunesha ◄────┘  device + queues + memory + buffers (the substrate)
```

## Why graphics is a bigger surface than compute

Graphics is *necessarily* stockier than compute (this was tension #1 of the
architectural assessment). A compute backend dispatches kernels; a graphics
backend must additionally:

- present to a surface (swapchain, acquire/present, present modes),
- encode a render pass (attachments, subpasses, load/store),
- build a graphics pipeline (vertex input, vertex + fragment shaders, viewport),
- and manage the acquire→draw→present frame lifecycle, which has real ordering
  constraints compute does not.

This cannot be hidden behind the compute abstraction, so Goldenweek carries its
own `GraphicsBackend` trait rather than reusing Borsalino's compute trait. See
[ADR 0001 — Graphics surface and scope](adr/0001-graphics-surface-and-scope.md).

## What Goldenweek refuses (v0.1 scope)

Goldenweek follows IA Design Principle 4 (Architectural Refusal). The v0.1
surface deliberately omits:

| Refusal | Rationale |
|---|---|
| **No windowing** — accept an external `SurfaceHandle` | Keep Goldenweek a pure device library; the caller (the app / Miriami) owns the window and its platform lifecycle. |
| **No `wgpu`** — raw FFI (Vulkan/`ash`, Metal/`objc`, `naga`) | Borsalino lineage: maximum auditability, no hidden translation layer. |
| **No scene graph / materials** | Those belong to Miriami, the framework layer above. Goldenweek renders what it is given. |
| **No async** | Synchronous, auditable frame loop at this layer. |
| **No depth/blend/textures** (v0.1) | Clear + flat vertex rendering first; these come in later increments. |

Full rationale in [ADR 0001](adr/0001-graphics-surface-and-scope.md).

## Relationship to Zunesha (the shared device)

Goldenweek does not open its own device. The decision to share a device
substrate is documented from Zunesha's side in
[Zunesha ADR 0001 — Shared device substrate](https://github.com/Industrial-Algebra/Zunesha/blob/main/docs/adr/0001-shared-device-substrate.md).

From Goldenweek's side, this means:

- Goldenweek holds a `zunesha::Device` and renders on its **graphics queue**
  (`queues().graphics`, present only when `has_graphics()`).
- A `GpuBuffer` wraps a `zunesha::Buffer`, so a buffer Borsalino filled via
  compute can be bound as a vertex buffer by Goldenweek with **zero copy**.
- Goldenweek refuses to initialise where the device has no graphics queue
  (`has_graphics() == false`) — the compute-only-hardware case (GB10 / DGX Spark).

> **Current state:** Goldenweek is not yet wired to depend on Zunesha. The
> scaffold defines its own `GpuBuffer` and an `init_for_surface` entry point over
> an independent device stub (`NoBackendStub`). The "Step C" reshape — making
> `GpuBuffer` wrap `zunesha::Buffer` and constructing the backend over a
> `zunesha::Device` — is the next integration step.

## Verification posture

Per
[Zunesha ADR 0003 — Cross-crate proof agreement](https://github.com/Industrial-Algebra/Zunesha/blob/main/docs/adr/0003-cross-crate-proof-agreement.md),
Goldenweek owns **structural graphics correctness** only:

- a render pipeline is valid before it is used,
- a frame is acquired before it is drawn into and drawn before it is presented,
- the frame lifecycle (acquire → draw → present, one owner at a time) is not
  violated.

It does **not** re-prove device or buffer safety (Zunesha's job) nor numerical
exactness (Borsalino's job).

## API surface

The [`GraphicsBackend`](../src/lib.rs) trait is the contract:

- **Construction** — `init_for_surface(surface: SurfaceHandle)`. The surface is
  caller-owned (see the no-windowing refusal).
- **Pipeline** — `compile_render_pipeline(PipelineConfig) -> RenderPipeline`.
- **Buffers** — `create_buffer(data) -> GpuBuffer` (wraps a `zunesha::Buffer` in
  the target architecture).
- **Frame loop** — `acquire_frame() -> Frame`, `draw(...)`, `present(frame)`.

Opaque handles (`RenderPipeline`, `GpuBuffer`, `Frame`) keep backend details out
of the public API.

## Current state & next steps

- ✅ Scaffold — `GraphicsBackend` trait, opaque handles, `PipelineConfig`,
  `SurfaceHandle`, `NoBackendStub`, full `#![warn(missing_docs)]`.
- ⏳ **Vulkan rendering backend** (next) — swapchain, render pass, graphics
  pipeline (vertex + fragment WGSL via `naga`), frame loop, all constructed over
  a `zunesha::Device`. Testable **headless** via `VK_EXT_headless_surface`
  (available on this machine), which matches the external-`SurfaceHandle`
  decision exactly.
- ⏳ **Step C** — wire Goldenweek to depend on Zunesha (`GpuBuffer` wraps
  `zunesha::Buffer`; backend constructed over a `zunesha::Device`).
- ⏳ Metal backend (raw `objc`) — mirroring Borsalino/Zunesha's backend split.
