# ADR 0001 — Graphics surface and scope

- **Date:** 2026-08-12
- **Status:** Accepted

## Context

Goldenweek is the graphics counterpart of Borsalino. Two questions shape its
boundary:

1. **How big is the graphics surface, and why is it separate from compute?**
   Graphics is irreducibly larger than compute (swapchain, render pass, graphics
   pipeline, acquire→draw→present lifecycle). Hiding it behind Borsalino's
   compute trait would either bloat that trait or silently drop graphics
   capabilities. The surface must be first-class and distinct.

2. **Where does Goldenweek's responsibility end?** A graphics library can easily
   accrete a windowing layer, a scene graph, a material system, an async runtime,
   and a depth/blend/texture suite. Each of these is a real feature, and each is
   a distraction from Goldenweek's actual job: turning a device + surface into
   rendered frames, transparently.

Industrial Algebra Design Principle 4 (Architectural Refusal) says: name what
you will *not* do, so the scope is auditable.

## Decision

Goldenweek v0.1 carries a distinct `GraphicsBackend` trait and a documented
**refusal list** defining its scope. The refusals:

| # | Refusal | Rationale |
|---|---|---|
| 1 | **No windowing** — the caller supplies the `SurfaceHandle` | Goldenweek is a pure device library. Window creation, platform event loops, and DPI handling belong to the application (or, ultimately, to Miriami). Accepting an external surface keeps Goldenweek testable headlessly and free of platform-binding baggage. |
| 2 | **No `wgpu`** — raw FFI (Vulkan/`ash`, Metal/`objc`, `naga`) | The Borsalino lineage values auditability over convenience. A hidden translation layer (wgpu) obscures exactly the device behavior we want to reason about. |
| 3 | **No scene graph / materials** | Those are framework-layer concerns and belong to Miriami. Goldenweek renders the pipelines and buffers it is handed. |
| 4 | **No async** | A synchronous, auditable frame loop at this layer. Async can be layered above. |
| 5 | **No depth/blend/textures** (v0.1 only) | Ship clear + flat vertex rendering first. Depth, blending, and textures are explicit later increments, not v0.1 scope. |

## Consequences

**Positive**

- The v0.1 surface is small and fully auditable: a trait with six operations,
  opaque handles, no platform or framework baggage.
- Headless rendering is a first-class test mode (the external `SurfaceHandle` can
  be a `VK_EXT_headless_surface`), not an afterthought.
- The boundary with Miriami (framework) and Zunesha (device) is unambiguous.

**Negative**

- Callers must supply their own surface and (eventually) their own window/event
  loop. This is the intended trade, not a defect: it buys Goldenweek's purity as
  a device library.
- Deferred features (depth/blend/textures) mean v0.1 cannot render anything that
  needs them; this is accepted to keep the first backend small and verifiable.

## Relationship to the wider design

- The *device* Goldenweek renders against is shared — see
  [Zunesha ADR 0001](https://github.com/Industrial-Algebra/Zunesha/blob/main/docs/adr/0001-shared-device-substrate.md).
- The *verification* Goldenweek performs (structural graphics correctness) is
  bounded — see
  [Zunesha ADR 0003](https://github.com/Industrial-Algebra/Zunesha/blob/main/docs/adr/0003-cross-crate-proof-agreement.md).
