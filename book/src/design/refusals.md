# Design Refusals

IA Design Principle 4: architectural refusal — what a library *will not
do* defines its scope as much as what it does. Goldenweek's v0.1
refusals, with rationale:

| Refusal | Rationale |
|---|---|
| **No windowing** — accept an external `SurfaceHandle` | Keep Goldenweek a pure device library; the caller (the app / Miriami) owns the window and its platform lifecycle. |
| **No `wgpu`** — raw FFI (Vulkan/`ash`, Metal/`objc`, `naga`) | Borsalino lineage: maximum auditability, no hidden translation layer. |
| **No scene graph / materials** | Those belong to Miriami, the framework layer above. Goldenweek renders what it is given. |
| **No async** | Synchronous, auditable frame loop at this layer. |
| **No device ownership** | The device is shared with Borsalino (Zunesha ADR 0001); interop is the point. |
| **No cross-queue import (v0.1)** | Compute-queue → graphics-queue visibility needs ownership transfer — M2, gated on Zunesha ADR 0005 Phase 2. |

Each refusal is reversible only through an ADR. The full argument is
[ADR 0001](./adr/0001-graphics-surface-and-scope.md).
