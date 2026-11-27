# Introduction

**Goldenweek** is a thin GPU graphics abstraction for the Industrial
Algebra ecosystem. It compiles WGSL render pipelines, acquires frames,
draws into them, and presents them to a surface — over Vulkan today,
Metal later — behind one trait, `GraphicsBackend`.

It exists because the ecosystem needed a render surface with the same
discipline as its compute sibling: Borsalino runs bit-exact compute on
the shared Zunesha device, and Goldenweek renders with the results —
**zero-copy**, on the *same* device, with no round-trip through host
memory.

## What it is

- A **`GraphicsBackend` trait** with opaque handles — pipelines,
  buffers, textures, frames — no raw Vulkan/Metal types in the public
  surface.
- **WGSL-first**: shaders are authored in WGSL and translated via
  [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga); the render
  pipeline rejects shader/bindings mismatches before any GPU work.
- **Synchronous by default**, single graphics queue, explicit frames.
- **Zero-copy interop**: `import_zunesha_buffer` adopts a buffer created
  by Borsalino (or any Zunesha consumer) with no upload — device
  provenance validated at the substrate.

## What it is not

- No windowing — accept an external `SurfaceHandle`; never own a window.
- No scene graph, no materials, no retained mode, no async runtime.
- No `wgpu` — backends hand-roll FFI (ash / objc). This is a deliberate
  architectural refusal, not a gap (see [Design
  Refusals](./design/refusals.md)).
- Not a numerical-exactness layer — Goldenweek targets **structural
  correctness** (valid surfaces, pipelines, frames); bit-exactness is
  Borsalino's concern.

## Status

v0.1.0: the trait surface, the complete Vulkan backend, textures and
sampled images (M1), instanced and dynamic vertex batches (M3), and the
zero-copy import seam. Metal lands substrate-first after v0.1 —
Zunesha-Metal precedes Goldenweek-Metal.
