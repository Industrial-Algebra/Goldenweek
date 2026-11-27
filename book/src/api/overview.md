# API Reference

The public surface in one page — rustdocs are the authority
(`cargo doc --open`); this is the map.

## Types

| Type | Role |
|---|---|
| `GraphicsBackend` | the trait every backend implements |
| `VulkanBackend` | the Vulkan implementation (feature `vulkan`, Linux/Windows) |
| `NoBackendStub` | the null backend — every operation returns `NoBackend` |
| `RenderPipeline` | compiled pipeline (opaque handle) |
| `GpuBuffer` | vertex/instance data (opaque handle) |
| `Texture` | immutable sampled RGBA8 image (opaque handle) |
| `Frame` | exclusive recording grant until present |
| `SurfaceHandle` | caller-owned surface, raw platform handle inside |

## Configuration

- `PipelineConfig` — topology, cull mode, `vertex_bindings: Vec<VertexBindingLayout>`
- `VertexBindingLayout` — `{ stride, rate, attributes }`; rate is
  `VertexInputRate::{Vertex, Instance}`
- `VertexAttribute` — `{ location, offset, format }`
- `TextureDesc`, `FilterMode`

## Operations

- `init(&device, &surface)` — borrow both, own neither
- `compile_render_pipeline(vs_entry, vs_src, fs_entry, fs_src, &config)`
- `create_buffer(&data)` / `create_texture(&desc, &data, filter)`
- `import_zunesha_buffer(&device, buffer)` (feature `vulkan`)
- `acquire_frame()` / `draw` × 4 / `present(frame)`
- `recreate_swapchain(...)` — format-stable resize; refuses on format
  change with a full explanation

## Errors

`GraphicsError`: `NoBackend`, `InitFailed`, `SurfaceUnavailable`,
`RecreateRefused`, `InvalidArgument`, `ShaderCompilation`, ... and
`ImportRejected { message, buffer }` — the non-consuming import
rejection (vulkan feature only).
