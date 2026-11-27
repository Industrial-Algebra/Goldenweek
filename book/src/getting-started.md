# Getting Started

## Prerequisites

- Rust 1.85+ (edition 2024).
- For the Vulkan backend: a Vulkan loader and a Vulkan-capable device.
  On NixOS the IA development shell pins one; on typical Linux
  distributions the loader is already present.
- The `vulkan` feature (Linux/Windows). Metal is macOS-only and lands
  after v0.1, substrate-first.

## Add the dependency

```toml
[dependencies]
goldenweek = { version = "0.1", features = ["vulkan"] }
zunesha = "0.3" # the shared device — you construct it, Goldenweek borrows it
```

## Your first frame

The caller owns two things Goldenweek never touches: the **window**
(through a `SurfaceHandle`) and the **device** (a `zunesha` device,
shared with Borsalino's compute).

```rust,ignore
use goldenweek::{init, PipelineConfig};

// 1. The shared device — graphics-preferred queue family.
let device = zunesha::vulkan::VulkanDevice::init_with(
    zunesha::InitRequest::prefer_graphics(),
)?;

// 2. A surface handle for the window you own.
let surface = goldenweek::SurfaceHandle::from_raw(surface_ptr)?;

// 3. Goldenweek borrows both — it owns neither.
let gpu = goldenweek::init(&device, &surface)?;

// 4. Compile a pipeline from WGSL.
let pipeline = gpu.compile_render_pipeline(
    "vs_main", goldenweek::kernels::FLAT_TRIANGLE_VERT,
    "fs_main", goldenweek::kernels::FLAT_TRIANGLE_FRAG,
    &PipelineConfig::default(),
)?;

// 5. Vertex data.
let vertices = gpu.create_buffer(&[
    0.0f32, 0.5,    // top
   -0.5f32, -0.5,   // bottom-left
    0.5f32, -0.5,   // bottom-right
])?;

// 6. Acquire → draw → present.
let mut frame = gpu.acquire_frame()?;
gpu.draw(&mut frame, &pipeline, &vertices, 3)?;
gpu.present(frame)?;
```

## Run the examples

```bash
cargo run --features vulkan --example triangle   # magenta triangle, read back and verified
cargo run --features vulkan --example textured   # textured quad, quadrant-sampled
cargo run --features vulkan --example instanced  # one draw call, many instances
```

## Where to go next

- [Render Pipelines & Vertex Bindings](./guide/pipelines.md) — the
  `VertexBindingLayout` model and its validation rules.
- [Drawing](./guide/drawing.md) — the 2×2 draw matrix, textured and
  instanced forms.
- [Zero-Copy Buffer Import](./guide/import.md) — adopt a
  Borsalino-written buffer with no upload.
