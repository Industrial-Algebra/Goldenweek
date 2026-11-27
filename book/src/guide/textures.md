# Textures & Sampling

M1 added immutable RGBA8 textures: `create_texture` validates, stages,
uploads, and builds the descriptor set in one synchronous call — the
texture is sampleable the moment it returns.

```rust,ignore
let texture = gpu.create_texture(
    &TextureDesc { width: 256, height: 256, ..Default::default() },
    &rgba_bytes,           // tightly packed RGBA8
    FilterMode::Linear,
)?;
```

The validation rules are fixed by `TextureDesc::validate`: tightly
packed RGBA8, non-zero dimensions, size = width × height × 4. The
sampler (linear or nearest) is created with the texture and bound
together — WGSL declares `texture_2d<f32>` + `sampler` and the split
descriptor set is wired for you.

Textured draws (`draw_textured`, `draw_instanced_textured`) bind the
texture's descriptor set at the pipeline's set slot. Pipeline state
(blend mode) is part of `PipelineConfig`.

Texture lifetime follows the same retirement protocol as buffers:
mid-flight drops retire to the graveyard; idle drains destroy them
after fence confirmation.
