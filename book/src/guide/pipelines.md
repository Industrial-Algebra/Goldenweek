# Render Pipelines & Vertex Bindings

A render pipeline compiles WGSL vertex + fragment stages plus the fixed
state (topology, culling, vertex input layout) into one opaque
`RenderPipeline`.

## `VertexBindingLayout`

Vertex input is a **list of bindings**; the binding index is the list
position. Each binding advances at a rate:

```rust,ignore
let config = PipelineConfig {
    vertex_bindings: vec![
        // binding 0 — per-vertex positions
        VertexBindingLayout {
            stride: 8,
            rate: VertexInputRate::Vertex,
            attributes: vec![VertexAttribute {
                location: 0, offset: 0, format: VertexFormat::Float32x2,
            }],
        },
        // binding 1 — per-instance transform (offset, rotation)
        VertexBindingLayout {
            stride: 24,
            rate: VertexInputRate::Instance,
            attributes: vec![
                VertexAttribute { location: 1, offset: 0,
                                  format: VertexFormat::Float32x2 },
                VertexAttribute { location: 2, offset: 8,
                                  format: VertexFormat::Float32x4 },
            ],
        },
    ],
    ..Default::default()
};
```

## Validation — compile-time, not runtime surprise

`compile_render_pipeline` rejects, with `InvalidArgument` **before any
GPU work**:

- **More than one Vertex-rate or more than one Instance-rate binding**
  (v0.1 draw arity; the list shape is the future-proofing, the draw API
  is 2×2).
- **Duplicate `@location` values across bindings** — locations are
  globally unique.
- **Attribute offsets+formats that exceed the binding stride.**
- **Shader/bindings location mismatch** — the vertex stage's reflected
  `@location` input set must **exactly equal** the bindings' location
  set, in both directions. This is the Input-07904 guard: a shader that
  declares `@location(0)` paired with empty bindings is a spec-invalid
  pipeline that happens to render on lenient drivers and breaks on
  strict ones — Goldenweek refuses it at compile time.

Struct-input WGSL (`fn vs_main(input: Input)` with `@location` on the
struct's members) is reflected exactly like direct argument locations,
nested structs included.
