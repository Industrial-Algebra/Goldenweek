# Example: Instanced

The M3 flagship — one draw call, many instances, per-instance data:

```bash
cargo run --features vulkan --example instanced
```

A two-binding pipeline (per-vertex geometry at binding 0, per-instance
offsets at binding 1), drawn with `draw_instanced`, and verified by
readback: each instance's quad must land at its own position. The
zero-copy seam variant (`import_zunesha_buffer`) is shown in the guide
— [Zero-Copy Buffer Import](../guide/import.md).
