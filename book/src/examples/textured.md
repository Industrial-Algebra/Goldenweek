# Example: Textured

The M1 showcase — an RGBA8 texture uploaded through `create_texture`
and sampled on a quad:

```bash
cargo run --features vulkan --example textured
```

The example verifies itself by sampling the readback at four quadrant
positions — four distinct texels must land, proving the sampler and UV
coordinates actually carried through the pipeline.
