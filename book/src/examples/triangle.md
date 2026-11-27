# Example: Triangle

The reference verification binary — the magenta-triangle readback proof
the test suite runs, promoted to a runnable example (ADR 0002 pinning
applies).

```bash
cargo run --features vulkan --example triangle
```

Compiles `FLAT_TRIANGLE_VERT` + `FLAT_TRIANGLE_FRAG` (the built-in WGSL
kernels), draws a triangle, presents, then **reads the pixels back**
and verifies magenta actually landed — the whole library exercised in
one visible pass: device borrow, surface, pipeline, buffer, frame
acquire/draw/present, and readback.
