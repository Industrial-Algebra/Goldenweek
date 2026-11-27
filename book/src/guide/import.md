# Zero-Copy Buffer Import

The seam that makes compute→render interop free:

```rust,ignore
// Borsalino (or any Zunesha consumer) writes a buffer on the shared device:
let buffer = device.create_buffer(&instance_data)?;

// Goldenweek adopts it — no upload, no staging, no queue operation:
let instances = gpu.import_zunesha_buffer(&device, buffer)?;
gpu.draw_instanced(&mut frame, &pipeline, &vertices, &instances, 4, 64)?;
```

## The provenance protocol

`zunesha::Buffer` is opaque — a raw handle with no device identity on
its face. Import validates provenance in two steps, **before any Vulkan
command**:

1. **Device identity** — the `device` you pass must be the backend's own
   borrowed device.
2. **Substrate ownership** — `VulkanDevice::owns_buffer` (zunesha 0.3)
   checks the buffer actually answers to that device.

Rejection is **non-consuming**: `GraphicsError::ImportRejected` carries
your buffer back — no adoption happened, the caller retains it (the
`String::from_utf8` pattern). The recovery tests round-trip a rejected
buffer back into the device that owns it.

## The v0.1 tier: same graphics queue

Import is same-graphics-queue: the buffer must be visible to
subsequently recorded draws. Zunesha's queue FIFO makes this hold for
same-queue writes. **Cross-family** visibility (a compute-queue write
consumed by a graphics-queue draw) needs ownership transfer — that is
M2, gated on Zunesha ADR 0005 Phase 2 (cross-queue sync), deliberately
out of scope for v0.1.

## After import

The imported `GpuBuffer` is a first-class citizen: draws bind it, drops
retire it through the same registry, and `import_zunesha_buffer` itself
never touches the queue (the adoption is pure bookkeeping).
