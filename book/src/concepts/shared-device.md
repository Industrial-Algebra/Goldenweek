# The Shared Device Model

Goldenweek never owns a GPU device. It **borrows** a
[`zunesha`](https://crates.io/crates/zunesha) device — the same device
Borsalino computes on:

```
      zunesha::vulkan::VulkanDevice   (one device, created by YOU)
        │                    │
   Borsalino            Goldenweek
   (compute)            (render)
   writes buffers       imports them — zero-copy
```

## Why borrowing matters

- **Zero-copy interop is not a bridge, it is the same memory.** A buffer
  Borsalino filled with compute output is bound as a Goldenweek vertex
  buffer through `import_zunesha_buffer` — no upload, no staging, no
  host round-trip.
- **One queue discipline.** The device's submission protocol (its
  epoch/quiescence accounting) governs both consumers; Goldenweek's
  queue operations join the same lock protocol rather than inventing a
  second one.
- **No hidden device management.** Device loss, memory strategy
  selection, and quiescence are Zunesha's contract, and are documented
  there (Zunesha ADR 0001).

## Provenance is validated, not asserted

`import_zunesha_buffer` refuses buffers that do not answer to the
borrowed device: the call takes the buffer's creating device, checks it
is the backend's own (identity), and asks the substrate whether the
buffer answers to that device (`VulkanDevice::owns_buffer`, zunesha
0.3). A foreign buffer is rejected **before any Vulkan command**, and
rejection is non-consuming — the error carries the buffer back.
