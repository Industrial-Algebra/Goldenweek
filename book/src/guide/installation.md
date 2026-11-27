# Installation & Feature Flags

## Cargo

```toml
[dependencies]
goldenweek = { version = "0.1", features = ["vulkan"] }
zunesha = "0.3"
```

## Feature flags

| Feature | Effect | Platform |
|---|---|---|
| *(default: none)* | trait + opaque handles + `NoBackendStub` — every operation returns `GraphicsError::NoBackend` | all |
| `vulkan` | the complete Vulkan backend (via `ash` FFI over the shared Zunesha device) | Linux, Windows |
| `metal` | *(reserved)* Metal backend — lands post-v0.1, substrate-first (Zunesha-Metal precedes Goldenweek-Metal) | macOS |

Feature-combination hygiene is part of CI: every configuration
(default / vulkan / metal / vulkan+metal) builds, tests, and lints with
`-D warnings`.

## Zunesha version requirement

The zero-copy import seam (`import_zunesha_buffer`) requires
**zunesha ≥ 0.3** — the substrate-side `owns_buffer` provenance check.
The dependency window is `zunesha = "0.3"`.

## Platform notes

- The Vulkan backend borrows a `zunesha::vulkan::VulkanDevice`. Construct
  it with `InitRequest::prefer_graphics()`; Goldenweek joins the queue
  discipline of the device you hand it.
- GPU tests are serialised (`#[serial]`) because Vulkan loader init is
  not concurrency-safe; the same discipline applies to any process
  creating devices in parallel.
