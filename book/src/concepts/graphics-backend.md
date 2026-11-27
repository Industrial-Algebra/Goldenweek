# The GraphicsBackend Trait

Everything Goldenweek does flows through one trait:

```rust,ignore
pub trait GraphicsBackend {
    fn compile_render_pipeline(&self, ..., config: &PipelineConfig)
        -> Result<RenderPipeline>;
    fn create_buffer<T: Pod>(&self, data: &[T]) -> Result<GpuBuffer>;
    fn create_texture(&self, ...) -> Result<GpuTexture>;
    fn acquire_frame(&mut self) -> Result<Frame>;
    fn draw(&mut self, frame: &mut Frame, ...) -> Result<()>;
    fn present(&mut self, frame: Frame) -> Result<()>;
    // ... the full draw matrix and import seam below
}
```

## Opaque handles, not raw API types

`RenderPipeline`, `GpuBuffer`, `GpuTexture`, and `Frame` are opaque:
a `{ raw: *mut c_void, drop_fn }` pair that only the backend that minted
it can unpack. No `vk::Pipeline`, no `MTLVertexDescriptor`, no `ash`
type appears in the public surface. The consequence for correctness:

- **A handle minted by one backend is rejected by another** — the owner
  check is `Arc::ptr_eq` on the backend identity, before any Vulkan
  command.
- **Handles retire, not die.** Dropping a buffer that is still bound in
  a not-yet-submitted frame moves it to a *graveyard*; destruction
  happens at the next idle drain, after the fence confirms the GPU is
  done. Destroying a resource mid-flight is the classic validation-layer
  finding (VUID 00922 / 00765) — the retirement registry makes it
  structurally impossible.

## The surface is deliberately small

One trait, two backends, a handful of methods. The surface is
stockier than Borsalino's compute trait — presentation, render passes,
and pipeline state are irreducible for rendering — but every method
earns its place. When in doubt, Goldenweek refuses (see [Design
Refusals](../design/refusals.md)).
