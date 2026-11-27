# Frames, Surfaces & the Present Cycle

A frame is the unit of presentation. The cycle is synchronous and
explicit:

```text
acquire_frame() ──▶ draw*() ──▶ present(frame)
       │                            │
   swapchain image            submits + presents,
   + render pass begins       then waits for the submit's fence
```

## `acquire_frame`

Begins a render pass on the next available swapchain image. Frame
counters rotate; a `Frame` can only be drawn into by the backend that
acquired it, and only one frame is live at a time (v0.1 is a single
in-flight frame).

## `draw` family

Recording only — draws append to the frame's command buffer. Queue
submission happens at present, under the caller-locks discipline:
every queue-mutating operation runs under an explicit lock acquired in
the same function, so recording never contends.

## `present`

One submission, one present, one fence wait — then resource retirement
drains (graveyarded buffers/pipelines whose fences have confirmed
completion get destroyed here).

## Surface loss

`acquire_frame` can report the surface is unavailable (resize,
minimise, display disconnection). The caller provides a fresh
`SurfaceHandle`; the backend exposes `recreate_swapchain` for the
format-stable case and refuses (with a full explanation) when the
surface format changed — pipelines are compiled against the format and
must be rebuilt.
