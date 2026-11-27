# Drawing

The draw matrix is 2×2 — plain/instanced × untextured/textured:

| | untextured | textured |
|---|---|---|
| **per-vertex** | `draw(frame, pipeline, vertices, vertex_count)` | `draw_textured(..., texture)` |
| **per-instance** | `draw_instanced(..., vertices, instances, vertex_count, instance_count)` | `draw_instanced_textured(..., texture)` |

## Recording, not submitting

All four methods **record** into the frame's command buffer. Submission
happens at `present`. The draw methods are the only public surface that
touches the queue indirectly — and they never take the queue lock
themselves (caller-locks discipline).

## Ownership and lifetime

Every draw registers its **pipeline and all its buffers** in the
retention registry:

- A buffer or pipeline minted by **another backend** is rejected
  (owner check before any Vulkan command).
- Dropping a registered resource mid-frame **retires** it — the
  graveyard destroys it at the next idle drain, fence-confirmed. Public
  drops are safe at any point; destroying an in-flight resource is not
  a thing callers can do by accident.
- Draw **arity** is checked at record time: the pipeline's binding count
  must match the draw's arguments (`draw_instanced` needs the pipeline
  to have an Instance-rate binding), instance_count 0 is rejected.
