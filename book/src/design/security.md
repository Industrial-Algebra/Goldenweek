# Security Considerations

Goldenweek is a device library, not a network service — its security
surface is **memory safety at the GPU boundary** and **provenance of
foreign handles**.

## Opaque-handle provenance

Every public handle is `{ raw: *mut c_void, drop_fn }` — only the minting
backend can unpack it. Every consuming operation checks ownership first:

- draws validate pipeline + buffers belong to this backend (`Arc::ptr_eq`),
- import validates the buffer answers to the borrowed device — first by
  device identity, then by the substrate's `owns_buffer` (drop-function
  discriminant + logical-device comparison, zunesha 0.3).

A handle from a foreign backend or device is rejected **before any
Vulkan command** — the failure modes (mis-unboxing, cross-device binds)
are memory-unsafe by nature and are made unreachable rather than
handled after the fact.

## Lifetime under teardown

The retirement registry (bound / graveyard / leaked) plus the poison
flag govern teardown: resources in flight retire to a graveyard and are
destroyed after fence confirmation; a poisoned backend leaks rather
than destroys. The backend's own drop only destroys when
`device_wait_idle` confirms quiescence. The invariant: **no public
operation can destroy a resource the GPU may still be reading.**

## Trust boundaries

- The caller owns the window and the surface handle — Goldenweek never
  derives or creates one.
- The caller owns the device — Goldenweek borrows.
- Imported buffers carry substrate-validated provenance; rejection
  returns the buffer unconsumed.
- WGSL is compiled via naga at pipeline-compile time; pipeline
  validation rejects spec-invalid configurations before GPU work.

## Known limitations

- v0.1 import is same-graphics-queue only (cross-family visibility is
  M2 / ADR 0005 Phase 2).
- One in-flight frame; the present cycle is synchronous by design.
- The 00067 present-mode latency race is a known accepted baseline
  (validation-layer observable, not a Goldenweek defect).
