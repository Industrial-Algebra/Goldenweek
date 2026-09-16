# ADR 0002 — Headless verification surfaces and device pinning

- **Date:** 2026-09-16
- **Status:** Accepted

## Context

Goldenweek's verification posture (per Zunesha ADR 0003) is *structural
graphics correctness*: frame lifecycle, pipeline validity, and — the
strongest form — that what was drawn actually reached the presentation
image. The strongest test therefore presents a real swapchain and reads
pixels back, which requires a Vulkan surface with no window: the
`VK_EXT_headless_surface` extension.

Driver support for that extension is not what its enumeration suggests.
The matrix below was established empirically during increments 3–5
(2026-08), on the development machines:

| Driver | Headless surface | Verdict |
|---|---|---|
| NVIDIA proprietary (RTX 5080 Laptop) | extension enumerated, surface caps / swapchain fail `ERROR_EXTENSION_NOT_PRESENT` | **unsupported** — the extension is a stub |
| Intel / Mesa (ARL integrated) | full acquire → draw → present → readback loop | **the test device** |
| llvmpipe (CPU rasterizer) | process crashes mid-probe | unusable — excluded |
| AMD / RADV (Phoenix1) | Zunesha device tests pass 22/22 on this hardware; Goldenweek's headless path not yet exercised there | untested for graphics |

Enumerating the extension is *not* proof it works: a WSI-less instance
happily returns a `vkCreateHeadlessSurfaceEXT` function pointer that then
fails the capability query. The surface must actually present.

Meanwhile the suite must also run — and skip gracefully — on machines with
no usable device at all (CI runners, compute-only hosts), and
`read_pixels` must assert pixel values without knowing the swapchain's
channel order (`B8G8R8A8` vs `R8G8B8A8`).

## Decision

1. **Headless graphics tests pin a driver via `GOLDENWEEK_TEST_DEVICE`** —
   an environment variable holding a case-insensitive device-name
   substring, forwarded to `zunesha::InitRequest::with_device_hint`
   (which wins over capability scoring). Default: `"Intel"`.
2. **Tests skip with a message, never fail, when no device or loader is
   present.** A host that cannot run the graphics suite is not a broken
   build. Hardware verification happens on the development machines
   before any release; CI runs the compile/clippy/doc gates plus
   device-free tests.
3. **Verification assertions are channel-order-symmetric.** The reference
   image is a flat magenta triangle (`R == B == 255`, `G == 0`) over a
   `0.1` gray clear (~26 per channel) — indistinguishable under either
   channel order, so tests do not depend on the chosen format.
4. **The NVIDIA proprietary headless gap is accepted, documented, and
   worked around — not fixed.** Windowed presentation on NVIDIA is
   unaffected; only the headless extension is stubbed. NVIDIA-only
   contributors run the default-feature suite; the vulkan-graphics tests
   run where a headless-capable driver exists.
5. **llvmpipe is excluded** from device enumeration for graphics tests
   (it crashes the process mid-probe, before a skip could fire).

## Consequences

**Positive**

- The verified-rendering claim (`draw_renders_triangle_verified_by_readback`)
  is deterministic and reproducible: same device, same image, same
  assertions, on every run.
- CI stays green on device-less runners by design, not by flaking.
- The empirical driver matrix — hard-won, easy to forget — is now a
  decision record that survives contributor turnover (carried as a
  recommendation by four consecutive research pulses).

**Negative**

- Graphics test coverage requires Mesa-class hardware on the host; an
  NVIDIA-only host cannot run the headless suite.
- The default `"Intel"` pin binds the suite to this fleet's hardware
  shape; other hosts override via the env var (or skip).
- llvmpipe exclusion means no pure-CPU fallback for graphics CI.

## Verification

The driver matrix was established by a device × query-matrix probe during
increment 3 (surface caps fail with `ERROR_EXTENSION_NOT_PRESENT` on the
stubbed path). The suite stands at 13 tests, green ×3 consecutive runs on
the Intel ARL headless path; `read_pixels` returns `255,0,255,255` inside
the triangle and `26,26,26,255` on the clear. Zunesha-level device tests
additionally pass on AMD/RADV (Phoenix1, 2026-09-02 research run).
