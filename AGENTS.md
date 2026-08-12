# Goldenweek — Agent Operating Guide

## Gitflow (Non-Negotiable)

Goldenweek follows IA gitflow as defined in the
[`ia-gitflow`](https://github.com/Industrial-Algebra/ia-toolkit/blob/main/skills/ia-gitflow/SKILL.md)
skill. Read it before touching branches.

> Note: the repo is currently bootstrapping on `main` (initial scaffold). Once
> the gitflow structure is initialized — `develop` branch, branch protection,
> CI — **all** feature work goes through `develop` via PRs. The rules below
> apply from that point on.

### Branch Model

```
feature/* ──PR──▶ develop ──release PR──▶ main ──tag v*──▶ publish
                     ▲                                        │
                     └──────── backmerge (merge commit) ──────┘
```

### Hard Rules

1. **Never push directly to `main` or `develop`.** Both are protected.
   No direct pushes — not "just a CI fix", not "a one-liner", not "it's faster".
   Branch it, PR it, let CI run. This is enforced by GitHub branch protection.

2. **Every release to `main` is followed by a `main → develop` backmerge**
   using a merge commit (never squash). This is the last step of releasing,
   not an optional chore.

3. **Release-only commits (version bump, changelog dating) live on a
   `release/*` branch**, not on `develop` or `main`.

### What went wrong elsewhere (do not repeat)

- **Direct pushes to main**: sibling projects bypassed review during CI
  emergencies. Branch protection now prevents this mechanically.
- **Silent `develop` recreation**: if `develop` is ever missing,
  **investigate why before recreating it** (check `delete_branch_on_merge`,
  recent deletions, etc.).
- **Skipped backmerges**: release PRs merged without backmerging `main`
  to `develop` cause the branches to diverge in history.

## Coding Standards

Follow the
[`ia-coding-standards`](https://github.com/Industrial-Algebra/ia-toolkit/blob/main/skills/ia-coding-standards/SKILL.md)
skill: TDD (test first), phantom types, `Result` not panic, exhaustive matching,
feature gates additive only, every public item documented.

## Project-Specific Conventions

- **Graphics sibling of Borsalino.** The architectural template is Borsalino
  (`../Borsalino`): one trait (`GraphicsBackend`), two backends (Metal/Vulkan),
  opaque-handle isolation, WGSL-first via naga. When in doubt, mirror Borsalino.
- **No windowing.** Accept an external `SurfaceHandle`; never own a window.
- **No `wgpu`.** Backends hand-roll FFI (objc / ash). This is a deliberate
  architectural refusal, not a gap — see `README.md` § Design refusals.
- **Structural correctness, not numerical exactness.** Verification effort
  targets valid surfaces/pipelines/frames. Numerical exactness is Borsalino's
  concern (and Miriami's design doc says so explicitly).
- **Device model (v0.1).** Goldenweek owns its own graphics+present device,
  independent of Borsalino's compute device. Compute→render interop is a
  documented future ADR — do not silently couple the two libraries without one.

## License

Apache-2.0. See `LICENSE` and `CONTRIBUTING.md`.
