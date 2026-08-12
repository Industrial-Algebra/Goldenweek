// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! # Goldenweek — Thin GPU Graphics Abstraction
//!
//! > Render pipelines, frames, and presentation. Compute stays in
//! > [`borsalino`](https://crates.io/crates/borsalino).
//!
//! Goldenweek is the graphics sibling of Borsalino. Where Borsalino is a
//! minimal, synchronous **compute** abstraction over Metal and Vulkan,
//! Goldenweek is the same shape applied to **rendering**: write WGSL vertex
//! and fragment shaders, compile a render pipeline, draw into a frame, and
//! present it to a surface. No windowing, no scene graph, no materials —
//! those belong to higher layers (Baedeker's host, Miriami's framework).
//!
//! ## Design
//!
//! - **Surface-agnostic (P4 refusal):** Goldenweek never owns a window.
//!   The caller provides a [`SurfaceHandle`] (a Core Animation layer on
//!   macOS, a Vulkan `VkSurfaceKHR` elsewhere). This keeps Goldenweek a
//!   pure device library with no windowing-system dependency.
//! - **WGSL-first:** Shaders are authored in WGSL. The Metal backend will
//!   translate vertex/fragment to MSL via naga; the Vulkan backend to
//!   SPIR-V via naga. One shader source, two backends.
//! - **Synchronous frame loop:** `acquire → draw → present` blocks until
//!   the frame is displayed. No async runtime, matching Borsalino's
//!   dispatch model.
//! - **Opaque-handle backend isolation:** Render pipelines, buffers, and
//!   frames are opaque handles carrying a raw pointer and a backend drop
//!   function — no coupling between this module and the backend modules.
//!
//! ## Backends
//!
//! | Feature  | Platform       | Status (v0.1)         |
//! |----------|----------------|-----------------------|
//! | `metal`  | macOS          | 🚧 Trait + stub only  |
//! | `vulkan` | Linux, Windows | 🚧 Trait + stub only  |
//!
//! v0.1 ships the [`GraphicsBackend`] trait, the opaque handle types, and a
//! [`NoBackendStub`] that returns [`GraphicsError::NoBackend`] for every
//! operation. The Vulkan backend is the first target (mirroring Borsalino's
//! development order).
//!
//! ## Status
//!
//! Pre-release. The trait surface is stabilising; backends land in v0.2.

#![warn(missing_docs)]
#![warn(clippy::all)]

mod error;

pub use error::{GraphicsError, Result};

use std::ffi::c_void;

// ── Surface ownership (P4: Goldenweek never owns a window) ─────────

/// An externally-provided presentation surface.
///
/// Goldenweek refuses to own a window — the caller (a Baedeker host, the
/// Miriami framework, or a native application) creates the platform surface
/// and hands it in. This keeps Goldenweek free of any windowing-system
/// dependency (`winit`, `raw-window-handle` as a runtime, etc.).
///
/// Backends interpret the handle natively:
///
/// - **Metal:** a `CAMetalLayer` pointer.
/// - **Vulkan:** a `VkSurfaceKHR` plus the `VkInstance` that created it
///   (Goldenweek needs the instance to bind the surface to its device).
///
/// # Safety
///
/// The caller guarantees the pointed-to platform object is a valid,
/// graphics-presentable surface for the current device and that it outlives
/// every [`GraphicsBackend`] initialised from it.
#[non_exhaustive]
pub enum SurfaceHandle {
    /// macOS: a `CAMetalLayer` pointer.
    #[cfg(target_os = "macos")]
    MetalLayer(*mut c_void),
    /// Linux / Windows: a Vulkan `VkSurfaceKHR` and its owning `VkInstance`.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    VulkanSurface {
        /// The `VkInstance` that created the surface.
        instance: *mut c_void,
        /// The `VkSurfaceKHR` handle.
        surface: *mut c_void,
    },
}

// ── Pipeline configuration ────────────────────────────────────────

/// How vertices are assembled into primitives by the input assembler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Topology {
    /// Independent triangles (default).
    #[default]
    TriangleList,
    /// Connected triangle strip.
    TriangleStrip,
    /// Independent line segments.
    LineList,
    /// Connected line strip.
    LineStrip,
    /// One point per vertex.
    PointList,
}

/// Which face winding the rasterizer discards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    /// No culling — draw both faces (default).
    #[default]
    None,
    /// Discard front-facing primitives.
    Front,
    /// Discard back-facing primitives.
    Back,
}

/// Numeric format of a single vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    /// Two 32-bit floats — WGSL `vec2<f32>`.
    Float32x2,
    /// Three 32-bit floats — WGSL `vec3<f32>`.
    Float32x3,
    /// Four 32-bit floats — WGSL `vec4<f32>`.
    Float32x4,
}

impl VertexFormat {
    /// Size in bytes of one element of this format.
    #[must_use]
    pub const fn byte_size(self) -> u32 {
        match self {
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
        }
    }
}

/// Description of a single vertex attribute bound at draw time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    /// WGSL `@location(N)` the attribute feeds.
    pub location: u32,
    /// Byte offset of the attribute within one vertex.
    pub offset: u32,
    /// In-memory format of the attribute data.
    pub format: VertexFormat,
}

/// Layout of interleaved vertex data bound into a draw call.
///
/// Passed to [`GraphicsBackend::compile_render_pipeline`] so the backend
/// can bake the vertex-input description into the render pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VertexLayout {
    /// Number of bytes between consecutive vertices.
    pub stride: u32,
    /// Vertex attributes, in binding order.
    pub attributes: Vec<VertexAttribute>,
}

/// Graphics pipeline state compiled into a [`RenderPipeline`].
///
/// v0.1 deliberately refuses depth/stencil, blend, and multisample state
/// (the flat-triangle slice). Each is reintroduced only when a concrete
/// need justifies the surface-area growth.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PipelineConfig {
    /// Primitive assembly topology.
    pub topology: Topology,
    /// Face culling mode.
    pub cull_mode: CullMode,
    /// Vertex input layout.
    pub vertex_layout: VertexLayout,
}

// ── Opaque handle types ───────────────────────────────────────────

/// Handle to a compiled render pipeline.
///
/// Created by [`GraphicsBackend::compile_render_pipeline`] from a WGSL
/// vertex + fragment pair and a [`PipelineConfig`]. Wraps a backend-specific
/// pipeline object (Metal `MTLRenderPipelineState`, Vulkan `VkPipeline`).
///
/// # Drop behaviour
///
/// When dropped, releases its GPU resources via the backend-specific drop
/// function stored at construction time.
pub struct RenderPipeline {
    pub(crate) raw: *mut c_void,
    pub(crate) drop_fn: fn(*mut c_void),
}

impl std::fmt::Debug for RenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderPipeline")
            .field("raw", &self.raw)
            .finish()
    }
}

// Safety: the raw pointer is an opaque backend handle. Backends guarantee
// thread-safe access to compiled pipeline state (it is immutable after
// construction).
unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        (self.drop_fn)(self.raw);
    }
}

/// Handle to a GPU buffer of vertex / index / uniform data.
///
/// Created by [`GraphicsBackend::create_buffer`]. Wraps a backend-specific
/// buffer object (Metal `MTLBuffer`, Vulkan `VkBuffer`).
///
/// # Drop behaviour
///
/// When dropped, releases its GPU resources.
pub struct GpuBuffer {
    pub(crate) raw: *mut c_void,
    pub(crate) len: usize,
    pub(crate) drop_fn: fn(*mut c_void),
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("raw", &self.raw)
            .field("len", &self.len)
            .finish()
    }
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        (self.drop_fn)(self.raw);
    }
}

/// Handle to an acquired presentation frame.
///
/// Created by [`GraphicsBackend::acquire_frame`]. Record draw calls into it
/// via [`GraphicsBackend::draw`], then display it via
/// [`GraphicsBackend::present`], which consumes the frame.
///
/// A frame holds exclusive access to one swapchain image for the duration of
/// its lifetime. Dropping a frame without presenting it returns the image to
/// the swapchain (it is simply not displayed).
///
/// # Drop behaviour
///
/// [`GraphicsBackend::present`] consumes the frame by value; the backend
/// implementation nulls out the raw handle so the subsequent `Drop` is a
/// no-op. For an unpresented frame, `Drop` releases the image back to the
/// swapchain.
pub struct Frame {
    pub(crate) raw: *mut c_void,
    pub(crate) drop_fn: fn(*mut c_void),
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame").field("raw", &self.raw).finish()
    }
}

// Safety: a frame owns exclusive access to a swapchain image. While it is
// alive no other frame references the same image, and draw/present operate
// through the backend's command stream. Send is sound because the backend
// serialises command recording; Sync is intentionally NOT implemented — two
// threads must not record into the same frame concurrently.
unsafe impl Send for Frame {}

impl Drop for Frame {
    fn drop(&mut self) {
        (self.drop_fn)(self.raw);
    }
}

// ── Trait ─────────────────────────────────────────────────────────

/// Backend-agnostic GPU graphics interface.
///
/// Each backend (Metal on macOS, Vulkan elsewhere) implements this trait.
/// Callers obtain a backend via [`init_for_surface`] against an
/// externally-provided [`SurfaceHandle`].
///
/// # Frame lifecycle
///
/// ```text
/// acquire_frame() -> Frame
/// draw(&mut Frame, &RenderPipeline, &GpuBuffer, count)   // zero or more
/// present(Frame)                                          // consumes
/// ```
///
/// `acquire` blocks until a swapchain image is available; `draw` records a
/// draw call into the frame; `present` queues the frame for display and
/// blocks until it is safe to begin the next frame.
pub trait GraphicsBackend: Sized {
    /// Initialise the backend against an externally-provided surface.
    ///
    /// Creates (or binds to) a graphics-capable device and a swapchain
    /// targeting the surface. The caller guarantees the surface outlives
    /// the backend.
    fn init_for_surface(surface: SurfaceHandle) -> Result<Self>;

    /// Compile a WGSL vertex + fragment pair into a render pipeline.
    ///
    /// `vertex_entry` / `fragment_entry` name the `@vertex` / `@fragment`
    /// functions in their respective sources. The [`PipelineConfig`] bakes
    /// topology, culling, and vertex layout into the pipeline object.
    fn compile_render_pipeline(
        &self,
        vertex_entry: &str,
        vertex_source: &str,
        fragment_entry: &str,
        fragment_source: &str,
        config: &PipelineConfig,
    ) -> Result<RenderPipeline>;

    /// Allocate a GPU buffer and upload `data`.
    ///
    /// Used for vertex, index, or uniform storage. The buffer's
    /// interpretation at draw time is fixed by the pipeline's
    /// [`VertexLayout`].
    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer>;

    /// Acquire the next frame for rendering.
    ///
    /// Blocks until a swapchain image is available. Returns [`Frame`],
    /// which grants exclusive recording access until it is presented or
    /// dropped.
    fn acquire_frame(&self) -> Result<Frame>;

    /// Record a non-indexed draw call into `frame`.
    ///
    /// `vertices` is bound as vertex input (interpreted per the pipeline's
    /// [`VertexLayout`]); `vertex_count` is the number of vertices to draw.
    /// Call zero or more times per frame before [`present`](Self::present).
    fn draw(
        &self,
        frame: &mut Frame,
        pipeline: &RenderPipeline,
        vertices: &GpuBuffer,
        vertex_count: u32,
    ) -> Result<()>;

    /// Queue `frame` for display and block until it is safe to acquire the
    /// next frame.
    ///
    /// Consumes the frame.
    fn present(&self, frame: Frame) -> Result<()>;
}

// ── Stub backend (compile-time sentinel) ──────────────────────────

/// Stub backend — no render backend compiled for this target.
///
/// Exists so that a backend-less build still type-checks and every
/// [`GraphicsBackend`] method returns [`GraphicsError::NoBackend`]. Will be
/// replaced by cfg-gated `metal::MetalBackend` / `vulkan::VulkanBackend`
/// when the render backends land.
pub struct NoBackendStub;

impl GraphicsBackend for NoBackendStub {
    fn init_for_surface(_surface: SurfaceHandle) -> Result<Self> {
        Err(GraphicsError::NoBackend)
    }
    fn compile_render_pipeline(
        &self,
        _vertex_entry: &str,
        _vertex_source: &str,
        _fragment_entry: &str,
        _fragment_source: &str,
        _config: &PipelineConfig,
    ) -> Result<RenderPipeline> {
        Err(GraphicsError::NoBackend)
    }
    fn create_buffer<T: bytemuck::Pod>(&self, _data: &[T]) -> Result<GpuBuffer> {
        Err(GraphicsError::NoBackend)
    }
    fn acquire_frame(&self) -> Result<Frame> {
        Err(GraphicsError::NoBackend)
    }
    fn draw(
        &self,
        _frame: &mut Frame,
        _pipeline: &RenderPipeline,
        _vertices: &GpuBuffer,
        _vertex_count: u32,
    ) -> Result<()> {
        Err(GraphicsError::NoBackend)
    }
    fn present(&self, _frame: Frame) -> Result<()> {
        Err(GraphicsError::NoBackend)
    }
}

// ── Top-level initialiser ─────────────────────────────────────────

/// Initialise the best available render backend against `surface`.
///
/// v0.1: returns [`GraphicsError::NoBackend`] unconditionally — the render
/// backends land in v0.2. Once present, this is cfg-gated per platform
/// exactly like `borsalino::init`:
///
/// - macOS: `metal::MetalBackend` (requires `metal` feature)
/// - Linux / Windows: `vulkan::VulkanBackend` (requires `vulkan` feature`)
pub fn init_for_surface(_surface: SurfaceHandle) -> Result<NoBackendStub> {
    Err(GraphicsError::NoBackend)
}

/// Reusable WGSL shaders for Goldenweek examples and first backend slices.
///
/// Use with [`GraphicsBackend::compile_render_pipeline`]:
///
/// ```ignore
/// let pipeline = gpu.compile_render_pipeline(
///     "vs_main", goldenweek::kernels::FLAT_TRIANGLE_VERT,
///     "fs_main", goldenweek::kernels::FLAT_TRIANGLE_FRAG,
///     &config,
/// )?;
/// ```
pub mod kernels {
    /// Minimal `@vertex` shader: passthrough 2D positions from
    /// `@location(0)` to clip space.
    pub const FLAT_TRIANGLE_VERT: &str = r#"
@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}
"#;

    /// Minimal `@fragment` shader: solid magenta output.
    pub const FLAT_TRIANGLE_FRAG: &str = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_default_is_triangle_list() {
        assert_eq!(Topology::default(), Topology::TriangleList);
    }

    #[test]
    fn cull_mode_default_is_none() {
        assert_eq!(CullMode::default(), CullMode::None);
    }

    #[test]
    fn pipeline_config_default_is_minimal() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.topology, Topology::TriangleList);
        assert_eq!(cfg.cull_mode, CullMode::None);
        assert!(cfg.vertex_layout.attributes.is_empty());
        assert_eq!(cfg.vertex_layout.stride, 0);
    }

    #[test]
    fn vertex_format_byte_sizes() {
        assert_eq!(VertexFormat::Float32x2.byte_size(), 8);
        assert_eq!(VertexFormat::Float32x3.byte_size(), 12);
        assert_eq!(VertexFormat::Float32x4.byte_size(), 16);
    }

    #[test]
    fn stub_reports_no_backend_for_every_operation() {
        // The stub must be a total, never-panicking surface: every method
        // returns NoBackend, so a backend-less build is well-defined.
        let stub = NoBackendStub;
        let pipeline = stub.compile_render_pipeline(
            "vs_main",
            kernels::FLAT_TRIANGLE_VERT,
            "fs_main",
            kernels::FLAT_TRIANGLE_FRAG,
            &PipelineConfig::default(),
        );
        assert!(matches!(pipeline, Err(GraphicsError::NoBackend)));

        let buffer = stub.create_buffer(&[0.0f32; 3]);
        assert!(matches!(buffer, Err(GraphicsError::NoBackend)));
    }

    #[test]
    fn init_for_surface_refuses_unconditionally_in_v0_1() {
        // v0.1's init refuses regardless of the surface handle. Construct a
        // null surface of the current platform's variant and assert refusal;
        // this exercises the real free-function entry point.
        #[cfg(target_os = "macos")]
        let surface = SurfaceHandle::MetalLayer(std::ptr::null_mut());
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let surface = SurfaceHandle::VulkanSurface {
            instance: std::ptr::null_mut(),
            surface: std::ptr::null_mut(),
        };
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            // No SurfaceHandle variant exists for this target; nothing to test.
            return;
        }

        assert!(matches!(
            init_for_surface(surface),
            Err(GraphicsError::NoBackend)
        ));
    }
}
