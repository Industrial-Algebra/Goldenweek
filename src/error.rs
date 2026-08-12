// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Error types for Goldenweek GPU graphics operations.
//!
//! All fallible operations return [`Result<T>`], an alias for
//! `std::result::Result<T, GraphicsError>`. Errors are structured with
//! context-rich variants using `thiserror`.
//!
//! Goldenweek's verification posture targets *structural* correctness
//! (valid surface, valid pipeline, valid frame lifecycle) — never numerical
//! exactness, which remains Borsalino's concern.

use thiserror::Error;

/// Errors that can occur in GPU graphics operations.
///
/// Each variant carries contextual information: what operation failed,
/// which pipeline or surface was involved, and the underlying platform
/// error message where available.
#[derive(Error, Debug)]
pub enum GraphicsError {
    /// No graphics backend is available for the current platform.
    ///
    /// Enable the `metal` feature on macOS or the `vulkan` feature
    /// on Linux/Windows once the render backends ship.
    #[error("no graphics backend available for current platform")]
    NoBackend,

    /// Failed to initialise the graphics device against the given surface.
    ///
    /// The surface handle could not be bound to a swapchain, or no
    /// graphics-capable device is present.
    #[error("failed to initialise graphics device: {0}")]
    InitFailed(String),

    /// The presentation surface is unavailable or has been lost.
    ///
    /// On resize, minimisation, or display disconnection the surface may
    /// become stale; the caller must provide a fresh [`crate::SurfaceHandle`].
    #[error("presentation surface unavailable: {message}")]
    SurfaceUnavailable {
        /// What went wrong with the surface.
        message: String,
    },

    /// Shader compilation failed.
    ///
    /// The WGSL vertex or fragment source could not be translated/compiled.
    /// The message contains the compiler error output.
    #[error("{stage} shader compilation failed for '{entry}': {message}")]
    CompileFailed {
        /// Which stage: `"vertex"` or `"fragment"`.
        stage: &'static str,
        /// The entry-point function name.
        entry: String,
        /// The compiler error message.
        message: String,
    },

    /// Render pipeline creation failed.
    ///
    /// The shaders compiled but the graphics pipeline object could not be
    /// assembled (invalid vertex layout, unsupported pipeline config, etc.).
    #[error("render pipeline creation failed: {message}")]
    PipelineFailed {
        /// The platform error message.
        message: String,
    },

    /// Frame acquisition failed.
    ///
    /// No swapchain image could be acquired within the operation's budget,
    /// or the swapchain is out of date and must be recreated.
    #[error("frame acquisition failed: {message}")]
    AcquireFailed {
        /// The platform error message.
        message: String,
    },

    /// Presentation failed.
    ///
    /// The recorded frame could not be queued for display.
    #[error("presentation failed: {message}")]
    PresentFailed {
        /// The platform error message.
        message: String,
    },

    /// Buffer creation failed.
    ///
    /// The GPU could not allocate a buffer of the requested type and size.
    #[error("buffer creation failed: {message}")]
    BufferCreationFailed {
        /// The platform error message.
        message: String,
    },

    /// Invalid draw call — mismatched vertex layout, null handle, or
    /// out-of-range vertex count.
    #[error("invalid draw call: {message}")]
    InvalidDraw {
        /// What was wrong with the draw.
        message: String,
    },

    /// Internal error — should not occur in normal operation.
    #[error("internal graphics error: {0}")]
    Internal(String),

    /// I/O error from the platform layer.
    #[error("platform I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for Goldenweek operations.
pub type Result<T> = std::result::Result<T, GraphicsError>;
