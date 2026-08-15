// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Presentation-frame lifecycle for Goldenweek.
//!
//! A [`Frame`] grants exclusive access to one swapchain image between
//! acquisition and presentation. Backends store the acquired-image state in an
//! opaque [`FrameInner`] behind the handle.

use std::ffi::c_void;

/// Handle to an acquired presentation frame.
///
/// Created by a backend's frame acquisition (e.g.
/// `vulkan::VulkanBackend::acquire_frame`). Record draw calls into it, then
/// display it via the backend's present operation, which consumes the frame.
///
/// A frame holds exclusive access to one swapchain image for the duration of
/// its lifetime. Dropping a frame without presenting it returns the image to
/// the swapchain (it is simply not displayed).
///
/// # Drop behaviour
///
/// Present consumes the frame by value; the backend implementation nulls out
/// the raw handle so the subsequent `Drop` is a no-op. For an unpresented
/// frame, `Drop` releases the acquired-image state.
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

/// Backend-supplied state carried by an acquired [`Frame`].
#[cfg(feature = "vulkan")]
pub(crate) enum FrameInner {
    /// A swapchain image was acquired; `index` identifies it.
    Acquired {
        /// Swapchain image index.
        index: u32,
    },
}

#[cfg(feature = "vulkan")]
impl Frame {
    /// Wrap backend state into a frame handle.
    pub(crate) fn from_inner(inner: FrameInner) -> Self {
        Self {
            raw: Box::into_raw(Box::new(inner)) as *mut c_void,
            drop_fn: drop_frame_inner,
        }
    }

    /// Take the acquired image index, nulling the handle so the frame's
    /// subsequent `Drop` is a no-op.
    ///
    /// Returns `None` if the frame was already presented (handle null).
    pub(crate) fn take_index(&mut self) -> Option<u32> {
        if self.raw.is_null() {
            return None;
        }
        // Safety: `raw` was produced by `Box::into_raw(Box::new(FrameInner))`
        // in `from_inner` and has not been taken (non-null).
        let inner = unsafe { Box::from_raw(self.raw as *mut FrameInner) };
        self.raw = std::ptr::null_mut();
        match *inner {
            FrameInner::Acquired { index } => Some(index),
        }
    }
}

/// Drop function stored in [`Frame`] — drops the `Box<FrameInner>`.
#[cfg(feature = "vulkan")]
fn drop_frame_inner(raw: *mut c_void) {
    if !raw.is_null() {
        // Safety: `raw` was produced by `Box::into_raw` in `from_inner`.
        unsafe {
            drop(Box::from_raw(raw as *mut FrameInner));
        }
    }
}
