// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Vulkan render backend for Goldenweek.
//!
//! Constructed over a **borrowed** [`zunesha::vulkan::VulkanDevice`] (the shared
//! device substrate) and an externally-provided [`SurfaceHandle`].
//! Per Zunesha ADR 0001 the device is shared between Borsalino (compute) and
//! Goldenweek (graphics), so this backend does **not** own the device — it holds
//! cloned Vulkan handles (a non-owning view) and the caller must keep the
//! `VulkanDevice` alive for the backend's lifetime. This is a documented
//! invariant, not a borrow-checked one, matching Zunesha's own buffer-lifetime
//! posture (ADR 0003).
//!
//! This increment delivers construction only. The [`crate::GraphicsBackend`]
//! runtime operations (swapchain, render pipeline, frame loop) land in later
//! increments and are not yet implemented here.

use ash::vk::Handle;
use ash::{Entry, ext, khr, vk};

use crate::{GraphicsError, Result, SurfaceHandle};
use zunesha::Device;

// ── Headless surface (tests / headless rendering) ──────────────────

/// RAII headless presentation surface.
///
/// Creates a `VK_EXT_headless_surface` against the given instance and destroys
/// it on drop. Intended for tests and headless rendering (CI, servers). The
/// instance must have been created with `VK_EXT_headless_surface` enabled —
/// which Zunesha does automatically under
/// [`InitRequest::prefer_graphics`](zunesha::InitRequest::prefer_graphics).
///
/// # Safety contract (enforced by drop order)
///
/// The instance must outlive this surface. In tests, declare the owning
/// `VulkanDevice` *before* the `HeadlessSurface` so the instance is dropped
/// last.
pub struct HeadlessSurface {
    surface: vk::SurfaceKHR,
    /// Loaded `VK_KHR_surface` instance functions, used to destroy the surface.
    surface_fn: khr::surface::Instance,
}

impl HeadlessSurface {
    /// Create a headless surface against `instance`, loading the extension
    /// functions via `entry`.
    ///
    /// # Safety
    ///
    /// Vulkan FFI. `instance` must have enabled `VK_EXT_headless_surface`, and
    /// must outlive the returned surface.
    pub unsafe fn new(entry: &Entry, instance: &ash::Instance) -> Result<Self> {
        let headless_fn = ext::headless_surface::Instance::new(entry, instance);
        let create_info = vk::HeadlessSurfaceCreateInfoEXT::default();
        let surface = unsafe { headless_fn.create_headless_surface(&create_info, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateHeadlessSurfaceEXT: {e}")))?;
        let surface_fn = khr::surface::Instance::new(entry, instance);
        Ok(Self {
            surface,
            surface_fn,
        })
    }

    /// The raw `VkSurfaceKHR` handle.
    #[must_use]
    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }
}

impl Drop for HeadlessSurface {
    fn drop(&mut self) {
        // Safety: the surface owns its resource; the instance is kept alive by
        // the caller's VulkanDevice (the documented drop-order invariant).
        unsafe {
            self.surface_fn.destroy_surface(self.surface, None);
        }
    }
}

// ── Render backend ─────────────────────────────────────────────────

/// Vulkan render backend — a non-owning view over a borrowed
/// [`zunesha::vulkan::VulkanDevice`] plus a caller-owned presentation surface.
///
/// Holds cloned Vulkan handles (`instance`, `device`, `physical_device`) and the
/// queue capabilities copied from the device. The caller must keep the
/// `VulkanDevice` — and the surface — alive for the backend's lifetime. The
/// backend performs no Vulkan destruction on drop in this increment; the device
/// and surface are torn down by their owners.
///
/// (Step C: construction only. The `GraphicsBackend` runtime operations land in
/// later increments.)
pub struct VulkanBackend {
    // These handles are read by the swapchain, render-pass, and pipeline
    // operations landing in increments 3-5.
    #[allow(dead_code)]
    instance: ash::Instance,
    #[allow(dead_code)]
    device: ash::Device,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    queues: zunesha::Queues,
    surface: vk::SurfaceKHR,
}

impl VulkanBackend {
    /// Construct the backend over a borrowed Zunesha device and an external
    /// surface.
    ///
    /// `device` is borrowed (cloned handles are stored); it is not consumed, so
    /// the shared device remains available to Borsalino. The surface must have
    /// been created against the same instance that owns `device` — the caller's
    /// responsibility per the [`SurfaceHandle`] safety contract.
    pub fn new(device: &zunesha::vulkan::VulkanDevice, surface: SurfaceHandle) -> Result<Self> {
        // On Vulkan targets the only [`SurfaceHandle`] variant is
        // `VulkanSurface`, so this binding is irrefutable.
        let SurfaceHandle::VulkanSurface {
            surface: raw_surface,
            ..
        } = surface;
        let surface = vk::SurfaceKHR::from_raw(raw_surface as usize as u64);
        Ok(Self {
            instance: device.raw_instance(),
            device: device.raw_device(),
            physical_device: device.physical_device(),
            queues: device.queues(),
            surface,
        })
    }

    /// The graphics queue family index, if the device exposes a graphics queue.
    ///
    /// Goldenweek refuses to render where this is `None` (compute-only hardware).
    #[must_use]
    pub fn graphics_family(&self) -> Option<u32> {
        self.queues.graphics.map(|q| q.family_index)
    }

    /// The borrowed presentation surface handle.
    #[must_use]
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::ffi::c_void;
    use zunesha::{Device, InitRequest};

    /// The backend constructs over a real Zunesha device + a headless surface.
    ///
    /// This exercises the raw-handle accessors Increment 1 added to Zunesha
    /// (`raw_instance` / `raw_device` / `physical_device`) and confirms
    /// `VK_EXT_headless_surface` is available — `prefer_graphics` enabled it.
    ///
    /// Drop order (reverse declaration) is the Vulkan teardown contract:
    /// backend (no-op, non-owning) → headless (destroys surface while the
    /// instance is alive) → device (destroys the instance last).
    #[test]
    #[serial]
    fn backend_constructs_over_headless_surface() {
        let device = match zunesha::vulkan::VulkanDevice::init_with(InitRequest::prefer_graphics())
        {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e})");
                return;
            }
        };
        let entry = unsafe { Entry::load() }.expect("Entry::load");
        let instance = device.raw_instance();
        let headless =
            unsafe { HeadlessSurface::new(&entry, &instance) }.expect("headless surface");

        let surface = SurfaceHandle::VulkanSurface {
            // Goldenweek recovers the instance from the device, so the handle's
            // instance field is informational here.
            instance: std::ptr::null_mut(),
            surface: headless.handle().as_raw() as usize as *mut c_void,
        };
        let backend = VulkanBackend::new(&device, surface).expect("VulkanBackend::new");

        assert!(
            backend.graphics_family().is_some(),
            "graphics queue expected under prefer_graphics"
        );
        println!(
            "VulkanBackend constructed; graphics family = {:?}",
            backend.graphics_family()
        );
    }
}
