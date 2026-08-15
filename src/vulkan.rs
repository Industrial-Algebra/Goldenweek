// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Vulkan render backend for Goldenweek.
//!
//! Constructed over a **borrowed** [`zunesha::vulkan::VulkanDevice`] (the shared
//! device substrate) and an externally-provided [`SurfaceHandle`]. Per Zunesha
//! ADR 0001 the device is shared between Borsalino (compute) and Goldenweek
//! (graphics), so this backend does **not** own the device — it holds cloned
//! Vulkan handles (a non-owning view) and the caller must keep the
//! `VulkanDevice` alive for the backend's lifetime. This is a documented
//! invariant, not a borrow-checked one, matching Zunesha's own buffer-lifetime
//! posture (ADR 0003).
//!
//! This increment delivers the swapchain and the frame lifecycle
//! ([`VulkanBackend::acquire_frame`] / [`VulkanBackend::present`]). Pipeline
//! compilation, buffers, and draw calls land in increments 4-5.

use ash::vk::Handle;
use ash::{Entry, ext, khr, vk};

use crate::frame::{Frame, FrameInner};
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

// ── Swapchain configuration ───────────────────────────────────────

/// Choose a surface format: prefer `B8G8R8A8_UNORM` (or SRGB variant), else
/// the first offered format.
fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::B8G8R8A8_SRGB)
        .copied()
        .unwrap_or(formats[0])
}

/// Choose the swap extent: the surface's current extent when it is defined,
/// otherwise a fixed headless default.
fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        // Undefined extent (headless surfaces typically report this): pick a
        // sane test default clamped to the surface's limits.
        let width = 256u32.clamp(
            caps.min_image_extent.width,
            caps.max_image_extent.width.max(1),
        );
        let height = 256u32.clamp(
            caps.min_image_extent.height,
            caps.max_image_extent.height.max(1),
        );
        vk::Extent2D { width, height }
    }
}

// ── Render backend ────────────────────────────────────────────────

/// Vulkan render backend — a non-owning view over a borrowed
/// [`zunesha::vulkan::VulkanDevice`] plus a caller-owned presentation surface.
///
/// Holds cloned Vulkan handles (`instance`, `device`, `physical_device`), the
/// graphics queue, and a swapchain built against the surface at construction.
/// The caller must keep the `VulkanDevice` — and the surface — alive for the
/// backend's lifetime. On drop the backend destroys its own swapchain and
/// synchronization objects only; the device and surface are torn down by their
/// owners.
pub struct VulkanBackend {
    // Read during construction; consumed again by the render-pass and
    // pipeline operations (increments 4-5) and by out-of-date-surface handling.
    #[allow(dead_code)]
    instance: ash::Instance,
    device: ash::Device,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    /// `VK_KHR_surface` instance functions (queries, surface destruction is
    /// the caller's).
    #[allow(dead_code)]
    surface_fn: khr::surface::Instance,
    /// `VK_KHR_swapchain` device functions.
    swapchain_fn: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    /// Number of images in the swapchain.
    image_count: u32,
    /// Chosen swap extent (needed by the render pass in increment 4).
    extent: vk::Extent2D,
    /// Graphics queue handle, reconstructed from the Zunesha queue view.
    graphics_queue: vk::Queue,
    /// Graphics queue family index (needed for command pools in increment 5).
    graphics_queue_family: u32,
    /// Signaled when a swapchain image is acquired.
    image_available: vk::Semaphore,
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // Safety: the backend owns exclusively the swapchain and semaphore.
        // Wait for outstanding work first (present may still be in flight).
        // The device/instance/surface belong to their owners and are NOT
        // destroyed here.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_semaphore(self.image_available, None);
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl VulkanBackend {
    /// Construct the backend over a borrowed Zunesha device and an external
    /// surface, building a swapchain against the surface.
    ///
    /// `device` is borrowed (cloned handles are stored); it is not consumed, so
    /// the shared device remains available to Borsalino. The surface must have
    /// been created against the same instance that owns `device` — the caller's
    /// responsibility per the [`SurfaceHandle`] safety contract.
    ///
    /// Refuses on compute-only hardware (no graphics queue) and when the
    /// graphics queue family cannot present to the surface.
    pub fn new(device: &zunesha::vulkan::VulkanDevice, surface: SurfaceHandle) -> Result<Self> {
        // On Vulkan targets the only [`SurfaceHandle`] variant is
        // `VulkanSurface`, so this binding is irrefutable.
        let SurfaceHandle::VulkanSurface {
            surface: raw_surface,
            ..
        } = surface;
        let surface = vk::SurfaceKHR::from_raw(raw_surface as usize as u64);

        let instance = device.raw_instance();
        let logical = device.raw_device();
        let physical_device = device.physical_device();
        let surface_fn = khr::surface::Instance::new(device.entry(), &instance);

        // Goldenweek refuses to initialise without a graphics queue
        // (compute-only hardware, e.g. GB10).
        let queues = device.queues();
        let graphics = queues.graphics.ok_or_else(|| {
            GraphicsError::InitFailed(
                "device exposes no graphics queue (compute-only hardware)".into(),
            )
        })?;
        let graphics_queue = vk::Queue::from_raw(graphics.raw as usize as u64);

        // The graphics family must support presentation to this surface.
        let supported = unsafe {
            surface_fn
                .get_physical_device_surface_support(
                    physical_device,
                    graphics.family_index,
                    surface,
                )
                .map_err(|e| GraphicsError::SurfaceUnavailable {
                    message: format!("vkGetPhysicalDeviceSurfaceSupportKHR: {e}"),
                })?
        };
        if !supported {
            return Err(GraphicsError::SurfaceUnavailable {
                message: "graphics queue family does not support presentation to this surface"
                    .into(),
            });
        }

        // Surface capabilities and format.
        let caps = unsafe {
            surface_fn
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(|e| GraphicsError::SurfaceUnavailable {
                    message: format!("vkGetPhysicalDeviceSurfaceCapabilitiesKHR: {e}"),
                })?
        };
        let formats = unsafe {
            surface_fn
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(|e| GraphicsError::SurfaceUnavailable {
                    message: format!("vkGetPhysicalDeviceSurfaceFormatsKHR: {e}"),
                })?
        };
        if formats.is_empty() {
            return Err(GraphicsError::SurfaceUnavailable {
                message: "surface offers no formats".into(),
            });
        }
        let format = choose_surface_format(&formats);
        let extent = choose_extent(&caps);

        // FIFO is universally supported and gives a well-defined blocking
        // present, matching the synchronous frame-loop contract.
        let min_count = caps.min_image_count.max(2);
        let image_count = if caps.max_image_count > 0 {
            min_count.min(caps.max_image_count)
        } else {
            min_count
        };

        let swapchain_fn = khr::swapchain::Device::new(&instance, &logical);
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(false)
            .old_swapchain(vk::SwapchainKHR::null());
        let swapchain = unsafe { swapchain_fn.create_swapchain(&create_info, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateSwapchainKHR: {e}")))?;
        let actual_count = unsafe { swapchain_fn.get_swapchain_images(swapchain) }
            .map_err(|e| GraphicsError::InitFailed(format!("swapchain images: {e}")))?
            .len() as u32;

        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { logical.create_semaphore(&semaphore_ci, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateSemaphore: {e}")))?;

        Ok(Self {
            instance,
            device: logical,
            physical_device,
            surface_fn,
            swapchain_fn,
            surface,
            swapchain,
            image_count: actual_count,
            extent,
            graphics_queue,
            graphics_queue_family: graphics.family_index,
            image_available,
        })
    }

    /// The graphics queue family index (always present — [`VulkanBackend::new`]
    /// refuses construction without one).
    #[must_use]
    pub fn graphics_family(&self) -> u32 {
        self.graphics_queue_family
    }

    /// The borrowed presentation surface handle.
    #[must_use]
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// The swap extent chosen at construction.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Number of swapchain images.
    #[must_use]
    pub fn image_count(&self) -> u32 {
        self.image_count
    }

    /// Acquire the next frame for rendering.
    ///
    /// Blocks until a swapchain image is available. The returned [`Frame`]
    /// grants exclusive recording access to that image until it is presented
    /// or dropped. Dropping an unpresented frame is well-defined: the image is
    /// simply not displayed.
    pub fn acquire_frame(&self) -> Result<Frame> {
        let (index, _suboptimal) = unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        }
        .map_err(|e| GraphicsError::AcquireFailed {
            message: format!("vkAcquireNextImageKHR: {e}"),
        })?;
        Ok(Frame::from_inner(FrameInner::Acquired { index }))
    }

    /// Queue `frame` for display and block until it is safe to acquire the
    /// next frame.
    ///
    /// Consumes the frame. In this increment no rendering has been recorded, so
    /// the image presents its previous (initial) contents — correct behaviour
    /// for the frame-lifecycle slice.
    pub fn present(&self, mut frame: Frame) -> Result<()> {
        let index = frame
            .take_index()
            .ok_or_else(|| GraphicsError::InvalidDraw {
                message: "frame has no acquired image (already presented?)".into(),
            })?;
        let swapchains = [self.swapchain];
        let indices = [index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices);
        let result = unsafe {
            self.swapchain_fn
                .queue_present(self.graphics_queue, &present_info)
        };
        // Suboptimal is acceptable for the headless/synchronous v0.1 loop.
        result
            .map(|_| ())
            .map_err(|e| GraphicsError::PresentFailed {
                message: format!("vkQueuePresentKHR: {e}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::ffi::c_void;
    use zunesha::InitRequest;

    /// Build a backend over a headless surface (shared by the increment tests).
    ///
    /// Drop order (reverse declaration) is the Vulkan teardown contract:
    /// backend (destroys swapchain + semaphore) → headless (destroys surface
    /// while the instance is alive) → device (destroys the instance last).
    struct TestRig {
        backend: VulkanBackend,
        _headless: HeadlessSurface,
        _device: zunesha::vulkan::VulkanDevice,
        _entry: Entry,
    }

    fn rig() -> Option<TestRig> {
        // NVIDIA's proprietary driver does not support headless-surface
        // swapchains (surface caps fail with ERROR_EXTENSION_NOT_PRESENT), so
        // headless tests pin to a driver that does — Intel (Mesa) by default,
        // overridable via GOLDENWEEK_TEST_DEVICE for other hosts.
        let hint = std::env::var("GOLDENWEEK_TEST_DEVICE").unwrap_or_else(|_| "Intel".into());
        let request = InitRequest::prefer_graphics().with_device_hint(hint);
        let device = match zunesha::vulkan::VulkanDevice::init_with(request) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e})");
                return None;
            }
        };
        let entry = unsafe { Entry::load() }.expect("Entry::load");
        let instance = device.raw_instance();
        let headless =
            unsafe { HeadlessSurface::new(&entry, &instance) }.expect("headless surface");
        let surface = SurfaceHandle::VulkanSurface {
            instance: std::ptr::null_mut(),
            surface: headless.handle().as_raw() as usize as *mut c_void,
        };
        let backend = VulkanBackend::new(&device, surface).expect("VulkanBackend::new");
        Some(TestRig {
            backend,
            _headless: headless,
            _device: device,
            _entry: entry,
        })
    }

    /// The backend constructs, builds a swapchain, and exposes a sane extent.
    #[test]
    #[serial]
    fn backend_constructs_with_swapchain() {
        let Some(rig) = rig() else { return };
        assert!(
            rig.backend.image_count() >= 2,
            "FIFO swapchain needs >= 2 images"
        );
        assert!(rig.backend.extent().width > 0 && rig.backend.extent().height > 0);
        println!(
            "swapchain: {} images, extent = {}x{}",
            rig.backend.image_count(),
            rig.backend.extent().width,
            rig.backend.extent().height
        );
    }

    /// The frame lifecycle: acquire → present, repeatedly.
    #[test]
    #[serial]
    fn acquire_present_roundtrip() {
        let Some(rig) = rig() else { return };
        for i in 0..3 {
            let frame = rig.backend.acquire_frame().expect("acquire_frame");
            rig.backend.present(frame).expect("present");
            println!("frame {i} acquired + presented");
        }
    }

    /// Dropping an unpresented frame is well-defined: the next acquire works.
    #[test]
    #[serial]
    fn dropping_unpresented_frame_is_safe() {
        let Some(rig) = rig() else { return };
        let frame = rig.backend.acquire_frame().expect("acquire_frame");
        drop(frame); // not presented
        let frame = rig.backend.acquire_frame().expect("acquire after drop");
        rig.backend.present(frame).expect("present after drop");
    }
}
