// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Vulkan render backend for Goldenweek.
//!
//! Constructed over a **borrowed** [`zunesha::vulkan::VulkanDevice`] (the shared
//! device substrate) and an externally-provided [`SurfaceHandle`]. Per Zunesha
//! ADR 0001 the device is shared between Borsalino (compute) and Goldenweek
//! (graphics), so this backend does **not** own the device — it borrows it
//! (`VulkanBackend<'a>`), and the borrow checker enforces what was previously a
//! documented invariant: the device outlives the backend.
//!
//! The full frame lifecycle is implemented: `acquire_frame` begins a command
//! buffer and the render pass (clearing to the backend's clear color),
//! [`VulkanBackend::draw`] records pipeline + vertex-buffer binds, and
//! `present` ends the pass, submits (waiting the acquire semaphore, signalling
//! the present semaphore + a fence), presents, and blocks on the fence so the
//! next acquire is safe — the synchronous frame loop the trait documents.
//!
//! Buffers created here are `zunesha::Buffer`s (zero-copy compute→render
//! interop, ADR 0001); `zunesha::vulkan::VulkanDevice::raw_buffer` recovers
//! the `VkBuffer` for binding.

use std::ffi::{CString, c_void};
use std::sync::Mutex;

use ash::vk::Handle;
use ash::{Entry, ext, khr, vk};

use crate::frame::{Frame, FrameInner};
use crate::{
    CullMode, GpuBuffer, GraphicsError, PipelineConfig, RenderPipeline, Result, SurfaceHandle,
    Topology, VertexFormat,
};
use zunesha::Device;

// ── Shader compilation (WGSL → SPIR-V via naga) ───────────────────

/// Human stage name for error reporting.
fn stage_name(stage: naga::ShaderStage) -> &'static str {
    match stage {
        naga::ShaderStage::Vertex => "vertex",
        naga::ShaderStage::Fragment => "fragment",
        _ => "compute",
    }
}

/// Compile one WGSL shader stage to SPIR-V words.
///
/// Parses, validates, and lowers via naga. The entry point is selected in the
/// backend pass ([`naga::back::spv::PipelineOptions`]), so a missing entry
/// name is reported here rather than by the driver.
fn compile_stage(entry: &str, source: &str, stage: naga::ShaderStage) -> Result<Vec<u32>> {
    let mut frontend = naga::front::wgsl::Frontend::new();
    let module = frontend
        .parse(source)
        .map_err(|e| GraphicsError::CompileFailed {
            stage: stage_name(stage),
            entry: entry.into(),
            message: e.emit_to_string(source),
        })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| GraphicsError::CompileFailed {
        stage: stage_name(stage),
        entry: entry.into(),
        message: e.emit_to_string(source),
    })?;
    let options = naga::back::spv::Options::default();
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry.to_string(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options)).map_err(|e| {
        GraphicsError::CompileFailed {
            stage: stage_name(stage),
            entry: entry.into(),
            message: format!("SPIR-V lowering: {e:?}"),
        }
    })
}

// ── Pipeline-config translation ───────────────────────────────────

/// Map [`Topology`] to Vulkan primitive topology.
fn to_vk_topology(t: Topology) -> vk::PrimitiveTopology {
    match t {
        Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        Topology::LineList => vk::PrimitiveTopology::LINE_LIST,
        Topology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        Topology::PointList => vk::PrimitiveTopology::POINT_LIST,
    }
}

/// Map [`CullMode`] to Vulkan cull-mode flags.
fn to_vk_cull(c: CullMode) -> vk::CullModeFlags {
    match c {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

/// Map [`VertexFormat`] to Vulkan formats.
fn to_vk_format(f: VertexFormat) -> vk::Format {
    match f {
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
    }
}

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
/// `VulkanDevice` *after* the `HeadlessSurface` so the instance is dropped
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

/// Per-frame command-recording state (interior mutability: every trait
/// operation takes `&self`, but the frame loop is synchronous by design — the
/// mutex is uncontended).
struct RenderState {
    /// The frame command buffer (reset + rerecorded each acquire).
    cmd: vk::CommandBuffer,
    /// Signalled when the frame's command buffer finishes.
    fence: vk::Fence,
    /// Where the current frame's recording stands.
    phase: RenderPhase,
}

/// The frame-loop state machine (exhaustive — every transition is explicit).
#[derive(Debug)]
enum RenderPhase {
    /// Image `u32` acquired; render pass open, commands recording.
    Open(u32),
    /// Recording submitted and fence-waited (a `read_pixels` flush);
    /// contents stable, not yet presented.
    Flushed(u32),
    /// No frame in flight.
    Idle,
}

/// Internal state for a compiled Vulkan pipeline, stored behind the opaque
/// [`RenderPipeline`] handle. Self-contained (clones the device) so it can
/// destroy itself on drop.
struct VulkanPipelineInner {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    vs_module: vk::ShaderModule,
    fs_module: vk::ShaderModule,
    device: ash::Device,
}

/// Drop function stored in [`RenderPipeline`] — drops the boxed inner and its
/// Vulkan objects.
pub(super) fn drop_vulkan_pipeline(raw: *mut c_void) {
    if !raw.is_null() {
        // Safety: `raw` was produced by `Box::into_raw` in
        // `compile_render_pipeline`.
        unsafe {
            let inner = Box::from_raw(raw as *mut VulkanPipelineInner);
            inner.device.destroy_pipeline(inner.pipeline, None);
            inner.device.destroy_pipeline_layout(inner.layout, None);
            inner.device.destroy_shader_module(inner.vs_module, None);
            inner.device.destroy_shader_module(inner.fs_module, None);
        }
    }
}

/// Drop function stored in [`GpuBuffer`] — drops the boxed `zunesha::Buffer`,
/// whose own drop destroys the Vulkan buffer + memory.
pub(super) fn drop_zunesha_buffer_box(raw: *mut c_void) {
    if !raw.is_null() {
        // Safety: `raw` was produced by `Box::into_raw` in `create_buffer`.
        unsafe {
            drop(Box::from_raw(raw as *mut zunesha::Buffer));
        }
    }
}

// ── Render backend ────────────────────────────────────────────────

/// Vulkan render backend — a borrowed view over a
/// [`zunesha::vulkan::VulkanDevice`] plus a caller-owned presentation surface.
///
/// The backend borrows the Zunesha device (`'a` lifetime, borrow-checked: the
/// device outlives the backend) and holds cloned Vulkan handles, the graphics
/// queue, a swapchain with its render pass / framebuffers, and the per-frame
/// command-recording state. The caller must keep the surface alive for the
/// backend's lifetime. On drop the backend destroys only what it created
/// (swapchain, views, render pass, framebuffers, pool, semaphores, fence); the
/// device and surface are torn down by their owners.
///
/// Buffers created via [`VulkanBackend::create_buffer`] are `zunesha::Buffer`s
/// — a buffer Borsalino filled by compute can be bound as a vertex buffer with
/// zero copies (ADR 0001). Pipelines own themselves; both handles must be
/// dropped before the backend (enforced for the device by the borrow, and for
/// pipelines by the documented invariant).
pub struct VulkanBackend<'a> {
    zunesha_device: &'a zunesha::vulkan::VulkanDevice,
    device: ash::Device,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    /// `VK_KHR_swapchain` device functions.
    swapchain_fn: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    /// The swapchain images (indexed by acquire index).
    images: Vec<vk::Image>,
    /// Chosen swap extent.
    extent: vk::Extent2D,
    /// Chosen swapchain image format (render-pass attachment format).
    format: vk::Format,
    /// One image view per swapchain image.
    image_views: Vec<vk::ImageView>,
    /// Render pass: single color attachment, clear→store, UNDEFINED→PRESENT.
    render_pass: vk::RenderPass,
    /// One framebuffer per swapchain image, sized to `extent`.
    framebuffers: Vec<vk::Framebuffer>,
    /// Graphics command pool (on the graphics family).
    command_pool: vk::CommandPool,
    /// Per-frame recording state (command buffer, fence, open-pass index).
    render: Mutex<RenderState>,
    /// Graphics queue handle, reconstructed from the Zunesha queue view.
    graphics_queue: vk::Queue,
    /// Graphics queue family index.
    graphics_queue_family: u32,
    /// Signalled when a swapchain image is acquired.
    image_available: vk::Semaphore,
    /// Signalled when the frame's rendering is complete (present waits on it).
    render_finished: vk::Semaphore,
}

impl Drop for VulkanBackend<'_> {
    fn drop(&mut self) {
        // Safety: the backend owns exclusively everything destroyed here.
        // Wait for outstanding work first (present may still be in flight).
        // Pipelines/buffers self-destruct and must already have been dropped
        // by the caller (documented invariant). The device/instance/surface
        // belong to their owners and are NOT destroyed here.
        unsafe {
            let _ = self.device.device_wait_idle();
            let render = self.render.lock().expect("render state lock");
            self.device.destroy_fence(render.fence, None);
            self.device
                .free_command_buffers(self.command_pool, &[render.cmd]);
            drop(render);
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_semaphore(self.render_finished, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
    }
}

impl<'a> VulkanBackend<'a> {
    /// Construct the backend over a borrowed Zunesha device and an external
    /// surface, building the swapchain, render pass, and framebuffers against
    /// the surface.
    ///
    /// The device is borrowed, not consumed — the shared device remains
    /// available to Borsalino. The surface must have been created against the
    /// same instance that owns the device (the caller's responsibility per the
    /// [`SurfaceHandle`] safety contract).
    ///
    /// Refuses on compute-only hardware (no graphics queue) and when the
    /// graphics queue family cannot present to the surface.
    pub fn new(
        device: &'a zunesha::vulkan::VulkanDevice,
        surface: SurfaceHandle,
    ) -> Result<VulkanBackend<'a>> {
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

        // Image views + render pass + framebuffers — the swapchain's rendering
        // surface. One framebuffer per swapchain image, all sized to `extent`.
        let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }
            .map_err(|e| GraphicsError::InitFailed(format!("swapchain images: {e}")))?;
        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let ci = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(subresource);
                unsafe { logical.create_image_view(&ci, None) }
                    .map_err(|e| GraphicsError::InitFailed(format!("vkCreateImageView: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let attachment = vk::AttachmentDescription::default()
            .format(format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref);
        let rp_ci = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass));
        let render_pass = unsafe { logical.create_render_pass(&rp_ci, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateRenderPass: {e}")))?;

        let framebuffers: Vec<vk::Framebuffer> = image_views
            .iter()
            .map(|&view| {
                let ci = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(std::slice::from_ref(&view))
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { logical.create_framebuffer(&ci, None) }
                    .map_err(|e| GraphicsError::InitFailed(format!("vkCreateFramebuffer: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        // Graphics command pool + frame command buffer + fence (created
        // signalled so the first acquire's wait passes immediately).
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics.family_index);
        let command_pool = unsafe { logical.create_command_pool(&pool_ci, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateCommandPool: {e}")))?;
        let cb_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { logical.allocate_command_buffers(&cb_ci) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkAllocateCommandBuffers: {e}")))?
            .first()
            .copied()
            .ok_or_else(|| GraphicsError::InitFailed("no command buffer allocated".into()))?;
        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { logical.create_fence(&fence_ci, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateFence: {e}")))?;

        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { logical.create_semaphore(&semaphore_ci, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateSemaphore: {e}")))?;
        let render_finished = unsafe { logical.create_semaphore(&semaphore_ci, None) }
            .map_err(|e| GraphicsError::InitFailed(format!("vkCreateSemaphore: {e}")))?;

        Ok(Self {
            zunesha_device: device,
            device: logical,
            physical_device,
            swapchain_fn,
            surface,
            swapchain,
            images,
            extent,
            format: format.format,
            image_views,
            render_pass,
            framebuffers,
            command_pool,
            render: Mutex::new(RenderState {
                cmd,
                fence,
                phase: RenderPhase::Idle,
            }),
            graphics_queue,
            graphics_queue_family: graphics.family_index,
            image_available,
            render_finished,
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
        self.images.len() as u32
    }

    /// The swapchain image format — needed to decode [`VulkanBackend::read_pixels`]
    /// bytes (channel order follows this format).
    #[must_use]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Compile a WGSL vertex + fragment pair into a render pipeline.
    ///
    /// Mirrors [`crate::GraphicsBackend::compile_render_pipeline`]: shaders are
    /// translated WGSL → SPIR-V via naga, and `config` (topology, culling,
    /// vertex layout) is baked into the pipeline object. Viewport and scissor
    /// are dynamic states — they are supplied at draw time, so the pipeline
    /// does not depend on the swap extent.
    ///
    /// The returned [`RenderPipeline`] owns its Vulkan objects and destroys
    /// them on drop; it must not outlive the backend (documented invariant —
    /// same posture as Zunesha buffers).
    pub fn compile_render_pipeline(
        &self,
        vertex_entry: &str,
        vertex_source: &str,
        fragment_entry: &str,
        fragment_source: &str,
        config: &PipelineConfig,
    ) -> Result<RenderPipeline> {
        let vs_spv = compile_stage(vertex_entry, vertex_source, naga::ShaderStage::Vertex)?;
        let fs_spv = compile_stage(fragment_entry, fragment_source, naga::ShaderStage::Fragment)?;

        let vs_ci = vk::ShaderModuleCreateInfo::default().code(&vs_spv);
        let vs_module = unsafe { self.device.create_shader_module(&vs_ci, None) }.map_err(|e| {
            GraphicsError::CompileFailed {
                stage: "vertex",
                entry: vertex_entry.into(),
                message: format!("vkCreateShaderModule: {e}"),
            }
        })?;
        let fs_ci = vk::ShaderModuleCreateInfo::default().code(&fs_spv);
        let fs_module = unsafe { self.device.create_shader_module(&fs_ci, None) }.map_err(|e| {
            GraphicsError::CompileFailed {
                stage: "fragment",
                entry: fragment_entry.into(),
                message: format!("vkCreateShaderModule: {e}"),
            }
        })?;

        let vs_name = CString::new(vertex_entry).map_err(|_| GraphicsError::CompileFailed {
            stage: "vertex",
            entry: vertex_entry.into(),
            message: "entry name contains a NUL byte".into(),
        })?;
        let fs_name = CString::new(fragment_entry).map_err(|_| GraphicsError::CompileFailed {
            stage: "fragment",
            entry: fragment_entry.into(),
            message: "entry name contains a NUL byte".into(),
        })?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_module)
                .name(&vs_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs_module)
                .name(&fs_name),
        ];

        // Vertex input from the caller's layout (binding 0, interleaved).
        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(config.vertex_layout.stride)
            .input_rate(vk::VertexInputRate::VERTEX);
        let attributes: Vec<vk::VertexInputAttributeDescription> = config
            .vertex_layout
            .attributes
            .iter()
            .map(|a| {
                vk::VertexInputAttributeDescription::default()
                    .binding(0)
                    .location(a.location)
                    .format(to_vk_format(a.format))
                    .offset(a.offset)
            })
            .collect();
        let bindings: Vec<vk::VertexInputBindingDescription> = if config.vertex_layout.stride == 0 {
            Vec::new()
        } else {
            vec![binding]
        };
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(to_vk_topology(config.topology))
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(to_vk_cull(config.cull_mode))
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Empty layout: v0.1 pipelines take no descriptors and no push
        // constants.
        let layout_ci = vk::PipelineLayoutCreateInfo::default();
        let layout =
            unsafe { self.device.create_pipeline_layout(&layout_ci, None) }.map_err(|e| {
                GraphicsError::PipelineFailed {
                    message: format!("vkCreatePipelineLayout: {e}"),
                }
            })?;

        let ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(self.render_pass)
            .subpass(0);
        let pipelines = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&ci),
                None,
            )
        }
        .map_err(|(_, e)| GraphicsError::PipelineFailed {
            message: format!("vkCreateGraphicsPipelines: {e}"),
        })?;
        let Some(&pipeline) = pipelines.first() else {
            return Err(GraphicsError::PipelineFailed {
                message: "vkCreateGraphicsPipelines returned no pipeline".into(),
            });
        };

        let inner = Box::new(VulkanPipelineInner {
            pipeline,
            layout,
            vs_module,
            fs_module,
            device: self.device.clone(),
        });
        Ok(RenderPipeline {
            raw: Box::into_raw(inner) as *mut c_void,
            drop_fn: drop_vulkan_pipeline,
        })
    }

    /// Allocate a GPU buffer and upload `data` — a `zunesha::Buffer`.
    ///
    /// Buffers live in Zunesha's memory domain (ADR 0001): a buffer Borsalino
    /// filled via compute can be bound here as a vertex buffer with **zero
    /// copies**. The buffer self-destructs on drop; it must not outlive the
    /// backend, and in-flight frames reading it must complete first (the
    /// synchronous `present` guarantees this).
    pub fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        let buffer = self.zunesha_device.create_buffer(data).map_err(|e| {
            GraphicsError::BufferCreationFailed {
                message: format!("{e}"),
            }
        })?;
        let len = std::mem::size_of_val(data);
        Ok(GpuBuffer {
            raw: Box::into_raw(Box::new(buffer)) as *mut c_void,
            len,
            drop_fn: drop_zunesha_buffer_box,
        })
    }

    /// Acquire the next frame for rendering.
    ///
    /// Blocks until a swapchain image is available, then begins the frame
    /// command buffer and the render pass (clearing the image to the backend's
    /// clear color) and records the viewport/scissor. The returned [`Frame`]
    /// grants exclusive recording access — draw into it via
    /// [`VulkanBackend::draw`], then present. Dropping an unpresented frame is
    /// well-defined: the recording is discarded (the image is simply not
    /// displayed).
    pub fn acquire_frame(&self) -> Result<Frame> {
        let mut render = self.render.lock().expect("render state lock");

        // Wait for the previous frame's command buffer to finish. The fence
        // is signalled whenever no frame is in flight (created signalled, and
        // left signalled by present's wait), so this passes immediately unless
        // a submission is still executing. The fence is NOT reset here: it is
        // reset only immediately before the submit that will signal it (see
        // `present`), so a dropped frame never leaves an unsignalled fence to
        // deadlock the next acquire.
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&render.fence), true, u64::MAX)
                .map_err(|e| GraphicsError::AcquireFailed {
                    message: format!("fence wait: {e}"),
                })?;
        }

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

        // Reset the command buffer explicitly: a dropped frame leaves it
        // mid-recording (begun, never ended), which vkBeginCommandBuffer alone
        // cannot recover from. The pool's RESET_COMMAND_BUFFER flag permits
        // this, discarding any stale recording.
        unsafe {
            self.device
                .reset_command_buffer(render.cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| GraphicsError::AcquireFailed {
                    message: format!("command buffer reset: {e}"),
                })?;
        }

        // Begin the command buffer and the render pass with a clear.
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(render.cmd, &begin)
                .map_err(|e| GraphicsError::AcquireFailed {
                    message: format!("begin command buffer: {e}"),
                })?;
        }
        let clear = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.1, 0.1, 1.0],
            },
        }];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .clear_values(&clear);
        unsafe {
            self.device
                .cmd_begin_render_pass(render.cmd, &rp_begin, vk::SubpassContents::INLINE);
            // Explicit clear in two sub-rects, on top of the load-op clear.
            //
            // Full-surface clears — load-op or explicit — execute on the
            // driver's "fast clear" metadata path, and at least one current
            // Mesa build decompresses that metadata to converted float
            // garbage on WSI swapchain images (headless). Sub-surface
            // clears write real texels (empirically established — ADR 0002).
            // Two rects covering the whole area between them make the
            // background deterministic on every conformant driver,
            // fast-clear bugs included; on healthy drivers this is a
            // redundant overwrite of identical values. (Degenerate 1-px-high
            // surfaces cannot avoid the full-surface path.)
            let (w, h) = (self.extent.width, self.extent.height);
            let mut clear_rects = Vec::with_capacity(2);
            if h > 1 {
                clear_rects.push(vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: w,
                            height: h - 1,
                        },
                    },
                });
                clear_rects.push(vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D {
                            x: 0,
                            y: h as i32 - 1,
                        },
                        extent: vk::Extent2D {
                            width: w,
                            height: 1,
                        },
                    },
                });
            } else {
                clear_rects.push(vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: w,
                            height: h,
                        },
                    },
                });
            }
            let clear_attachments = [vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: clear[0],
            }];
            self.device
                .cmd_clear_attachments(render.cmd, &clear_attachments, &clear_rects);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device
                .cmd_set_viewport(render.cmd, 0, std::slice::from_ref(&viewport));
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            self.device
                .cmd_set_scissor(render.cmd, 0, std::slice::from_ref(&scissor));
        }
        render.phase = RenderPhase::Open(index);
        drop(render);

        Ok(Frame::from_inner(FrameInner::Acquired { index }))
    }

    /// Record a non-indexed draw call into `frame`.
    ///
    /// `vertices` is bound as vertex input (interpreted per the pipeline's
    /// vertex layout); `vertex_count` is the number of vertices to draw. Call
    /// zero or more times per frame before
    /// [`VulkanBackend::present`].
    pub fn draw(
        &self,
        frame: &mut Frame,
        pipeline: &RenderPipeline,
        vertices: &GpuBuffer,
        vertex_count: u32,
    ) -> Result<()> {
        let Some(frame_index) = frame.peek_index() else {
            return Err(GraphicsError::InvalidDraw {
                message: "frame has no acquired image (already presented?)".into(),
            });
        };
        let pipeline_inner =
            // Safety: `raw` was produced by `Box::into_raw::<VulkanPipelineInner>`
            // in `compile_render_pipeline` and is still valid.
            unsafe { &*(pipeline.raw as *const VulkanPipelineInner) };
        // Safety: `raw` was produced by `Box::into_raw::<zunesha::Buffer>` in
        // `create_buffer` and is still valid.
        let zunesha_buffer = unsafe { &*(vertices.raw as *const zunesha::Buffer) };
        let vertex_buffer = self.zunesha_device.raw_buffer(zunesha_buffer);

        let render = self.render.lock().expect("render state lock");
        if !matches!(render.phase, RenderPhase::Open(i) if i == frame_index) {
            return Err(GraphicsError::InvalidDraw {
                message: format!(
                    "frame {frame_index} is not the frame being recorded ({:?})",
                    render.phase
                ),
            });
        }

        unsafe {
            self.device.cmd_bind_pipeline(
                render.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_inner.pipeline,
            );
            self.device.cmd_bind_vertex_buffers(
                render.cmd,
                0,
                std::slice::from_ref(&vertex_buffer),
                &[0],
            );
            self.device.cmd_draw(render.cmd, vertex_count, 1, 0, 0);
        }
        Ok(())
    }

    /// Finish, submit, and display `frame`, then block until it is safe to
    /// acquire the next frame.
    ///
    /// Ends the render pass and command buffer, submits on the graphics queue
    /// (waiting the acquire semaphore, signalling the present semaphore and
    /// the frame fence), presents, and waits the fence — the synchronous
    /// present contract documented on the trait.
    pub fn present(&self, mut frame: Frame) -> Result<()> {
        let index = frame
            .take_index()
            .ok_or_else(|| GraphicsError::InvalidDraw {
                message: "frame has no acquired image (already presented?)".into(),
            })?;
        let mut render = self.render.lock().expect("render state lock");
        let flushed = match render.phase {
            RenderPhase::Open(i) if i == index => false,
            // A `read_pixels` flush already submitted the rendering and
            // fence-waited it; present without semaphore waits (the image is
            // idle from the host's perspective).
            RenderPhase::Flushed(i) if i == index => true,
            ref other => {
                return Err(GraphicsError::InvalidDraw {
                    message: format!("frame {index} is not the frame being recorded ({other:?})"),
                });
            }
        };

        let present_result = if flushed {
            let swapchains = [self.swapchain];
            let indices = [index];
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .image_indices(&indices);
            // Safety: queue and image are idle (flush fence-waited).
            unsafe {
                self.swapchain_fn
                    .queue_present(self.graphics_queue, &present_info)
            }
        } else {
            unsafe {
                self.device.cmd_end_render_pass(render.cmd);
                self.device.end_command_buffer(render.cmd).map_err(|e| {
                    GraphicsError::PresentFailed {
                        message: format!("end command buffer: {e}"),
                    }
                })?;

                let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let submit = vk::SubmitInfo::default()
                    .wait_semaphores(std::slice::from_ref(&self.image_available))
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(std::slice::from_ref(&render.cmd))
                    .signal_semaphores(std::slice::from_ref(&self.render_finished));
                // Reset the fence only now — immediately before the submit that
                // will signal it — so it is unsignalled for the shortest possible
                // window (unobservable in the synchronous loop).
                self.device
                    .reset_fences(std::slice::from_ref(&render.fence))
                    .map_err(|e| GraphicsError::PresentFailed {
                        message: format!("fence reset: {e}"),
                    })?;
                self.device
                    .queue_submit(
                        self.graphics_queue,
                        std::slice::from_ref(&submit),
                        render.fence,
                    )
                    .map_err(|e| GraphicsError::PresentFailed {
                        message: format!("vkQueueSubmit: {e}"),
                    })?;

                let swapchains = [self.swapchain];
                let indices = [index];
                let present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(std::slice::from_ref(&self.render_finished))
                    .swapchains(&swapchains)
                    .image_indices(&indices);
                self.swapchain_fn
                    .queue_present(self.graphics_queue, &present_info)
            }
        };
        // Block until the frame's work is complete, so the next acquire can
        // safely reset and reuse the command buffer (the synchronous
        // frame-loop contract). In the Flushed path the fence is already
        // signalled (and waited) — the wait returns immediately.
        // Safety: the fence was submitted on this device (present or flush).
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&render.fence), true, u64::MAX)
                .map_err(|e| GraphicsError::PresentFailed {
                    message: format!("fence wait: {e}"),
                })?;
        }

        render.phase = RenderPhase::Idle;
        present_result
            .map(|_| ())
            .map_err(|e| GraphicsError::PresentFailed {
                message: format!("vkQueuePresentKHR: {e}"),
            })
    }

    /// Read the current frame's pixels back to host memory.
    ///
    /// Verification / screenshot path: flushes the pending recording (the
    /// frame loop submits at [`VulkanBackend::present`], so the rendering is
    /// not yet on the GPU), fence-waits it, then barriers the image to a
    /// transfer layout, copies to a Zunesha staging buffer, and reads it
    /// back. Returns tightly packed `w*h*4` bytes, row-major, in the
    /// swapchain format's channel order (B8G8R8A8 under the preferred UNORM
    /// format).
    ///
    /// Call between [`VulkanBackend::draw`] and
    /// [`VulkanBackend::present`] — **before** presenting. The Vulkan spec
    /// does not guarantee a presented image's contents survive the
    /// presentation operation (the WSI may composite or clobber it), and at
    /// least one current Mesa headless-WSI build rewrites the background as
    /// converted float data — verification must read the rendered image,
    /// not the presented one (ADR 0002). Repeatable: a second call on the
    /// same un-presented frame reads stable contents without re-submitting.
    pub fn read_pixels(&self, frame: &Frame) -> Result<Vec<u8>> {
        let Some(index) = frame.peek_index() else {
            return Err(GraphicsError::InvalidDraw {
                message: "frame has no acquired image (already presented?)".into(),
            });
        };
        {
            let mut render = self.render.lock().expect("render state lock");
            match render.phase {
                RenderPhase::Open(i) if i == index => {
                    // Flush the recording: end the pass, submit (consuming the
                    // acquire semaphore), wait the fence. The image contents
                    // are then stable on the host timeline.
                    // Safety: command buffer is in recording state; fence is
                    // signalled (reset immediately before the submit that
                    // signals it).
                    unsafe {
                        self.device.cmd_end_render_pass(render.cmd);
                        self.device
                            .end_command_buffer(render.cmd)
                            .map_err(|e| GraphicsError::Internal(format!("readback end: {e}")))?;
                        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                        let submit = vk::SubmitInfo::default()
                            .wait_semaphores(std::slice::from_ref(&self.image_available))
                            .wait_dst_stage_mask(&wait_stages)
                            .command_buffers(std::slice::from_ref(&render.cmd));
                        self.device
                            .reset_fences(std::slice::from_ref(&render.fence))
                            .map_err(|e| GraphicsError::Internal(format!("fence reset: {e}")))?;
                        self.device
                            .queue_submit(
                                self.graphics_queue,
                                std::slice::from_ref(&submit),
                                render.fence,
                            )
                            .map_err(|e| {
                                GraphicsError::Internal(format!("readback submit: {e}"))
                            })?;
                        self.device
                            .wait_for_fences(std::slice::from_ref(&render.fence), true, u64::MAX)
                            .map_err(|e| GraphicsError::Internal(format!("readback wait: {e}")))?;
                    }
                    render.phase = RenderPhase::Flushed(index);
                }
                // Already flushed (repeat verification read) — contents stable.
                RenderPhase::Flushed(i) if i == index => {}
                ref other => {
                    return Err(GraphicsError::InvalidDraw {
                        message: format!("frame {index} is not the current frame ({other:?})"),
                    });
                }
            }
        }

        let Some(&image) = self.images.get(index as usize) else {
            return Err(GraphicsError::InvalidDraw {
                message: format!("image index {index} out of range"),
            });
        };
        let width = self.extent.width;
        let height = self.extent.height;
        let size = (width as usize) * (height as usize) * 4;

        // Staging via Zunesha (host-visible path), read via Zunesha.
        let staging = self
            .zunesha_device
            .create_buffer(&vec![0u8; size])
            .map_err(|e| GraphicsError::BufferCreationFailed {
                message: format!("staging: {e}"),
            })?;
        let staging_vk = self.zunesha_device.raw_buffer(&staging);

        // One-shot copy command buffer from the same pool.
        let cb_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.device.allocate_command_buffers(&cb_ci) }
            .map_err(|e| GraphicsError::Internal(format!("one-shot allocate: {e}")))?
            .first()
            .copied()
            .ok_or_else(|| GraphicsError::Internal("no one-shot command buffer".into()))?;

        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let to_transfer = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource);
        let to_present = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource);
        let copy = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| GraphicsError::Internal(format!("one-shot begin: {e}")))?;
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_transfer),
            );
            self.device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_vk,
                std::slice::from_ref(&copy),
            );
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&to_present),
            );
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GraphicsError::Internal(format!("one-shot end: {e}")))?;
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            self.device
                .queue_submit(
                    self.graphics_queue,
                    std::slice::from_ref(&submit),
                    vk::Fence::null(),
                )
                .map_err(|e| GraphicsError::Internal(format!("one-shot submit: {e}")))?;
            self.device
                .queue_wait_idle(self.graphics_queue)
                .map_err(|e| GraphicsError::Internal(format!("one-shot wait: {e}")))?;
            self.device
                .free_command_buffers(self.command_pool, std::slice::from_ref(&cmd));
        }

        let pixels: Vec<u8> = self
            .zunesha_device
            .read_buffer(&staging)
            .map_err(|e| GraphicsError::Internal(format!("staging readback: {e}")))?;
        drop(staging);
        Ok(pixels)
    }
}

// ── GraphicsBackend trait impl ────────────────────────────────────

impl crate::GraphicsBackend for VulkanBackend<'_> {
    fn compile_render_pipeline(
        &self,
        vertex_entry: &str,
        vertex_source: &str,
        fragment_entry: &str,
        fragment_source: &str,
        config: &PipelineConfig,
    ) -> Result<RenderPipeline> {
        VulkanBackend::compile_render_pipeline(
            self,
            vertex_entry,
            vertex_source,
            fragment_entry,
            fragment_source,
            config,
        )
    }

    fn create_buffer<T: bytemuck::Pod>(&self, data: &[T]) -> Result<GpuBuffer> {
        VulkanBackend::create_buffer(self, data)
    }

    fn acquire_frame(&self) -> Result<Frame> {
        VulkanBackend::acquire_frame(self)
    }

    fn draw(
        &self,
        frame: &mut Frame,
        pipeline: &RenderPipeline,
        vertices: &GpuBuffer,
        vertex_count: u32,
    ) -> Result<()> {
        VulkanBackend::draw(self, frame, pipeline, vertices, vertex_count)
    }

    fn present(&self, frame: Frame) -> Result<()> {
        VulkanBackend::present(self, frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::ffi::c_void;
    use zunesha::InitRequest;

    /// Test context owning the device-lifetime resources. Field order is the
    /// drop order (declaration order): the headless surface must be destroyed
    /// before the device that owns its instance.
    struct TestCtx {
        headless: HeadlessSurface,
        _entry: Entry,
        device: zunesha::vulkan::VulkanDevice,
    }

    fn rig() -> Option<TestCtx> {
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
        Some(TestCtx {
            headless,
            _entry: entry,
            device,
        })
    }

    impl TestCtx {
        /// Build a backend borrowing this context's device. The backend is a
        /// local in each test, so it drops before the context — the borrow
        /// checker verifies it.
        fn backend(&self) -> VulkanBackend<'_> {
            let surface = SurfaceHandle::VulkanSurface {
                instance: std::ptr::null_mut(),
                surface: self.headless.handle().as_raw() as usize as *mut c_void,
            };
            VulkanBackend::new(&self.device, surface).expect("VulkanBackend::new")
        }
    }

    /// A flat-triangle vertex layout (vec2<f32> at location 0).
    fn flat_config() -> PipelineConfig {
        PipelineConfig {
            topology: Topology::TriangleList,
            cull_mode: CullMode::None,
            vertex_layout: crate::VertexLayout {
                stride: 8,
                attributes: vec![crate::VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x2,
                }],
            },
        }
    }

    /// The backend constructs with a swapchain, render pass, framebuffers,
    /// and a sane extent.
    #[test]
    #[serial]
    fn backend_constructs_with_swapchain() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        assert!(
            backend.image_count() >= 2,
            "FIFO swapchain needs >= 2 images"
        );
        assert!(backend.extent().width > 0 && backend.extent().height > 0);
        println!(
            "swapchain: {} images, extent = {}x{}",
            backend.image_count(),
            backend.extent().width,
            backend.extent().height
        );
    }

    /// The frame lifecycle: acquire → present, repeatedly.
    #[test]
    #[serial]
    fn acquire_present_roundtrip() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        for i in 0..3 {
            let frame = backend.acquire_frame().expect("acquire_frame");
            backend.present(frame).expect("present");
            println!("frame {i} acquired + presented");
        }
    }

    /// Dropping an unpresented frame is well-defined: the next acquire works.
    #[test]
    #[serial]
    fn dropping_unpresented_frame_is_safe() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        let frame = backend.acquire_frame().expect("acquire_frame");
        drop(frame); // not presented
        let frame = backend.acquire_frame().expect("acquire after drop");
        backend.present(frame).expect("present after drop");
    }

    /// A simple flat-triangle pipeline compiles from WGSL via naga.
    #[test]
    #[serial]
    fn pipeline_compiles_from_flat_triangle_shaders() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        let pipeline = backend
            .compile_render_pipeline(
                "vs_main",
                crate::kernels::FLAT_TRIANGLE_VERT,
                "fs_main",
                crate::kernels::FLAT_TRIANGLE_FRAG,
                &flat_config(),
            )
            .expect("compile_render_pipeline");
        drop(pipeline);
        println!("flat-triangle pipeline compiled + dropped cleanly");
    }

    /// Malformed WGSL is rejected as a vertex CompileFailed with source
    /// location context.
    #[test]
    #[serial]
    fn bad_wgsl_is_rejected_with_stage_context() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        let err = backend
            .compile_render_pipeline(
                "vs_main",
                "this is not wgsl at all",
                "fs_main",
                crate::kernels::FLAT_TRIANGLE_FRAG,
                &PipelineConfig::default(),
            )
            .expect_err("bad source must fail");
        match err {
            GraphicsError::CompileFailed { stage, entry, .. } => {
                assert_eq!(stage, "vertex");
                assert_eq!(entry, "vs_main");
            }
            other => panic!("expected CompileFailed, got {other:?}"),
        }
    }

    /// A missing entry point is caught by naga (PipelineOptions), not the
    /// driver.
    #[test]
    #[serial]
    fn missing_entry_point_is_rejected() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        let err = backend
            .compile_render_pipeline(
                "no_such_entry",
                crate::kernels::FLAT_TRIANGLE_VERT,
                "fs_main",
                crate::kernels::FLAT_TRIANGLE_FRAG,
                &PipelineConfig::default(),
            )
            .expect_err("missing entry must fail");
        assert!(matches!(err, GraphicsError::CompileFailed { .. }));
    }

    /// The complete path: draw a flat triangle, present, and read the pixels
    /// back — proving the triangle actually rendered over the clear color.
    ///
    /// The fragment shader is magenta (1,0,1) and the clear is (0.1,0.1,0.1) —
    /// both channel-order symmetric, so the assertions hold under either
    /// B8G8R8A8 or R8G8B8A8 packing.
    #[test]
    #[serial]
    fn draw_renders_triangle_verified_by_readback() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        let pipeline = backend
            .compile_render_pipeline(
                "vs_main",
                crate::kernels::FLAT_TRIANGLE_VERT,
                "fs_main",
                crate::kernels::FLAT_TRIANGLE_FRAG,
                &flat_config(),
            )
            .expect("pipeline");
        // Triangle spanning much of the viewport in NDC.
        let vertices: [f32; 6] = [-0.5, -0.5, 0.5, -0.5, 0.0, 0.5];
        let vb = backend.create_buffer(&vertices).expect("vertex buffer");

        let mut frame = backend.acquire_frame().expect("acquire");
        let _index = frame.peek_index().expect("frame index");
        backend.draw(&mut frame, &pipeline, &vb, 3).expect("draw");
        // Verification reads AFTER the render completes but BEFORE present —
        // presented-image contents are not guaranteed to survive the WSI's
        // presentation (spec) and current Mesa headless WSI mutates them
        // (ADR 0002). `read_pixels` flushes the pending recording itself.
        let pixels = backend
            .read_pixels(&frame)
            .expect("read_pixels pre-present");
        let pixels_twice = backend
            .read_pixels(&frame)
            .expect("read_pixels is idempotent");
        assert_eq!(
            pixels, pixels_twice,
            "second read of an unflushed-but-flushed frame"
        );
        backend.present(frame).expect("present after readback");
        let w = backend.extent().width as usize;
        let h = backend.extent().height as usize;
        assert_eq!(pixels.len(), w * h * 4, "tightly packed w*h*4 bytes");

        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = (y * w + x) * 4;
            [pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]]
        };
        // Inside the triangle (its centroid projects slightly below centre).
        let inside = px(w / 2, h * 7 / 12);
        // A corner far outside the triangle.
        let outside = px(8, 8);
        println!("inside pixel = {inside:?}, outside pixel = {outside:?}");

        // Magenta: (255, 0, 255, 255) — R==B==255, G==0 regardless of channel
        // order. Clear: 0.1 → ~26 for every channel.
        assert!(
            inside[0] == 255 && inside[2] == 255 && inside[1] == 0,
            "inside pixel should be magenta, got {inside:?}"
        );
        assert!(
            outside[0] == outside[1] && outside[1] == outside[2] && (24..=28).contains(&outside[0]),
            "outside pixel should be the 0.1 gray clear, got {outside:?}"
        );

        // Not every pixel is the clear color → something was drawn.
        let magenta_count = pixels
            .chunks_exact(4)
            .filter(|c| c[0] == 255 && c[1] == 0 && c[2] == 255)
            .count();
        println!("magenta pixels: {magenta_count} / {}", w * h);
        assert!(
            magenta_count > 0 && magenta_count < w * h,
            "triangle covers part of the image"
        );
    }

    /// Drawing into a frame that is not the one being recorded is rejected —
    /// the frame-lifecycle guard.
    #[test]
    #[serial]
    fn draw_rejects_stale_frame() {
        let Some(ctx) = rig() else { return };
        let backend = ctx.backend();
        let pipeline = backend
            .compile_render_pipeline(
                "vs_main",
                crate::kernels::FLAT_TRIANGLE_VERT,
                "fs_main",
                crate::kernels::FLAT_TRIANGLE_FRAG,
                &flat_config(),
            )
            .expect("pipeline");
        let vb = backend
            .create_buffer(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0])
            .expect("vb");

        // Acquire a frame, drop it, acquire another — the first frame's index
        // is stale (the second frame is the one being recorded).
        let stale = backend.acquire_frame().expect("acquire");
        let stale_index = stale.peek_index().unwrap();
        drop(stale);
        let _current = backend.acquire_frame().expect("acquire 2");

        // Build a stale Frame manually with the old index.
        let mut stale = Frame::from_inner(FrameInner::Acquired { index: stale_index });
        assert!(matches!(
            backend.draw(&mut stale, &pipeline, &vb, 3),
            Err(GraphicsError::InvalidDraw { .. })
        ));
        // `_current` is dropped unpresented below — safe by design.
    }
}
