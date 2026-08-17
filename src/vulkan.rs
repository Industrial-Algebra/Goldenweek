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

use std::ffi::{CString, c_void};

use ash::vk::Handle;
use ash::{Entry, ext, khr, vk};

use crate::frame::{Frame, FrameInner};
use crate::{
    CullMode, GraphicsError, PipelineConfig, RenderPipeline, Result, SurfaceHandle, Topology,
    VertexFormat,
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

// ── Pipeline-config translation ──────────────────────────────────

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
    /// Chosen swapchain image format (render-pass attachment format).
    /// Read by readback verification in increment 5.
    #[allow(dead_code)]
    format: vk::Format,
    /// One image view per swapchain image.
    image_views: Vec<vk::ImageView>,
    /// Render pass: single color attachment, clear→store, UNDEFINED→PRESENT.
    render_pass: vk::RenderPass,
    /// One framebuffer per swapchain image, sized to `extent`.
    framebuffers: Vec<vk::Framebuffer>,
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
            for &fb in &self.framebuffers {
                self.device.destroy_framebuffer(fb, None);
            }
            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
        }
    }
}

// ── Pipeline inner + drop ───────────────────────────────────────

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

/// Drop function stored in [`RenderPipeline`] — drops the boxed inner.
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
        // Image views + render pass + framebuffers — the swapchain's rendering
        // surface. One framebuffer per swapchain image, all sized to `extent`.
        let images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }
            .map_err(|e| GraphicsError::InitFailed(format!("swapchain images: {e}")))?;
        let actual_count = images.len() as u32;
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
            format: format.format,
            image_views,
            render_pass,
            framebuffers,
            graphics_queue,
            graphics_queue_family: graphics.family_index,
            image_available,
        })
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

    /// A simple flat-triangle pipeline compiles from WGSL via naga.
    #[test]
    #[serial]
    fn pipeline_compiles_from_flat_triangle_shaders() {
        let Some(rig) = rig() else { return };
        let config = PipelineConfig {
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
        };
        let pipeline = rig
            .backend
            .compile_render_pipeline(
                "vs_main",
                crate::kernels::FLAT_TRIANGLE_VERT,
                "fs_main",
                crate::kernels::FLAT_TRIANGLE_FRAG,
                &config,
            )
            .expect("compile_render_pipeline");
        drop(pipeline); // exercises the pipeline drop path
        println!("flat-triangle pipeline compiled + dropped cleanly");
    }

    /// Malformed WGSL is rejected as a vertex CompileFailed with source
    /// location context.
    #[test]
    #[serial]
    fn bad_wgsl_is_rejected_with_stage_context() {
        let Some(rig) = rig() else { return };
        let err = rig
            .backend
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
        let Some(rig) = rig() else { return };
        let err = rig
            .backend
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
}
