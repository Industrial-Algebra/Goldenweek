// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Goldenweek's M1 textured-quad verification image, in runnable form: a 2×2
//! RGBA8 texture sampled across a fullscreen quad and *verified by pixel
//! readback* — each quadrant's centre must equal its texel.
//!
//! The fragment shader declares its texture and sampler as the two WGSL
//! globals naga's WGSL→SPIR-V path produces (`@binding(0)` image,
//! `@binding(1)` sampler); the backend's texture set layout declares exactly
//! that split at set 0. (WGSL has no combined-image-sampler type, so the
//! single-binding combined form is not expressible through naga.)
//!
//! Run (feature-gated; see ADR 0002 for the driver matrix and pinning
//! rationale):
//!
//! ```sh
//! cargo run --features vulkan --example textured
//! # Pin a specific device (default "Intel" — the headless-capable path):
//! GOLDENWEEK_TEST_DEVICE=intel cargo run --features vulkan --example textured
//! ```

#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
mod textured {
    use std::process::ExitCode;

    use ash::Entry;
    use ash::vk::Handle;
    use goldenweek::vulkan::{HeadlessSurface, VulkanBackend};
    use goldenweek::{
        CullMode, FilterMode, PipelineConfig, SurfaceHandle, TextureDesc, Topology,
        VertexAttribute, VertexFormat, VertexLayout,
    };
    use zunesha::InitRequest;

    /// Vertex shader: `(x, y)` position + `(u, v)` uv in, uv interpolated out.
    ///
    /// The quad is a triangle strip in NDC (y-up) with uv origin top-left, so
    /// the top of the screen samples texture row 0.
    const TEXTURED_VERT: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VOut {
    return VOut(vec4<f32>(pos, 0.0, 1.0), uv);
}
"#;

    /// Fragment shader: sample set 0. naga emits a sampled image at binding 0
    /// and its sampler at binding 1 for these two globals.
    const TEXTURED_FRAG: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(@location(1) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, uv);
}
"#;

    /// The 2×2, four-distinct-texel RGBA8 payload (row-major, top-left
    /// origin): red, green / blue, yellow.
    const QUAD_TEXELS: [u8; 16] = [
        255, 0, 0, 255, // red
        0, 255, 0, 255, // green
        0, 0, 255, 255, // blue
        255, 255, 0, 255, // yellow
    ];

    /// The fullscreen quad (triangle strip): `(x, y, u, v)` per vertex.
    const QUAD_VERTICES: [f32; 16] = [
        -1.0, -1.0, 0.0, 1.0, // bottom-left  → blue
        1.0, -1.0, 1.0, 1.0, // bottom-right → yellow
        -1.0, 1.0, 0.0, 0.0, // top-left     → red
        1.0, 1.0, 1.0, 0.0, // top-right    → green
    ];

    pub(super) fn run() -> ExitCode {
        // Same pin convention as the test rig (ADR 0002): default to the
        // headless-capable Mesa path; skip — not fail — when there is no
        // device.
        let hint = std::env::var("GOLDENWEEK_TEST_DEVICE").unwrap_or_else(|_| "Intel".into());
        let request = InitRequest::prefer_graphics().with_device_hint(hint);
        let device = match zunesha::init_with(request) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("textured: no Vulkan device ({e}) — nothing to do on this host");
                return ExitCode::SUCCESS;
            }
        };

        // Headless surface over the device's instance. Locals drop in reverse
        // declaration order, so the surface dies before the device.
        let entry = unsafe { Entry::load() }.expect("vulkan loader");
        let surface = match unsafe { HeadlessSurface::new(&entry, &device.raw_instance()) } {
            Ok(s) => s,
            Err(e) => {
                eprintln!("textured: headless surface unavailable ({e}) — see ADR 0002");
                return ExitCode::SUCCESS;
            }
        };

        let handle = SurfaceHandle::VulkanSurface {
            surface: surface.handle().as_raw() as usize as *mut std::ffi::c_void,
        };
        let backend = match VulkanBackend::new(&device, handle) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "textured: backend init failed ({e}) — driver may stub headless swapchains (ADR 0002)"
                );
                return ExitCode::SUCCESS;
            }
        };

        // Textured pipeline: triangle-strip quad, vec2 pos + vec2 uv.
        let config = PipelineConfig {
            topology: Topology::TriangleStrip,
            cull_mode: CullMode::None,
            depth: None,
            blend: None,
            vertex_layout: VertexLayout {
                stride: 16,
                attributes: vec![
                    VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x2,
                    },
                    VertexAttribute {
                        location: 1,
                        offset: 8,
                        format: VertexFormat::Float32x2,
                    },
                ],
            },
        };
        let pipeline = match backend.compile_render_pipeline(
            "vs_main",
            TEXTURED_VERT,
            "fs_main",
            TEXTURED_FRAG,
            &config,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("textured: pipeline compile failed: {e}");
                return ExitCode::FAILURE;
            }
        };

        let texture = match backend.create_texture(
            &TextureDesc {
                width: 2,
                height: 2,
            },
            &QUAD_TEXELS,
            FilterMode::Nearest,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("textured: texture creation failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let vertex_buffer = match backend.create_buffer(&QUAD_VERTICES) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("textured: vertex buffer failed: {e}");
                return ExitCode::FAILURE;
            }
        };

        let mut frame = match backend.acquire_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("textured: acquire failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = backend.draw_textured(&mut frame, &pipeline, &vertex_buffer, 4, &texture) {
            eprintln!("textured: draw failed: {e}");
            return ExitCode::FAILURE;
        }

        // Verify BEFORE presenting (ADR 0002): `read_pixels` flushes the
        // render.
        let pixels = match backend.read_pixels(&frame) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("textured: readback failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = backend.present(frame) {
            eprintln!("textured: present failed: {e}");
            return ExitCode::FAILURE;
        }
        let w = backend.extent().width as usize;
        let h = backend.extent().height as usize;
        assert_eq!(pixels.len(), w * h * 4, "tightly packed w*h*4 bytes");

        // The readback channel order follows the swapchain format.
        let readback = |texel: [u8; 4]| -> [u8; 4] {
            match backend.format() {
                ash::vk::Format::B8G8R8A8_UNORM | ash::vk::Format::B8G8R8A8_SRGB => {
                    [texel[2], texel[1], texel[0], texel[3]]
                }
                _ => texel,
            }
        };
        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = (y * w + x) * 4;
            [pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]]
        };
        // Quadrant centres (integer pixels): TL, TR, BL, BR.
        let tl = px(w / 4, h / 4);
        let tr = px(3 * w / 4, h / 4);
        let bl = px(w / 4, 3 * h / 4);
        let br = px(3 * w / 4, 3 * h / 4);
        println!("quadrants: TL={tl:?} TR={tr:?} BL={bl:?} BR={br:?}");
        println!("swapchain format {:?}", backend.format());

        // Each quadrant's centre must equal its texel, channel-order-aware.
        let checks = [
            (tl, readback([255, 0, 0, 255]), "TL red"),
            (tr, readback([0, 255, 0, 255]), "TR green"),
            (bl, readback([0, 0, 255, 255]), "BL blue"),
            (br, readback([255, 255, 0, 255]), "BR yellow"),
        ];
        let mut seen: Vec<[u8; 4]> = Vec::new();
        for (got, want, label) in checks {
            if got != want {
                eprintln!("textured: {label} quadrant is {got:?}, expected {want:?} — FAILED");
                return ExitCode::FAILURE;
            }
            if seen.contains(&got) {
                eprintln!("textured: quadrant texels are not distinct — FAILED");
                return ExitCode::FAILURE;
            }
            seen.push(got);
        }
        println!("textured: verified — 4 distinct quadrant texels sampled");
        ExitCode::SUCCESS
    }
}

#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
fn main() -> std::process::ExitCode {
    textured::run()
}

#[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
fn main() {}
