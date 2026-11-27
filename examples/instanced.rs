// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

//! Goldenweek's M3 flagship, in runnable form: one `draw_instanced` call,
//! four instances — each a small quad with its own per-instance offset and
//! color — *verified by pixel readback* at each instance's centre.
//!
//! The pipeline carries the v0.1 two-binding shape (ADR: M3 plan): binding 0
//! is the per-vertex quad geometry (Vertex rate), binding 1 the per-instance
//! data (Instance rate). The vertex shader adds the per-instance offset and
//! passes the per-instance color through; the fragment shader just emits it.
//!
//! Run (feature-gated; see ADR 0002 for the driver matrix and pinning
//! rationale):
//!
//! ```sh
//! cargo run --features vulkan --example instanced
//! # Pin a specific device (default "Intel" — the headless-capable path):
//! GOLDENWEEK_TEST_DEVICE=intel cargo run --features vulkan --example instanced
//! ```

#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
mod instanced {
    use std::process::ExitCode;

    use ash::Entry;
    use ash::vk::Handle;
    use goldenweek::vulkan::{HeadlessSurface, VulkanBackend};
    use goldenweek::{
        CullMode, PipelineConfig, SurfaceHandle, Topology, VertexAttribute, VertexBindingLayout,
        VertexFormat, VertexInputRate,
    };
    use zunesha::InitRequest;

    /// Vertex shader: per-vertex quad position + per-instance offset and
    /// color. Two bindings, three locations, one draw call.
    const INSTANCED_VERT: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) quad_pos: vec2<f32>,
    @location(1) instance_offset: vec2<f32>,
    @location(2) instance_color: vec3<f32>,
) -> VOut {
    let world = quad_pos + instance_offset;
    return VOut(vec4<f32>(world, 0.0, 1.0), instance_color);
}
"#;

    const INSTANCED_FRAG: &str = r#"
@fragment
fn fs_main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
"#;

    /// Half-extent of each instance's quad (NDC units).
    const QUAD_HALF: f32 = 0.2;

    /// The unit quad (triangle strip), centred on the origin: `(x, y)`.
    const QUAD_VERTICES: [f32; 8] = [
        -QUAD_HALF, -QUAD_HALF, // bottom-left
        QUAD_HALF, -QUAD_HALF, // bottom-right
        -QUAD_HALF, QUAD_HALF, // top-left
        QUAD_HALF, QUAD_HALF, // top-right
    ];

    /// Per-instance data: `(offset_x, offset_y, r, g, b)` — four quads side
    /// by side, red / green / blue / yellow. The colors are chosen so a
    /// channel-order swap between formats is detectable, exactly like the
    /// textured example's readback discipline.
    const INSTANCES: [f32; 20] = [
        -0.75, 0.0, 1.0, 0.0, 0.0, // 0: red
        -0.25, 0.0, 0.0, 1.0, 0.0, // 1: green
        0.25, 0.0, 0.0, 0.0, 1.0, // 2: blue
        0.75, 0.0, 1.0, 1.0, 0.0, // 3: yellow
    ];
    const INSTANCE_COUNT: u32 = 4;

    pub(super) fn run() -> ExitCode {
        // Same pin convention as the test rig (ADR 0002): default to the
        // headless-capable Mesa path; skip — not fail — when there is no
        // device.
        let hint = std::env::var("GOLDENWEEK_TEST_DEVICE").unwrap_or_else(|_| "Intel".into());
        let request = InitRequest::prefer_graphics().with_device_hint(hint);
        let device = match zunesha::init_with(request) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("instanced: no Vulkan device ({e}) — nothing to do on this host");
                return ExitCode::SUCCESS;
            }
        };

        let entry = unsafe { Entry::load() }.expect("vulkan loader");
        let surface = match unsafe { HeadlessSurface::new(&entry, &device.raw_instance()) } {
            Ok(s) => s,
            Err(e) => {
                eprintln!("instanced: headless surface unavailable ({e}) — see ADR 0002");
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
                    "instanced: backend init failed ({e}) — driver may stub headless swapchains (ADR 0002)"
                );
                return ExitCode::SUCCESS;
            }
        };

        // Two bindings: geometry at 0 (Vertex rate), per-instance at 1
        // (Instance rate) — the v0.1 arity, one of each.
        let config = PipelineConfig {
            topology: Topology::TriangleStrip,
            cull_mode: CullMode::None,
            depth: None,
            blend: None,
            vertex_bindings: vec![
                VertexBindingLayout {
                    stride: 8,
                    rate: VertexInputRate::Vertex,
                    attributes: vec![VertexAttribute {
                        location: 0,
                        offset: 0,
                        format: VertexFormat::Float32x2,
                    }],
                },
                VertexBindingLayout {
                    stride: 20,
                    rate: VertexInputRate::Instance,
                    attributes: vec![
                        VertexAttribute {
                            location: 1,
                            offset: 0,
                            format: VertexFormat::Float32x2,
                        },
                        VertexAttribute {
                            location: 2,
                            offset: 8,
                            format: VertexFormat::Float32x3,
                        },
                    ],
                },
            ],
        };
        let pipeline = match backend.compile_render_pipeline(
            "vs_main",
            INSTANCED_VERT,
            "fs_main",
            INSTANCED_FRAG,
            &config,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("instanced: pipeline compile failed: {e}");
                return ExitCode::FAILURE;
            }
        };

        let vertices = match backend.create_buffer(&QUAD_VERTICES) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("instanced: vertex buffer failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        let instances = match backend.create_buffer(&INSTANCES) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("instanced: instance buffer failed: {e}");
                return ExitCode::FAILURE;
            }
        };

        let mut frame = match backend.acquire_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("instanced: acquire failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = backend.draw_instanced(
            &mut frame,
            &pipeline,
            &vertices,
            &instances,
            4,
            INSTANCE_COUNT,
        ) {
            eprintln!("instanced: draw failed: {e}");
            return ExitCode::FAILURE;
        }

        // Verify BEFORE presenting (ADR 0002): `read_pixels` flushes the
        // render.
        let pixels = match backend.read_pixels(&frame) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("instanced: readback failed: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = backend.present(frame) {
            eprintln!("instanced: present failed: {e}");
            return ExitCode::FAILURE;
        }
        let w = backend.extent().width as usize;
        let h = backend.extent().height as usize;
        assert_eq!(pixels.len(), w * h * 4, "tightly packed w*h*4 bytes");

        // The readback channel order follows the swapchain format.
        let readback = |texel: [u8; 3]| -> [u8; 4] {
            match backend.format() {
                ash::vk::Format::B8G8R8A8_UNORM | ash::vk::Format::B8G8R8A8_SRGB => {
                    [texel[2], texel[1], texel[0], 255]
                }
                _ => [texel[0], texel[1], texel[2], 255],
            }
        };
        let px = |x: usize, y: usize| -> [u8; 4] {
            let o = (y * w + x) * 4;
            [pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]]
        };

        // Each instance's centre must be that instance's color — the
        // per-instance data actually reached the shader.
        let expected: [(f32, [u8; 3], &str); 4] = [
            (-0.75, [255, 0, 0], "0 red"),
            (-0.25, [0, 255, 0], "1 green"),
            (0.25, [0, 0, 255], "2 blue"),
            (0.75, [255, 255, 0], "3 yellow"),
        ];
        let mut seen: Vec<[u8; 4]> = Vec::new();
        for (offset_x, want, label) in expected {
            let cx = (((offset_x + 1.0) / 2.0) * w as f32) as usize;
            let cy = h / 2;
            let got = px(cx.min(w - 1), cy);
            let want = readback(want);
            if got != want {
                eprintln!(
                    "instanced: instance {label} centre is {got:?}, expected {want:?} — FAILED"
                );
                return ExitCode::FAILURE;
            }
            if seen.contains(&got) {
                eprintln!("instanced: instance colors are not distinct — FAILED");
                return ExitCode::FAILURE;
            }
            seen.push(got);
        }
        println!(
            "instanced: verified — one draw call, 4 instances, each at its own position in its own color"
        );
        ExitCode::SUCCESS
    }
}

#[cfg(all(feature = "vulkan", not(target_os = "macos")))]
fn main() -> std::process::ExitCode {
    instanced::run()
}

#[cfg(not(all(feature = "vulkan", not(target_os = "macos"))))]
fn main() {}
