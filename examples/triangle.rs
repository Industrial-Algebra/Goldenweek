// Copyright (C) 2026 Industrial Algebra
// SPDX-License-Identifier: Apache-2.0

// Mirrors the cfg of `goldenweek::vulkan` (not macOS).
#![cfg(all(feature = "vulkan", not(target_os = "macos")))]

//! Goldenweek's reference verification image, in runnable form: a flat
//! magenta triangle rasterized over a dark clear and *verified by pixel
//! readback* — the same proof the test suite runs, promoted to a binary so
//! the public API is pinned to something a consumer can run (report
//! recommendation, carried by three pulses).
//!
//! Run (feature-gated; see ADR 0002 for the driver matrix and pinning
//! rationale):
//!
//! ```sh
//! cargo run --features vulkan --example triangle
//! # Pin a specific device (default "Intel" — the headless-capable path):
//! GOLDENWEEK_TEST_DEVICE=intel cargo run --features vulkan --example triangle
//! ```

use std::process::ExitCode;

use ash::Entry;
use ash::vk::Handle;
use goldenweek::kernels::{FLAT_TRIANGLE_FRAG, FLAT_TRIANGLE_VERT};
use goldenweek::vulkan::{HeadlessSurface, VulkanBackend};
use goldenweek::{
    CullMode, PipelineConfig, SurfaceHandle, Topology, VertexAttribute, VertexFormat, VertexLayout,
};
use zunesha::InitRequest;

fn main() -> ExitCode {
    // Same pin convention as the test rig (ADR 0002): default to the
    // headless-capable Mesa path; skip — not fail — when there is no device.
    let hint = std::env::var("GOLDENWEEK_TEST_DEVICE").unwrap_or_else(|_| "Intel".into());
    let request = InitRequest::prefer_graphics().with_device_hint(hint);
    let device = match zunesha::init_with(request) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("triangle: no Vulkan device ({e}) — nothing to do on this host");
            return ExitCode::SUCCESS;
        }
    };

    // Headless surface over the device's instance. Locals drop in reverse
    // declaration order, so the surface dies before the device (the
    // instance must outlive the surface).
    let entry = unsafe { Entry::load() }.expect("vulkan loader");
    let surface = match unsafe { HeadlessSurface::new(&entry, &device.raw_instance()) } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("triangle: headless surface unavailable ({e}) — see ADR 0002");
            return ExitCode::SUCCESS;
        }
    };

    let handle = SurfaceHandle::VulkanSurface {
        instance: std::ptr::null_mut(),
        surface: surface.handle().as_raw() as usize as *mut std::ffi::c_void,
    };
    let backend = match VulkanBackend::new(&device, handle) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "triangle: backend init failed ({e}) — driver may stub headless swapchains (ADR 0002)"
            );
            return ExitCode::SUCCESS;
        }
    };

    // The reference pipeline: passthrough vec2 positions, flat magenta output.
    let config = PipelineConfig {
        topology: Topology::TriangleList,
        cull_mode: CullMode::None,
        vertex_layout: VertexLayout {
            stride: 8,
            attributes: vec![VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32x2,
            }],
        },
    };
    let pipeline = match backend.compile_render_pipeline(
        "vs_main",
        FLAT_TRIANGLE_VERT,
        "fs_main",
        FLAT_TRIANGLE_FRAG,
        &config,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("triangle: pipeline compile failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    // A triangle spanning much of the viewport in NDC.
    let vertices: [f32; 6] = [-0.5, -0.5, 0.5, -0.5, 0.0, 0.5];
    let vertex_buffer = match backend.create_buffer(&vertices) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("triangle: vertex buffer failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut frame = match backend.acquire_frame() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("triangle: acquire failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = backend.draw(&mut frame, &pipeline, &vertex_buffer, 3) {
        eprintln!("triangle: draw failed: {e}");
        return ExitCode::FAILURE;
    }

    // Verify BEFORE presenting: presented-image contents are not guaranteed
    // to survive the WSI (ADR 0002) — read_pixels flushes the render itself.
    let pixels = match backend.read_pixels(&frame) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("triangle: readback failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = backend.present(frame) {
        eprintln!("triangle: present failed: {e}");
        return ExitCode::FAILURE;
    }
    let w = backend.extent().width as usize;
    let h = backend.extent().height as usize;
    assert_eq!(pixels.len(), w * h * 4, "tightly packed w*h*4 bytes");

    let px = |x: usize, y: usize| -> [u8; 4] {
        let o = (y * w + x) * 4;
        [pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]]
    };
    let inside = px(w / 2, h * 7 / 12);
    let outside = px(8, 8);
    println!("inside pixel = {inside:?}, outside pixel = {outside:?}");

    // Magenta and the 0.1 gray clear are channel-order-symmetric (ADR 0002):
    // the assertions hold under both B8G8R8A8 and R8G8B8A8 swapchains.
    if !(inside[0] == 255 && inside[1] == 0 && inside[2] == 255) {
        eprintln!("triangle: inside pixel is not magenta — verification FAILED");
        return ExitCode::FAILURE;
    }
    if !(outside[0] == outside[1] && outside[1] == outside[2] && (24..=28).contains(&outside[0])) {
        eprintln!("triangle: outside pixel is not the 0.1 gray clear — verification FAILED");
        return ExitCode::FAILURE;
    }
    println!("triangle: verified — magenta rendered and read back ({w}x{h})");
    ExitCode::SUCCESS
}
