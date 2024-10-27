use three_d::{context, renderer::*, FrameInputGenerator, SurfaceSettings, WindowedContext};
use std::sync::Mutex;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

lazy_static::lazy_static! {
    static ref CAMERA_INSTANCE: Mutex<Option<Camera>> = Mutex::new(None);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn resize_callback(width: u32, height: u32) {
    log::info!("Resized to {:?}", (width, height));
    if let Some(camera) = CAMERA_INSTANCE.lock().unwrap().as_mut() {
        camera.set_viewport(Viewport::new_at_origo(width, height));
    }
}

pub fn main() {
    let event_loop = winit::event_loop::EventLoop::new();

    #[cfg(not(target_arch = "wasm32"))]
    let window_builder = winit::window::WindowBuilder::new()
        .with_title("Full-Screen Window")
        .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080))
        .with_decorations(false)
        .with_maximized(true);

    #[cfg(target_arch = "wasm32")]
    let window_builder = {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        winit::window::WindowBuilder::new()
            .with_canvas(Some(
                web_sys::window()
                    .unwrap()
                    .document()
                    .unwrap()
                    .get_element_by_id("canvas")
                    .unwrap()
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .unwrap(),
            ))
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080))
            .with_prevent_default(true)
    };

    let window = window_builder.build(&event_loop).unwrap();
    let context = WindowedContext::from_winit_window(&window, SurfaceSettings::default()).unwrap();

    let mut camera = Camera::new_perspective(
        Viewport::new_at_origo(1, 1), 
        vec3(0.0, 2.0, 4.0),            // Camera position
        vec3(0.0, 0.0, 0.0),              // Target position (where the camera is looking)
        vec3(0.0, 1.0, 0.0),                  // Up direction
        degrees(45.0),                  // Field of view
        0.1,                                     // Near clipping plane
        100.0,                                    // Far clipping plane
    );

    *CAMERA_INSTANCE.lock().unwrap() = Some(camera.clone());

    let mut control = OrbitControl::new(*camera.target(), 1.0, 100.0);

    let mut model = Gm::new(
        Mesh::new(&context, &CpuMesh::cube()),
        ColorMaterial {
            color: Srgba::GREEN,
            ..Default::default()
        },
    );
    model.set_animation(|time| Mat4::from_angle_y(radians(time * 0.0005)));

    let mut frame_input_generator = FrameInputGenerator::from_winit_window(&window);
    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::MainEventsCleared => {
            window.request_redraw();
        }
        winit::event::Event::RedrawRequested(_) => {
            let mut frame_input = frame_input_generator.generate(&context);

            control.handle_events(&mut camera, &mut frame_input.events);
            camera.set_viewport(frame_input.viewport);
            model.animate(frame_input.accumulated_time as f32);
            frame_input
                .screen()
                .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
                .render(&camera, &model, &[]);

            context.swap_buffers().unwrap();
            control_flow.set_poll();
            window.request_redraw();
        }
        winit::event::Event::WindowEvent { ref event, .. } => {
            frame_input_generator.handle_winit_window_event(event);
            match event {
                winit::event::WindowEvent::Resized(physical_size) => {
                    log::info!("Resized to {:?}", physical_size);
                    context.resize(*physical_size);
                    camera.set_viewport(Viewport::new_at_origo(physical_size.width, physical_size.height));
                }
                winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    context.resize(**new_inner_size);
                    camera.set_viewport(Viewport::new_at_origo(new_inner_size.width, new_inner_size.height));
                }
                winit::event::WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                }
                _ => (),
            }
        }
        _ => {}
    });
}
