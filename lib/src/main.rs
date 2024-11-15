use three_d::{context, renderer::*, FrameInputGenerator, SurfaceSettings, WindowedContext};
use std::sync::Mutex;
use winit::dpi::{PhysicalSize, LogicalSize};
use web_sys::{window, CanvasRenderingContext2d, Document, HtmlCanvasElement};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

lazy_static::lazy_static! {
    static ref CAMERA_INSTANCE: Mutex<Option<Camera>> = Mutex::new(None);
}

pub fn copy_to_visible_canvas() -> Result<(), JsValue> {
    let document = window().unwrap().document().unwrap();

    let resizable_container = document.get_element_by_id("resizable-container")
        .unwrap();

    let visible_canvas = document.get_element_by_id("visible-canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    let visible_ctx = visible_canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();

    let hidden_canvas = document.get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    let width = resizable_container.client_width();
    let height = resizable_container.client_height();

    visible_canvas.set_width(width as u32);
    visible_canvas.set_height(height as u32);

    visible_ctx.draw_image_with_html_canvas_element_and_dw_and_dh(&hidden_canvas, 0.0, 0.0, width as f64, height as f64)?;

    Ok(())
}

pub fn main() {
    let event_loop = winit::event_loop::EventLoop::new();

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
            .with_inner_size(LogicalSize::new(1920, 1080))
            .with_prevent_default(true)
    };

    let render_window = window_builder.build(&event_loop).unwrap();
    let context = WindowedContext::from_winit_window(&render_window, SurfaceSettings::default()).unwrap();

    let mut camera = Camera::new_perspective(
        Viewport::new_at_origo(1, 1), 
        vec3(0.0, 2.0, 4.0),// Camera position
        vec3(0.0, 0.0, 0.0),// Target position (where the camera is looking)
        vec3(0.0, 1.0, 0.0),// Up direction
        degrees(45.0),// Field of view
        0.1,// Near clipping plane
        100.0,// Far clipping plane
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

    let mut frame_input_generator = FrameInputGenerator::from_winit_window(&render_window);
    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::MainEventsCleared => {
            render_window.request_redraw();
        }
        winit::event::Event::RedrawRequested(_) => {
            let mut frame_input = frame_input_generator.generate(&context);
            copy_to_visible_canvas().unwrap();
            control.handle_events(&mut camera, &mut frame_input.events);
            camera.set_viewport(frame_input.viewport);
            model.animate(frame_input.accumulated_time as f32);
            frame_input
                .screen()
                .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
                .render(&camera, &model, &[]);
            context.swap_buffers().unwrap();
            control_flow.set_poll();
            render_window.request_redraw();

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
                    log::info!("Scale factor changed to {:?}", new_inner_size);
                    context.resize(**new_inner_size);
                    camera.set_viewport(Viewport::new_at_origo(new_inner_size.width, new_inner_size.height));
                }
                winit::event::WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                }
                _ =>
                    log::info!("Event: {:?}", event),
            }
        }
        _ => {
        }
    });
}