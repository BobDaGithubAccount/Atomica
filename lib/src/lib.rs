#![allow(special_module_name)]

pub mod commands;
use crate::commands::Command;

use wasm_bindgen::prelude::*;

use three_d::{renderer::*, FrameInputGenerator, SurfaceSettings, WindowedContext};
use web_sys::HtmlElement;
use std::sync::Mutex;
use lazy_static::lazy_static;
use log::info;

lazy_static! {
    static ref CAMERA_INSTANCE: Mutex<Option<Camera>> = Mutex::new(None);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    use commands::register_command;

    console_log::init_with_level(log::Level::Debug).unwrap();
    info!("Logging works!");
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    register_commands();
    main();
    Ok(())
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
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        let width = canvas.client_width() as f64;
        let height = canvas.client_height() as f64;

        winit::window::WindowBuilder::new()
            .with_canvas(Some(canvas))
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .with_prevent_default(true)
    };

    let window = window_builder.build(&event_loop).unwrap();
    let context =
        WindowedContext::from_winit_window(&window, SurfaceSettings::default()).unwrap();

    let camera = Camera::new_perspective(
        Viewport::new_at_origo(1, 1),
        vec3(0.0, 2.0, 4.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        100.0,
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

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::RedrawRequested(_) => {
                let mut frame_input = frame_input_generator.generate(&context);

                if let Some(camera) = CAMERA_INSTANCE.lock().unwrap().as_mut() {
                    control.handle_events(camera, &mut frame_input.events);
                    camera.set_viewport(frame_input.viewport);
                    model.animate(frame_input.accumulated_time as f32);
                    frame_input
                        .screen()
                        .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
                        .render(camera, &model, &[]);
                }

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
                    }
                    winit::event::WindowEvent::ScaleFactorChanged {
                        new_inner_size, ..
                    } => {
                        context.resize(**new_inner_size);
                    }
                    winit::event::WindowEvent::CloseRequested => {
                        control_flow.set_exit();
                    }
                    _ => (),
                }
            }
            _ => {}
        }
    });
}

#[wasm_bindgen]
pub fn log(message: String) {
    info!("{}", message);
    let document = web_sys::window().unwrap().document().unwrap();
    let log_area = document.get_element_by_id("log-area")
        .unwrap()
        .dyn_into::<HtmlElement>()
        .unwrap();
    let timestamp = js_sys::Date::new_0().to_locale_time_string("en-GB");
    let formatted_message = format!("[{}] {}", timestamp, message);
    log_area.set_inner_html(&format!("{}<br>{}", log_area.inner_html(), formatted_message));
    log_area.set_scroll_top(log_area.scroll_height());  // This scrolls to the bottom
}

pub fn register_commands() {
    register_command!("reset_camera", reset_camera_command);
    register_command!("fov", update_camera_fov_command);
}

fn reset_camera_command(args: Vec<String>) {
    if(args.len() != 0) {
        log(format!("Invalid number of arguments for reset_camera_command"));
        return;
    }
    log(format!("Resetting camera: {:?}", args));
    //TODO: Implement reset camera logic
}

fn update_camera_fov_command(args: Vec<String>) {
    if args.len() != 1 {
        log(format!("Invalid number of arguments for update_camera_fov_command"));
        return;
    }

    match args[0].parse::<f32>() {
        Ok(fov_degrees) if fov_degrees > 0.0 && fov_degrees < 180.0 => {
            if let Some(mut camera) = CAMERA_INSTANCE.lock().unwrap().as_mut() {
                let current_viewport = camera.viewport().clone();
                let current_position = *camera.position();
                let current_target = *camera.target();
                let current_up = *camera.up();
                let current_z_near = camera.z_near();
                let current_z_far = camera.z_far();

                *camera = Camera::new_perspective(
                    current_viewport,
                    current_position,
                    current_target,
                    current_up,
                    degrees(fov_degrees),
                    current_z_near,
                    current_z_far,
                );

                log(format!("Camera FOV updated to {} degrees", fov_degrees));
            } else {
                log(format!("Camera instance not initialized"));
            }
        }
        _ => {
            log(format!("Invalid FOV value: {:?}", args[0]));
        }
    }
}

#[wasm_bindgen]
pub fn handle_command(command_line: &str) {
    let parts: Vec<&str> = command_line.split_whitespace().collect();
    if parts.is_empty() {
        log(format!("No command entered"));
        return;
    }
    let command_name = parts[0];
    let args: Vec<String> = parts[1..].iter().map(|&s| s.to_string()).collect();

    let registry = commands::COMMAND_REGISTRY.lock().unwrap();
    if let Some(command) = registry.get(command_name) {
        (command.func)(args);
    } else {
        log(format!("Command not found: {}", command_name));
    }
}