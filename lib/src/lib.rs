#![allow(special_module_name)]

pub mod commands;
pub mod renderer;

use wasm_bindgen::prelude::*;

use web_sys::HtmlElement;
use log::info;
use three_d::renderer::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_log::init_with_level(log::Level::Debug).unwrap();
    info!("Logging works!");
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    register_commands();
    renderer::main();
    Ok(())
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
    register_command!("reset_camera", reset_camera_command, "Reset camera to default settings", "reset_camera");
    register_command!("fov", update_camera_fov_command, "Update camera field of view (FOV)", "fov [degrees]");
    commands::init();
}

fn reset_camera_command(args: Vec<String>) {
    if args.len() != 0 {
        log(format!("Invalid number of arguments for reset_camera_command"));
        return;
    }
    log(format!("Resetting camera to default settings"));
    if let Some(mut camera) = renderer::CAMERA_INSTANCE.lock().unwrap().as_mut() {
        let default_position = vec3(0.0, 2.0, 4.0);
        let default_target = vec3(0.0, 0.0, 0.0);
        let default_up = vec3(0.0, 1.0, 0.0);
        let default_fov = 45.0;
        let default_z_near = 0.1;
        let default_z_far = 100.0;
        *camera = Camera::new_perspective(
            camera.viewport().clone(),
            default_position,
            default_target,
            default_up,
            degrees(default_fov),
            default_z_near,
            default_z_far,
        );
        log(format!("Camera reset to default position: {:?}, target: {:?}, FOV: {} degrees",
            default_position, default_target, default_fov));
    } else {
        log(format!("Camera instance not initialized"));
    }
}

fn update_camera_fov_command(args: Vec<String>) {
    if args.len() != 1 {
        log(format!("Invalid number of arguments for update_camera_fov_command"));
        return;
    }

    match args[0].parse::<f32>() {
        Ok(fov_degrees) if fov_degrees > 0.0 && fov_degrees < 180.0 => {
            if let Some(mut camera) = renderer::CAMERA_INSTANCE.lock().unwrap().as_mut() {
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

    let registry_clone = {
        let registry = commands::COMMAND_REGISTRY.lock().unwrap();
        registry.clone()
    };

    if let Some(command) = registry_clone.get(command_name) {
        (command.func)(args);
    } else {
        log(format!("Command not found: {}", command_name));
    }
}