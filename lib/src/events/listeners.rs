use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Document, Event, KeyboardEvent, MouseEvent, Window};
use log::info;

#[wasm_bindgen]
pub fn initialize_event_listeners() {
    let document = web_sys::window().unwrap().document().unwrap();
    let window = web_sys::window().unwrap();

    register_mouse_listeners(&document);
    register_keyboard_listeners(&document);
    register_window_listeners(&window);
}

fn register_mouse_listeners(document: &Document) {
    let mouse_move_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        info!("Mouse moved: ({}, {})", event.client_x(), event.client_y());
    }) as Box<dyn FnMut(_)>);
    document
        .add_event_listener_with_callback("mousemove", mouse_move_closure.as_ref().unchecked_ref())
        .unwrap();
    mouse_move_closure.forget();

    let mouse_click_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        info!("Mouse clicked at: ({}, {})", event.client_x(), event.client_y());
    }) as Box<dyn FnMut(_)>);
    document
        .add_event_listener_with_callback("click", mouse_click_closure.as_ref().unchecked_ref())
        .unwrap();
    mouse_click_closure.forget();

    let mouse_down_closure = Closure::wrap(Box::new(move |event: MouseEvent| {
        info!("Mouse button down: {}", event.button());
    }) as Box<dyn FnMut(_)>);
    document
        .add_event_listener_with_callback("mousedown", mouse_down_closure.as_ref().unchecked_ref())
        .unwrap();
    mouse_down_closure.forget();
}

fn register_keyboard_listeners(document: &Document) {
    let keydown_closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
        info!("Key pressed: {}", event.key());
    }) as Box<dyn FnMut(_)>);
    document
        .add_event_listener_with_callback("keydown", keydown_closure.as_ref().unchecked_ref())
        .unwrap();
    keydown_closure.forget();

    let keyup_closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
        info!("Key released: {}", event.key());
    }) as Box<dyn FnMut(_)>);
    document
        .add_event_listener_with_callback("keyup", keyup_closure.as_ref().unchecked_ref())
        .unwrap();
    keyup_closure.forget();
}

fn register_window_listeners(window: &Window) {
    let resize_closure = Closure::wrap(Box::new(move |event: Event| {
        info!("Window resized");
    }) as Box<dyn FnMut(_)>);
    window
        .add_event_listener_with_callback("resize", resize_closure.as_ref().unchecked_ref())
        .unwrap();
    resize_closure.forget();
}
