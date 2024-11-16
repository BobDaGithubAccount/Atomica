use web_sys::{window, CanvasRenderingContext2d, HtmlCanvasElement};
use wasm_bindgen::prelude::*;

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
