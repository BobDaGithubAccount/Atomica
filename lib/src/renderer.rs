use crate::log;
use std::sync::Mutex;
use lazy_static::lazy_static;
use three_d::{renderer::*, FrameInputGenerator, SurfaceSettings, WindowedContext};

lazy_static! {
    pub static ref CAMERA_INSTANCE: Mutex<Option<Camera>> = Mutex::new(None);
}

// pub fn main() {
//     let event_loop = winit::event_loop::EventLoop::new();

//     #[cfg(not(target_arch = "wasm32"))]
//     let window_builder = winit::window::WindowBuilder::new()
//         .with_title("Full-Screen Window")
//         .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080))
//         .with_decorations(false)
//         .with_maximized(true);

//     #[cfg(target_arch = "wasm32")]
//     let window_builder = {
//         use wasm_bindgen::JsCast;
//         use winit::platform::web::WindowBuilderExtWebSys;
//         let window = web_sys::window().unwrap();
//         let document = window.document().unwrap();
//         let canvas = document
//             .get_element_by_id("canvas")
//             .unwrap()
//             .dyn_into::<web_sys::HtmlCanvasElement>()
//             .unwrap();
//         let width = canvas.client_width() as f64;
//         let height = canvas.client_height() as f64;

//         winit::window::WindowBuilder::new()
//             .with_canvas(Some(canvas))
//             .with_inner_size(winit::dpi::LogicalSize::new(width, height))
//             .with_prevent_default(true)
//     };

//     let window = window_builder.build(&event_loop).unwrap();
//     let context =
//         WindowedContext::from_winit_window(&window, SurfaceSettings::default()).unwrap();

//     let camera = Camera::new_perspective(
//         Viewport::new_at_origo(1, 1),
//         vec3(0.0, 2.0, 4.0),
//         vec3(0.0, 0.0, 0.0),
//         vec3(0.0, 1.0, 0.0),
//         degrees(45.0),
//         0.1,
//         100.0,
//     );

//     *CAMERA_INSTANCE.lock().unwrap() = Some(camera.clone());

//     let mut control = OrbitControl::new(*camera.target(), 1.0, 100.0);

//     let mut model = Gm::new(
//         Mesh::new(&context, &CpuMesh::cube()),
//         ColorMaterial {
//             color: Srgba::GREEN,
//             ..Default::default()
//         },
//     );
//     model.set_animation(|time| Mat4::from_angle_y(radians(time * 0.0005)));

//     let mut frame_input_generator = FrameInputGenerator::from_winit_window(&window);

//     event_loop.run(move |event, _, control_flow| {
//         match event {
//             winit::event::Event::MainEventsCleared => {
//                 window.request_redraw();
//             }
//             winit::event::Event::RedrawRequested(_) => {
//                 let mut frame_input = frame_input_generator.generate(&context);

//                 if let Some(camera) = CAMERA_INSTANCE.lock().unwrap().as_mut() {
//                     control.handle_events(camera, &mut frame_input.events);
//                     camera.set_viewport(frame_input.viewport);
//                     model.animate(frame_input.accumulated_time as f32);
//                     frame_input
//                         .screen()
//                         .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
//                         .render(camera, &model, &[]);
//                 }

//                 context.swap_buffers().unwrap();
//                 control_flow.set_poll();
//                 window.request_redraw();
//             }
//             winit::event::Event::WindowEvent { ref event, .. } => {
//                 frame_input_generator.handle_winit_window_event(event);

//                 match event {
//                     winit::event::WindowEvent::Resized(physical_size) => {
//                         log::info!("Resized to {:?}", physical_size);
//                         context.resize(*physical_size);
//                     }
//                     winit::event::WindowEvent::ScaleFactorChanged {
//                         new_inner_size, ..
//                     } => {
//                         context.resize(**new_inner_size);
//                     }
//                     winit::event::WindowEvent::CloseRequested => {
//                         control_flow.set_exit();
//                     }
//                     _ => (),
//                 }
//             }
//             _ => {}
//         }
//     });
// }

pub fn fetch_electron_density() -> Option<Vec<f32>> {
    return None;
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

    let mut frame_input_generator = FrameInputGenerator::from_winit_window(&window);

    event_loop.run(move |event, _, control_flow| {
        match event {
            winit::event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::RedrawRequested(_) => {

                let mesh_model = Gm::new(
                    Mesh::new(&context, &generate_mesh_from_density(fetch_electron_density())),
                    ColorMaterial {
                        color: Srgba::GREEN,
                        ..Default::default()
                    },
                );

                let mut frame_input = frame_input_generator.generate(&context);

                if let Some(camera) = CAMERA_INSTANCE.lock().unwrap().as_mut() {
                    control.handle_events(camera, &mut frame_input.events);
                    camera.set_viewport(frame_input.viewport);
                    frame_input
                        .screen()
                        .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
                        .render(camera, &mesh_model, &[]);
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

use marching_cubes::marching::polygonise;
use marching_cubes::marching::GridCell;
use marching_cubes::marching::Triangle;
use marching_cubes::marching::MarchingCubes;
//marching_cubes::marching::polygonise(grid_cell, isolevel, triangles);

pub fn generate_mesh_from_density(density_data: Option<Vec<f32>>) -> CpuMesh {
    match density_data {
        Some(density) => {
            let grid_size = 50;
            let isolevel = 0.5;
            let mut vertices = Vec::new();
            let mut indices = Vec::new();

            for z in 0..grid_size - 1 {
                for y in 0..grid_size - 1 {
                    for x in 0..grid_size - 1 {
                        let grid_cell = GridCell {
                            positions: [
                                [x as f32, y as f32, z as f32],
                                [(x + 1) as f32, y as f32, z as f32],
                                [(x + 1) as f32, (y + 1) as f32, z as f32],
                                [x as f32, (y + 1) as f32, z as f32],
                                [x as f32, y as f32, (z + 1) as f32],
                                [(x + 1) as f32, y as f32, (z + 1) as f32],
                                [(x + 1) as f32, (y + 1) as f32, (z + 1) as f32],
                                [x as f32, (y + 1) as f32, (z + 1) as f32],
                            ],
                            value: [
                                density[(x + y * grid_size + z * grid_size * grid_size) as usize],
                                density[((x + 1) + y * grid_size + z * grid_size * grid_size) as usize],
                                density[((x + 1) + (y + 1) * grid_size + z * grid_size * grid_size) as usize],
                                density[(x + (y + 1) * grid_size + z * grid_size * grid_size) as usize],
                                density[(x + y * grid_size + (z + 1) * grid_size * grid_size) as usize],
                                density[((x + 1) + y * grid_size + (z + 1) * grid_size * grid_size) as usize],
                                density[((x + 1) + (y + 1) * grid_size + (z + 1) * grid_size * grid_size) as usize],
                                density[(x + (y + 1) * grid_size + (z + 1) * grid_size * grid_size) as usize],
                            ],
                        };

                        let mut triangles = Vec::new();
                        let mc = MarchingCubes::new(isolevel, grid_cell);
                        let number_of_triangles = mc.polygonise(&mut triangles);
                        log(format!("Generated {} triangles", number_of_triangles));

                        for triangle in triangles {
                            let offset = vertices.len();
                            vertices.push(triangle.positions[0]);
                            vertices.push(triangle.positions[1]);
                            vertices.push(triangle.positions[2]);
                            indices.push(offset);
                            indices.push(offset + 1);
                            indices.push(offset + 2);
                        }
                    }
                }
            }

            CpuMesh {
                positions: Positions::F32(
                    vertices
                        .into_iter()
                        .map(|p| Vector3::new(p[0], p[1], p[2]))
                        .collect()
                ),
                indices: Indices::U32(indices.into_iter().map(|i| i as u32).collect()),
                ..Default::default()
            }
        }
        None => CpuMesh::cube(),
    }
}
