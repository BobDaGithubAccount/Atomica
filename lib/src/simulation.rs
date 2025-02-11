use crate::log;
use std::sync::Mutex;
use lazy_static::lazy_static;
use three_d::{renderer::*, FrameInputGenerator, SurfaceSettings, WindowedContext};
use log::info;
use web_sys::console::info;
use crate::simulation::*;

lazy_static! {
    pub static ref CAMERA_INSTANCE: Mutex<Option<Camera>> = Mutex::new(None);
    pub static ref CASHED_CPU_MESH: Mutex<CpuMesh> = Mutex::new(CpuMesh::default());
    pub static ref FRAME_COUNTER: Mutex<u32> = Mutex::new(0);
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
                
                let cpu_mesh = generate_mesh_from_density();
                
                CASHED_CPU_MESH.try_lock().unwrap().clone_from(&cpu_mesh);

                let mesh_model = Gm::new(
                    Mesh::new(&context, &generate_mesh_from_density()),
                    ColorMaterial {
                        color: Srgba::GREEN,
                        ..Default::default()
                    },
                );

                let mut frame_input: three_d::FrameInput = frame_input_generator.generate(&context);

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

use marching_cubes::marching::{GridCell, MarchingCubes, Triangle};
use marching_cubes::container::{Vec3, vector3}; // Use vector3 for consistent Vec3 initialization

pub fn generate_mesh_from_density() -> CpuMesh {
    if let Ok(state) = SIMULATION_STATE.try_lock() {
        if state.dft_simulator.final_density.is_empty() {
            return CpuMesh::default();
        }

        let mut frame_counter = FRAME_COUNTER.lock().unwrap();
        *frame_counter = frame_counter.wrapping_add(1);
        if *frame_counter % 50 != 0 {
            return CASHED_CPU_MESH.lock().unwrap().clone();
        }

        let grid_size = (state.dft_simulator.final_density.len() as f64).powf(1.0 / 3.0) as usize;
        let mean = state.dft_simulator.final_density.iter().sum::<f32>() / state.dft_simulator.final_density.len() as f32;
        let std_dev = (state.dft_simulator.final_density.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / state.dft_simulator.final_density.len() as f32).sqrt();
        let isolevel = mean + 2.0 * std_dev;
        
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut triangles = vec![
            Triangle {
                positions: [
                    vector3(0.0, 0.0, 0.0),
                    vector3(0.0, 0.0, 0.0),
                    vector3(0.0, 0.0, 0.0),
                ]
            }; 5
        ];

        let grid_points = state.dft_simulator.build_grid(grid_size);
        let density = &state.dft_simulator.final_density;

        for x in 0..(grid_size - 1) {
            for y in 0..(grid_size - 1) {
                for z in 0..(grid_size - 1) {
                    let index = |xi, yi, zi| zi + grid_size * (yi + grid_size * xi);
                    let cube = GridCell {
                        positions: [
                            grid_points[index(x, y, z)],
                            grid_points[index(x + 1, y, z)],
                            grid_points[index(x + 1, y + 1, z)],
                            grid_points[index(x, y + 1, z)],
                            grid_points[index(x, y, z + 1)],
                            grid_points[index(x + 1, y, z + 1)],
                            grid_points[index(x + 1, y + 1, z + 1)],
                            grid_points[index(x, y + 1, z + 1)],
                        ],
                        value: [
                            density[index(x, y, z)],
                            density[index(x + 1, y, z)],
                            density[index(x + 1, y + 1, z)],
                            density[index(x, y + 1, z)],
                            density[index(x, y, z + 1)],
                            density[index(x + 1, y, z + 1)],
                            density[index(x + 1, y + 1, z + 1)],
                            density[index(x, y + 1, z + 1)],
                        ],
                    };
                    
                    let marching_cubes = MarchingCubes::new(isolevel, cube);
                    info!("Number of triangles: {}", triangles.len());
                    let num_triangles = marching_cubes.polygonise(&mut triangles);

                    for t in &triangles[..num_triangles as usize] {
                        let start_index = vertices.len() as u32;
                        
                        vertices.extend_from_slice(&[
                            vector3(t.positions[0][0], t.positions[0][1], t.positions[0][2]),
                            vector3(t.positions[1][0], t.positions[1][1], t.positions[1][2]),
                            vector3(t.positions[2][0], t.positions[2][1], t.positions[2][2]),
                        ]);
                        indices.extend_from_slice(&[start_index, start_index + 1, start_index + 2]);
                    }
                }
            }
        }
        
        let vertex_vectors: Vec<Vector3<f32>> = vertices
        .into_iter()
        .map(|v| Vector3::new(v[0], v[1], v[2]))
        .collect();

        let positions = Positions::F32(vertex_vectors);

        let mesh = CpuMesh {
            positions: positions,
            indices: Indices::U32(indices.into_iter().map(|i| i as u32).collect()),
            ..Default::default()
        };
        
        *CASHED_CPU_MESH.lock().unwrap() = mesh.clone();
        mesh
    } else {
        CpuMesh::default()
    }
}










//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

// use marching_cubes::marching::GridCell;
// use marching_cubes::marching::MarchingCubes;
// use marching_cubes::marching::Triangle;
// use marching_cubes::container::Vec3;

// pub fn generate_mesh_from_density() -> CpuMesh {
//     if let Ok(state) = SIMULATION_STATE.try_lock() {
//         if state.dft_simulator.final_density.is_empty() {
//             return CpuMesh::default();
//         }

//         let mut frame_counter = FRAME_COUNTER.lock().unwrap();
//         *frame_counter = frame_counter.wrapping_add(1);
//         if *frame_counter % 50 != 0 {
//             return CASHED_CPU_MESH.lock().unwrap().clone();
//         }

//         let grid_size = (state.dft_simulator.final_density.len() as f64).powf(1.0 / 3.0) as usize;
//         let mean = state.dft_simulator.final_density.iter().sum::<f32>() / state.dft_simulator.final_density.len() as f32;
//         let std_dev = (state.dft_simulator.final_density.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / state.dft_simulator.final_density.len() as f32).sqrt();
//         let isolevel = mean + 2.0 * std_dev;
        
//         let mut vertices = Vec::new();
//         let mut indices = Vec::new();
//         let mut triangles = vec![
//             Triangle {
//                 positions: [
//                     [0.0, 0.0, 0.0],
//                     [0.0, 0.0, 0.0],
//                     [0.0, 0.0, 0.0],
//                 ]
//             }; 5
//         ];

//         let grid_points = state.dft_simulator.build_grid(grid_size);
//         let density = &state.dft_simulator.final_density;

//         for x in 0..(grid_size - 1) {
//             for y in 0..(grid_size - 1) {
//                 for z in 0..(grid_size - 1) {
//                     let index = |xi, yi, zi| zi + grid_size * (yi + grid_size * xi);
//                     let cube = GridCell {
//                         positions: [
//                             grid_points[index(x, y, z)],
//                             grid_points[index(x + 1, y, z)],
//                             grid_points[index(x + 1, y + 1, z)],
//                             grid_points[index(x, y + 1, z)],
//                             grid_points[index(x, y, z + 1)],
//                             grid_points[index(x + 1, y, z + 1)],
//                             grid_points[index(x + 1, y + 1, z + 1)],
//                             grid_points[index(x, y + 1, z + 1)],
//                         ],
//                         value: [
//                             density[index(x, y, z)],
//                             density[index(x + 1, y, z)],
//                             density[index(x + 1, y + 1, z)],
//                             density[index(x, y + 1, z)],
//                             density[index(x, y, z + 1)],
//                             density[index(x + 1, y, z + 1)],
//                             density[index(x + 1, y + 1, z + 1)],
//                             density[index(x, y + 1, z + 1)],
//                         ],
//                     };
                    
//                     let marching_cubes = MarchingCubes::new(isolevel, cube);
//                     let num_triangles = marching_cubes.polygonise(&mut triangles);

//                     for t in &triangles[..num_triangles as usize] {
//                         let start_index = vertices.len() as u32;
//                         vertices.extend_from_slice(&[t.positions[0], t.positions[1], t.positions[2]]);
//                         indices.extend_from_slice(&[start_index, start_index + 1, start_index + 2]);
//                     }
//                 }
//             }
//         }
        
//         let mesh = CpuMesh {
//             positions: Positions::F32(
//                 vertices.into_iter().map(|p| p).collect()
//             ),
//             indices: Indices::U32(indices.into_iter().map(|i| i as u32).collect()),
//             ..Default::default()
//         };
        
//         *CASHED_CPU_MESH.lock().unwrap() = mesh.clone();
//         mesh
//     } else {
//         CpuMesh::default()
//     }
// }


//////////////////////////////////////////////////////////////////////////////////////
// pub fn generate_mesh_from_density() -> CpuMesh {

//     if let Ok(state) = SIMULATION_STATE.try_lock() {
//         if(state.dft_simulator.final_density.len() == 0) {
//             return CpuMesh::default();
//         }
//         let mut frame_counter = FRAME_COUNTER.lock().unwrap();
//         *frame_counter = frame_counter.wrapping_add(1);
//         if *frame_counter % 50 != 0 {
//             return CASHED_CPU_MESH.try_lock().unwrap().clone();
//         }
//         let grid_size = (state.dft_simulator.final_density.len() as f64).powf(1.0 / 3.0) as usize;
//         let mean = state.dft_simulator.final_density.iter().sum::<f32>() / state.dft_simulator.final_density.len() as f32;
//         let std_dev = state.dft_simulator.final_density.iter().map(|x| (x - mean).powi(2)).sum::<f32>().sqrt();
//         let isolevel = mean + 2.0 * std_dev;
//         let mut vertices = Vec::new();
//         let mut indices = Vec::new();

//         for z in 0..grid_size - 1 {
//             for y in 0..grid_size - 1 {
//                 for x in 0..grid_size - 1 {
//                     let grid_cell = GridCell {
//                         positions: [
//                             [x as f32, y as f32, z as f32],
//                             [(x + 1) as f32, y as f32, z as f32],
//                             [(x + 1) as f32, (y + 1) as f32, z as f32],
//                             [x as f32, (y + 1) as f32, z as f32],
//                             [x as f32, y as f32, (z + 1) as f32],
//                             [(x + 1) as f32, y as f32, (z + 1) as f32],
//                             [(x + 1) as f32, (y + 1) as f32, (z + 1) as f32],
//                             [x as f32, (y + 1) as f32, (z + 1) as f32],
//                         ],
//                         value: [
//                             state.dft_simulator.final_density[(x + y * grid_size + z * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[((x + 1) + y * grid_size + z * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[((x + 1) + (y + 1) * grid_size + z * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[(x + (y + 1) * grid_size + z * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[(x + y * grid_size + (z + 1) * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[((x + 1) + y * grid_size + (z + 1) * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[((x + 1) + (y + 1) * grid_size + (z + 1) * grid_size * grid_size) as usize],
//                             state.dft_simulator.final_density[(x + (y + 1) * grid_size + (z + 1) * grid_size * grid_size) as usize],
//                         ],
//                     };

//                     let mut triangles = Vec::new();
//                     let mc = MarchingCubes::new(isolevel, grid_cell);
//                     let number_of_triangles = mc.polygonise(&mut triangles);
//                     let min_val = state.dft_simulator.final_density.iter().cloned().fold(f32::INFINITY, f32::min);
//                     let max_val = state.dft_simulator.final_density.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
//                     info!("Final density length: {} Implicit Grid Size: {} Number of triangles: {}, Minimum value: {}, Maximum value: {}" , state.dft_simulator.final_density.len(), grid_size, number_of_triangles, min_val, max_val);

//                     for triangle in triangles {
//                         let offset = vertices.len();
//                         vertices.push(triangle.positions[0]);
//                         vertices.push(triangle.positions[1]);
//                         vertices.push(triangle.positions[2]);
//                         indices.push(offset);
//                         indices.push(offset + 1);
//                         indices.push(offset + 2);
//                     }
//                 }
//             }
//         }

//         CpuMesh {
//             positions: Positions::F32(
//                 vertices
//                     .into_iter()
//                     .map(|p| Vector3::new(p[0], p[1], p[2]))
//                     .collect()
//             ),
//             indices: Indices::U32(indices.into_iter().map(|i| i as u32).collect()),
//             ..Default::default()
//         }
        
//     }
//     else {
//         CpuMesh::default()
//     }
// }
