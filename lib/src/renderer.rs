// use crate::log;
use std::sync::Mutex;
use lazy_static::lazy_static;
use three_d::{renderer::*, FrameInputGenerator, SurfaceSettings, WindowedContext};
use log::info;
use crate::simulation::*;

lazy_static! {
    pub static ref CAMERA_INSTANCE: Mutex<Option<Camera>> = Mutex::new(None);
    pub static ref CASHED_CPU_MESH: Mutex<CpuMesh> = Mutex::new(CpuMesh::default());
    pub static ref FRAME_COUNTER: Mutex<u32> = Mutex::new(0);
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
        1000.0,
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
                {
                    let mut frame_count = FRAME_COUNTER.lock().unwrap();
                    *frame_count += 1;
                    if *frame_count % 60 == 0 {
                        let cpu_mesh = generate_mesh_from_density();
                        info!("Generated mesh!");
                        let mut cached = CASHED_CPU_MESH.try_lock().unwrap();
                        cached.clone_from(&cpu_mesh);
                    }
                }

                let mesh_model: Gm<Mesh, ColorMaterial> = Gm::new(
                    Mesh::new(&context, &CASHED_CPU_MESH.lock().unwrap()),
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
                    winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
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

// fn generate_mesh_from_density() -> CpuMesh {

//     if let Ok(state) = SIMULATION_STATE.try_lock() {
//         if state.dft_simulator.final_density.is_empty() {
//             return CpuMesh::default();
//         }

//         let density = state.dft_simulator.final_density.clone();
//         let grid_size = (state.dft_simulator.final_density.len() as f64)
//             .powf(1.0 / 3.0)
//             .round() as usize;
//         let iso_value = (*state.dft_simulator.final_density.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()) * 1.1;

//         let mut vertices = Vec::new();
//         let mut indices: Vec<u32> = Vec::new();
    
//         // Linear interpolation along an edge. (might want to improve this)
//         fn vertex_interp(p1: Vec3, p2: Vec3, valp1: f32, valp2: f32, iso_value: f32) -> Vec3 {
//             if (iso_value - valp1).abs() < 0.00001 {
//                 return p1;
//             }
//             if (iso_value - valp2).abs() < 0.00001 {
//                 return p2;
//             }
//             if (valp1 - valp2).abs() < 0.00001 {
//                 return p1;
//             }
//             let mu = (iso_value - valp1) / (valp2 - valp1);
//             p1 + (p2 - p1) * mu
//         }
    
//         // Loop over each cell in the grid.
//         for z in 0..(grid_size - 1) {
//             for y in 0..(grid_size - 1) {
//                 for x in 0..(grid_size - 1) {
//                     // Indices into the density array for the 8 corners.
//                     let cube_indices = [
//                         x     + y * grid_size + z * grid_size * grid_size,
//                         x + 1 + y * grid_size + z * grid_size * grid_size,
//                         x + 1 + (y + 1) * grid_size + z * grid_size * grid_size,
//                         x     + (y + 1) * grid_size + z * grid_size * grid_size,
//                         x     + y * grid_size + (z + 1) * grid_size * grid_size,
//                         x + 1 + y * grid_size + (z + 1) * grid_size * grid_size,
//                         x + 1 + (y + 1) * grid_size + (z + 1) * grid_size * grid_size,
//                         x     + (y + 1) * grid_size + (z + 1) * grid_size * grid_size,
//                     ];
    
//                     // 3D positions for the cell corners.
//                     let cube_positions = [
//                         vec3(x as f32, y as f32, z as f32),
//                         vec3((x + 1) as f32, y as f32, z as f32),
//                         vec3((x + 1) as f32, (y + 1) as f32, z as f32),
//                         vec3(x as f32, (y + 1) as f32, z as f32),
//                         vec3(x as f32, y as f32, (z + 1) as f32),
//                         vec3((x + 1) as f32, y as f32, (z + 1) as f32),
//                         vec3((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
//                         vec3(x as f32, (y + 1) as f32, (z + 1) as f32),
//                     ];
    
//                     // Get the density values at the corners.
//                     let cube_values = [
//                         density[cube_indices[0]],
//                         density[cube_indices[1]],
//                         density[cube_indices[2]],
//                         density[cube_indices[3]],
//                         density[cube_indices[4]],
//                         density[cube_indices[5]],
//                         density[cube_indices[6]],
//                         density[cube_indices[7]],
//                     ];
    
//                     // Determine the cube configuration.
//                     let mut cube_index = 0;
//                     if cube_values[0] < iso_value { cube_index |= 1; }
//                     if cube_values[1] < iso_value { cube_index |= 2; }
//                     if cube_values[2] < iso_value { cube_index |= 4; }
//                     if cube_values[3] < iso_value { cube_index |= 8; }
//                     if cube_values[4] < iso_value { cube_index |= 16; }
//                     if cube_values[5] < iso_value { cube_index |= 32; }
//                     if cube_values[6] < iso_value { cube_index |= 64; }
//                     if cube_values[7] < iso_value { cube_index |= 128; }
    
//                     // If there is no intersection, continue.
//                     if EDGE_TABLE[cube_index] == 0 {
//                         continue;
//                     }
    
//                     // Compute the intersection points.
//                     let mut vert_list: [Vec3; 12] = [vec3(0.0, 0.0, 0.0); 12];
//                     if (EDGE_TABLE[cube_index] & 1) != 0 {
//                         vert_list[0] = vertex_interp(cube_positions[0], cube_positions[1], cube_values[0], cube_values[1], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 2) != 0 {
//                         vert_list[1] = vertex_interp(cube_positions[1], cube_positions[2], cube_values[1], cube_values[2], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 4) != 0 {
//                         vert_list[2] = vertex_interp(cube_positions[2], cube_positions[3], cube_values[2], cube_values[3], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 8) != 0 {
//                         vert_list[3] = vertex_interp(cube_positions[3], cube_positions[0], cube_values[3], cube_values[0], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 16) != 0 {
//                         vert_list[4] = vertex_interp(cube_positions[4], cube_positions[5], cube_values[4], cube_values[5], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 32) != 0 {
//                         vert_list[5] = vertex_interp(cube_positions[5], cube_positions[6], cube_values[5], cube_values[6], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 64) != 0 {
//                         vert_list[6] = vertex_interp(cube_positions[6], cube_positions[7], cube_values[6], cube_values[7], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 128) != 0 {
//                         vert_list[7] = vertex_interp(cube_positions[7], cube_positions[4], cube_values[7], cube_values[4], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 256) != 0 {
//                         vert_list[8] = vertex_interp(cube_positions[0], cube_positions[4], cube_values[0], cube_values[4], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 512) != 0 {
//                         vert_list[9] = vertex_interp(cube_positions[1], cube_positions[5], cube_values[1], cube_values[5], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 1024) != 0 {
//                         vert_list[10] = vertex_interp(cube_positions[2], cube_positions[6], cube_values[2], cube_values[6], iso_value);
//                     }
//                     if (EDGE_TABLE[cube_index] & 2048) != 0 {
//                         vert_list[11] = vertex_interp(cube_positions[3], cube_positions[7], cube_values[3], cube_values[7], iso_value);
//                     }
    
//                     // Build the triangles for this cell.
//                     let mut i = 0;
//                     while TRI_TABLE[cube_index][i] != -1 {
//                         let index0 = TRI_TABLE[cube_index][i] as usize;
//                         let index1 = TRI_TABLE[cube_index][i + 1] as usize;
//                         let index2 = TRI_TABLE[cube_index][i + 2] as usize;
    
//                         let base_index = vertices.len() as u32;
//                         vertices.push(vert_list[index0]);
//                         vertices.push(vert_list[index1]);
//                         vertices.push(vert_list[index2]);
    
//                         indices.push(base_index);
//                         indices.push(base_index + 1);
//                         indices.push(base_index + 2);
    
//                         i += 3;
//                     }
//                 }
//             }
//         }
    
//         let vertex_vectors: Vec<Vector3<f32>> = vertices
//         .into_iter()
//         .map(|v| Vector3::new(v[0], v[1], v[2]))
//         .collect();
    
//         let positions = Positions::F32(vertex_vectors);
//         println!("Positions length: {}", positions.len());
    
//         let mesh = CpuMesh {
//             positions,
//             indices: Indices::U32(indices.into_iter().map(|i| i as u32).collect()),
//             ..Default::default()
//         };
//         println!("Mesh vertex count: {}", mesh.vertex_count());

//         return mesh;
//     }
//     else {
//         return CpuMesh::default();
//     }
// }

fn generate_mesh_from_density() -> CpuMesh {

    if let Ok(state) = SIMULATION_STATE.try_lock() {
        if state.dft_simulator.final_density.is_empty() {
            return CpuMesh::default();
        }

        let density = state.dft_simulator.final_density.clone();
        let grid_size = (density.len() as f64).powf(1.0 / 3.0).round() as usize;
        let iso_value = (*density.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()) * 1.1;

        let mut vertices = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        fn vertex_interp(p1: Vec3, p2: Vec3, valp1: f32, valp2: f32, iso_value: f32) -> Vec3 {
            if (iso_value - valp1).abs() < 0.00001 {
                return p1;
            }
            if (iso_value - valp2).abs() < 0.00001 {
                return p2;
            }
            if (valp1 - valp2).abs() < 0.00001 {
                return p1;
            }
            let mu = (iso_value - valp1) / (valp2 - valp1);
            p1 + (p2 - p1) * mu
        }

        let gsq = grid_size * grid_size;
        let grid_size_usize = grid_size as usize;

        for z in 0..(grid_size_usize - 1) {
            for y in 0..(grid_size_usize - 1) {
                for x in 0..(grid_size_usize - 1) {
                    let x = x as usize;
                    let y = y as usize;
                    let z = z as usize;

                    let cube_indices = [
                        x     + y * grid_size_usize + z * gsq,
                        x + 1 + y * grid_size_usize + z * gsq,
                        x + 1 + (y + 1) * grid_size_usize + z * gsq,
                        x     + (y + 1) * grid_size_usize + z * gsq,
                        x     + y * grid_size_usize + (z + 1) * gsq,
                        x + 1 + y * grid_size_usize + (z + 1) * gsq,
                        x + 1 + (y + 1) * grid_size_usize + (z + 1) * gsq,
                        x     + (y + 1) * grid_size_usize + (z + 1) * gsq,
                    ];

                    // Bounds check for safety (should not trigger, but helps debug)
                    if cube_indices.iter().any(|&i| i >= density.len()) {
                        continue;
                    }

                    let cube_positions = [
                        vec3(x as f32, y as f32, z as f32),
                        vec3((x + 1) as f32, y as f32, z as f32),
                        vec3((x + 1) as f32, (y + 1) as f32, z as f32),
                        vec3(x as f32, (y + 1) as f32, z as f32),
                        vec3(x as f32, y as f32, (z + 1) as f32),
                        vec3((x + 1) as f32, y as f32, (z + 1) as f32),
                        vec3((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                        vec3(x as f32, (y + 1) as f32, (z + 1) as f32),
                    ];

                    let cube_values = [
                        density[cube_indices[0]],
                        density[cube_indices[1]],
                        density[cube_indices[2]],
                        density[cube_indices[3]],
                        density[cube_indices[4]],
                        density[cube_indices[5]],
                        density[cube_indices[6]],
                        density[cube_indices[7]],
                    ];

                    let mut cube_index = 0;
                    if cube_values[0] < iso_value { cube_index |= 1; }
                    if cube_values[1] < iso_value { cube_index |= 2; }
                    if cube_values[2] < iso_value { cube_index |= 4; }
                    if cube_values[3] < iso_value { cube_index |= 8; }
                    if cube_values[4] < iso_value { cube_index |= 16; }
                    if cube_values[5] < iso_value { cube_index |= 32; }
                    if cube_values[6] < iso_value { cube_index |= 64; }
                    if cube_values[7] < iso_value { cube_index |= 128; }

                    if EDGE_TABLE[cube_index] == 0 {
                        continue;
                    }

                    let mut vert_list: [Vec3; 12] = [vec3(0.0, 0.0, 0.0); 12];
                    if (EDGE_TABLE[cube_index] & 1) != 0 {
                        vert_list[0] = vertex_interp(cube_positions[0], cube_positions[1], cube_values[0], cube_values[1], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 2) != 0 {
                        vert_list[1] = vertex_interp(cube_positions[1], cube_positions[2], cube_values[1], cube_values[2], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 4) != 0 {
                        vert_list[2] = vertex_interp(cube_positions[2], cube_positions[3], cube_values[2], cube_values[3], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 8) != 0 {
                        vert_list[3] = vertex_interp(cube_positions[3], cube_positions[0], cube_values[3], cube_values[0], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 16) != 0 {
                        vert_list[4] = vertex_interp(cube_positions[4], cube_positions[5], cube_values[4], cube_values[5], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 32) != 0 {
                        vert_list[5] = vertex_interp(cube_positions[5], cube_positions[6], cube_values[5], cube_values[6], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 64) != 0 {
                        vert_list[6] = vertex_interp(cube_positions[6], cube_positions[7], cube_values[6], cube_values[7], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 128) != 0 {
                        vert_list[7] = vertex_interp(cube_positions[7], cube_positions[4], cube_values[7], cube_values[4], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 256) != 0 {
                        vert_list[8] = vertex_interp(cube_positions[0], cube_positions[4], cube_values[0], cube_values[4], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 512) != 0 {
                        vert_list[9] = vertex_interp(cube_positions[1], cube_positions[5], cube_values[1], cube_values[5], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 1024) != 0 {
                        vert_list[10] = vertex_interp(cube_positions[2], cube_positions[6], cube_values[2], cube_values[6], iso_value);
                    }
                    if (EDGE_TABLE[cube_index] & 2048) != 0 {
                        vert_list[11] = vertex_interp(cube_positions[3], cube_positions[7], cube_values[3], cube_values[7], iso_value);
                    }

                    let mut i = 0;
                    while TRI_TABLE[cube_index][i] != -1 {
                        let index0 = TRI_TABLE[cube_index][i] as usize;
                        let index1 = TRI_TABLE[cube_index][i + 1] as usize;
                        let index2 = TRI_TABLE[cube_index][i + 2] as usize;

                        let base_index = vertices.len() as u32;
                        vertices.push(vert_list[index0]);
                        vertices.push(vert_list[index1]);
                        vertices.push(vert_list[index2]);

                        indices.push(base_index);
                        indices.push(base_index + 1);
                        indices.push(base_index + 2);

                        i += 3;
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
            positions,
            indices: Indices::U32(indices.into_iter().map(|i| i as u32).collect()),
            ..Default::default()
        };

        return mesh;
    } else {
        return CpuMesh::default();
    }
}

// Marching cubes lookup tables: SOURCE: https://github.com/TristanAntonsen/marching-cubes/blob/main/src/lib.rs //TODO ADD TO SOURCE TRACKER
const EDGE_TABLE: [i32; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

const TRI_TABLE: [[i8; 16]; 256] = [
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
];