use std::{collections::HashMap, sync::Mutex, any::Any};
use lazy_static::lazy_static;
use crate::dft_simulator::DFTSolver;

////////////////////////////////////////////////////////////////////////
/// Basis functions
////////////////////////////////////////////////////////////////////////

pub trait BasisFunction: Send + Sync {
    fn center(&self) -> [f32; 3];
    fn value(&self, point: &[f32; 3]) -> f32;
    fn clone_box(&self) -> Box<dyn BasisFunction>;
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn BasisFunction> {
    fn clone(&self) -> Box<dyn BasisFunction> {
        self.clone_box()
    }
}

/// A Cartesian Gaussian of arbitrary angular momentum (ℓₓ,ℓᵧ,ℓ𝓏).
#[derive(Debug, Clone)]
pub struct AngularGaussian {
    pub center: [f32; 3],
    pub alpha: f32,
    pub l: (usize, usize, usize),
}

impl BasisFunction for AngularGaussian {
    fn center(&self) -> [f32; 3] {
        self.center
    }

    fn value(&self, point: &[f32; 3]) -> f32 {
        let (lx, ly, lz) = self.l;
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;

        let alpha = self.alpha;
        let lsum = (lx + ly + lz) as f32;

        // avoid underflow by treating lx==0 specially
        let df_x_arg = if lx == 0 { 0 } else { 2 * lx - 1 };
        let df_y_arg = if ly == 0 { 0 } else { 2 * ly - 1 };
        let df_z_arg = if lz == 0 { 0 } else { 2 * lz - 1 };

        let norm_prefactor = (2.0 * alpha / std::f32::consts::PI).powf(0.75);
        let poly_prefactor = (4.0 * alpha).powf(lsum / 2.0);

        let norm_denom = (
            double_factorial(df_x_arg) *
            double_factorial(df_y_arg) *
            double_factorial(df_z_arg)
        ).sqrt();

        let norm = norm_prefactor * poly_prefactor / norm_denom;

        norm
            * dx.powi(lx as i32)
            * dy.powi(ly as i32)
            * dz.powi(lz as i32)
            * (-alpha * r2).exp()
    }
    fn clone_box(&self) -> Box<dyn BasisFunction> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

}

fn double_factorial(n: usize) -> f32 {
    if n == 0 || n == 1 {
        1.0
    } else {
        (n as f32) * double_factorial(n - 2)
    }
}

////////////////////////////////////////////////////////////////////////

lazy_static! {
    pub static ref SIMULATION_STATE: Mutex<SimulationState> = Mutex::new(SimulationState::default());

    pub static ref SIMULATION_CONFIGS: Mutex<HashMap<String, SimulationConfig>> = {
        let mut m = HashMap::new();

        // Hydrogen (Gaussian)
        m.insert(
            "hydrogen".into(),
            SimulationConfig {
                nuclei: vec![Nucleus { species: "H".into(), atomic_number: 1, coordinates: [0.0, 0.0, 0.0] }],
                num_electrons: 1,
                basis: vec![Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.0, l: (0, 0, 0) })],
                points_per_axis: 32,
                tolerance: 1e-10
            },
        );

        // Helium
        m.insert(
            "helium".into(),
            SimulationConfig {
                nuclei: vec![Nucleus { species: "He".into(), atomic_number: 2, coordinates: [0.0, 0.0, 0.0] }],
                num_electrons: 2,
                basis: vec![Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.0, l: (0, 0, 0)})],
                points_per_axis: 32,
                tolerance: 1e-10
            },
        );

        // H2 molecule
        m.insert(
            "h2_molecule".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.0, 0.0, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [1.0, 0.0, 0.0] },
                ],
                num_electrons: 2,
                basis: vec![
                    Box::new(AngularGaussian { center: [-1.0, 0.0, 0.0], alpha: 1.0, l: (0, 0, 0)}),
                    Box::new(AngularGaussian { center: [1.0, 0.0, 0.0], alpha: 1.0, l: (0, 0, 0)}),
                ],
                points_per_axis: 32,
                tolerance: 1e-10
            },
        );

        // H2 low-resolution
        m.insert(
            "h2_molecule_low_resolution".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.0, 0.0, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [1.0, 0.0, 0.0] },
                ],
                num_electrons: 2,
                basis: vec![
                    Box::new(AngularGaussian { center: [-0.37, 0.0, 0.0], alpha: 1.0, l: (0, 0, 0)}),
                    Box::new(AngularGaussian { center: [0.37, 0.0, 0.0], alpha: 1.0, l: (0, 0, 0)}),
                ],
                points_per_axis: 10,
                tolerance: 1e-10
            },
        );

        m.insert(
            "oxygen".into(),
            SimulationConfig {
                points_per_axis: 48,
                nuclei: vec![Nucleus {
                    species: "O".into(),
                    atomic_number: 8,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 8,
                basis: vec![
                    // Core 1s STO-3G-like
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 130.07, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 23.81, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 6.44, l: (0, 0, 0) }),

                    // 2s valence
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.1687, l: (0, 0, 0) }),

                    // 2p valence (px, py, pz)
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1, l: (1, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1, l: (0, 1, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1, l: (0, 0, 1) }),

                    // Optional diffuse d-functions
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25, l: (2, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25, l: (0, 2, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25, l: (0, 0, 2) }),
                ],
                tolerance: 1e-1
            },
        );
        m.insert(
            "oxygen_low_res".into(),
            SimulationConfig {
                points_per_axis: 20,
                nuclei: vec![Nucleus {
                    species: "O".into(),
                    atomic_number: 8,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 8,
                basis: vec![
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 130.070932 / 12.0, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 23.808861 / 12.0, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 6.4436083 / 12.0, l: (0, 0, 0) }),

                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1 / 12.0, l: (1, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1 / 12.0, l: (0, 1, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1 / 12.0, l: (0, 0, 1) }),

                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.1687144 / 12.0, l: (0, 0, 0) }),

                    // optional d‐functions
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25 / 12.0, l: (2, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25 / 12.0, l: (0, 2, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25 / 12.0, l: (0, 0, 2) }),
                ],
                tolerance: 1e-1
            },
        );

        m.insert(
            "oxygen_minimal".into(),
            SimulationConfig {
                nuclei: vec![Nucleus {
                    species: "O".into(),
                    atomic_number: 8,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 8,
                points_per_axis: 40,
                basis: vec![
                    // 1s core shell (3 Gaussian primitives)
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 130.70932, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 23.808861, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 6.4436083, l: (0, 0, 0) }),

                    // 2s valence shell
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.502, l: (0, 0, 0) }),

                    // 2p orbitals
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.502, l: (1, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.502, l: (0, 1, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.502, l: (0, 0, 1) }),
                ],
                tolerance: 1e-1
            },
        );

        Mutex::new(m)
    };
}

#[derive(Debug, Clone)]
pub enum SimulationStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Nucleus {
    pub species: String,
    pub atomic_number: u32,
    pub coordinates: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct SimulationState {
    pub total_time: f64,
    pub status: SimulationStatus,
    pub dft_simulator: DFTSolver,
    pub file_context: Option<String>,
}

impl SimulationState {
    pub fn new(total_time: f64, dft_simulator: DFTSolver, status: SimulationStatus) -> Self {
        let state = SimulationState {
            total_time,
            status,
            dft_simulator,
            file_context: None,
        };
        *SIMULATION_STATE.lock().unwrap() = state.clone();
        state
    }
    pub fn print_summary(&self) -> String {
        format!(
            "Simulation State:\n  Total Time: {:.3} a.u.\n  Status: {:?}\n",
            self.total_time, self.status
        )
    }
}

impl Default for SimulationState {
    fn default() -> Self {
        SimulationState {
            total_time: 1.0,
            status: SimulationStatus::Completed,
            dft_simulator: DFTSolver::new(),
            file_context: None,
        }
    }
}

#[derive(Clone)]
pub struct SimulationConfig {
    pub points_per_axis: usize,
    pub nuclei: Vec<Nucleus>,
    pub num_electrons: usize,
    pub basis: Vec<Box<dyn BasisFunction>>,
    pub tolerance: f32,

}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            points_per_axis: 32,
            nuclei: vec![Nucleus {
                species: "H".into(),
                atomic_number: 1,
                coordinates: [0.0, 0.0, 0.0],
            }],
            num_electrons: 1,
            basis: vec![Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.0, l: (0,0,0)})],
            tolerance: 1e-10
        }
    }
}