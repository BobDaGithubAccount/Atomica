use std::{collections::HashMap, sync::Mutex};
use lazy_static::lazy_static;
use crate::dft_simulator::DFTSolver;

////////////////////////////////////////////////////////////////////////
/// Basis functions
////////////////////////////////////////////////////////////////////////

pub trait BasisFunction: Send + Sync {
    fn center(&self) -> [f32; 3];
    fn value(&self, point: &[f32; 3]) -> f32;
    fn clone_box(&self) -> Box<dyn BasisFunction>;
}

impl Clone for Box<dyn BasisFunction> {
    fn clone(&self) -> Box<dyn BasisFunction> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct GaussianBasis {
    pub center: [f32; 3],
    pub alpha:  f32,
}

impl BasisFunction for GaussianBasis {
    fn center(&self) -> [f32; 3] {
        self.center
    }
    fn value(&self, point: &[f32; 3]) -> f32 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        (-self.alpha * (dx * dx + dy * dy + dz * dz)).exp()
    }
    fn clone_box(&self) -> Box<dyn BasisFunction> {
        Box::new(self.clone())
    }
}

/// A Slater-type orbital:  ψ(r) ∝ r^(n−1) e^(−ζ r)
#[derive(Debug, Clone)]
pub struct SlaterBasis {
    pub center: [f32; 3],
    pub zeta:   f32,
    pub n:      usize,
}

impl BasisFunction for SlaterBasis {
    fn center(&self) -> [f32;3] { self.center }
    fn value(&self, point: &[f32;3]) -> f32 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let r = (dx*dx + dy*dy + dz*dz).sqrt();
        // r^(n-1) * exp(-ζ r)
        r.powi((self.n-1) as i32) * (-self.zeta * r).exp()
    }
    fn clone_box(&self) -> Box<dyn BasisFunction> {
        Box::new(self.clone())
    }
}

/// A Cartesian Gaussian of arbitrary angular momentum (lx,ly,lz).
#[derive(Debug, Clone)]
pub struct AngularGaussian {
    pub center: [f32;3],
    pub alpha:  f32,
    pub l:      (usize, usize, usize),
}

impl BasisFunction for AngularGaussian {
    fn center(&self) -> [f32;3] { self.center }
    fn value(&self, point: &[f32;3]) -> f32 {
        let (lx, ly, lz) = self.l;
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let r2 = dx*dx + dy*dy + dz*dz;
        let norm = (2.0 * self.alpha / std::f32::consts::PI).powf(0.75);
        norm * dx.powi(lx as i32) * dy.powi(ly as i32) * dz.powi(lz as i32) * (-self.alpha * r2).exp()
    }
    fn clone_box(&self) -> Box<dyn BasisFunction> {
        Box::new(self.clone())
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
                nuclei: vec![Nucleus { species: "H".into(), atomic_number: 1, coordinates: [0.0,0.0,0.0] }],
                num_electrons: 1,
                basis: vec![ Box::new(GaussianBasis { center: [0.0,0.0,0.0], alpha: 1.0 }) ],
                points_per_axis: 32,
            }
        );

        // Helium
        m.insert(
            "helium".into(),
            SimulationConfig {
                nuclei: vec![Nucleus { species: "He".into(), atomic_number: 2, coordinates: [0.0,0.0,0.0] }],
                num_electrons: 2,
                basis: vec![ Box::new(GaussianBasis { center: [0.0,0.0,0.0], alpha: 1.5 }) ],
                points_per_axis: 32,
            }
        );

        // H2 molecule
        m.insert(
            "h2_molecule".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.37,0.0,0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.37,0.0,0.0] },
                ],
                num_electrons: 2,
                basis: vec![
                    Box::new(GaussianBasis { center: [-0.37,0.0,0.0], alpha: 1.0 }),
                    Box::new(GaussianBasis { center: [ 0.37,0.0,0.0], alpha: 1.0 }),
                ],
                points_per_axis: 32,
            }
        );

        // H2 low-resolution
        m.insert(
            "h2_molecule_low_resolution".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.0,0.0,0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 1.0,0.0,0.0] },
                ],
                num_electrons: 2,
                basis: vec![
                    Box::new(GaussianBasis { center: [-0.37,0.0,0.0], alpha: 1.0 }),
                    Box::new(GaussianBasis { center: [ 0.37,0.0,0.0], alpha: 1.0 }),
                ],
                points_per_axis: 10,
            }
        );

        // Oxygen
        m.insert(
            "oxygen".into(),
            SimulationConfig {
                nuclei: vec![Nucleus { species: "O".into(), atomic_number: 8, coordinates: [0.0,0.0,0.0] }],
                num_electrons: 8,
                basis: vec![
                    Box::new(GaussianBasis { center: [0.0,0.0,0.0], alpha: 1.0 }),
                    Box::new(GaussianBasis { center: [0.5,0.0,0.0], alpha: 0.5 }),
                    Box::new(GaussianBasis { center: [-0.5,0.0,0.0], alpha: 0.5 }),
                    Box::new(GaussianBasis { center: [0.0,0.5,0.0], alpha: 0.5 }),
                    Box::new(GaussianBasis { center: [0.0,-0.5,0.0], alpha: 0.5 }),
                    Box::new(GaussianBasis { center: [0.0,0.0,0.5], alpha: 0.5 }),
                    Box::new(GaussianBasis { center: [0.0,0.0,-0.5], alpha: 0.5 }),
                ],
                points_per_axis: 32,
            }
        );

        m.insert("oxygen_low_res".into(), SimulationConfig {
            points_per_axis: 32,
            nuclei: vec![ Nucleus {
            species: "O".into(),
            atomic_number: 8,
            coordinates: [0.0,0.0,0.0],
            } ],
            num_electrons: 8,
            basis: vec![
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 130.070932 / 8.0, l: (0,0,0) }),
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 23.808861 / 8.0, l: (0,0,0) }),
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 6.4436083 / 8.0,  l: (0,0,0) }),

            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 1.1 / 8.0, l: (1,0,0) }), 
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 1.1 / 8.0, l: (0,1,0) }),
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 1.1 / 8.0, l: (0,0,1) }),

            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.1687144 / 8.0, l: (0,0,0) }),

            //optional
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.25 / 8.0, l: (2,0,0) }),
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.25 / 8.0, l: (0,2,0) }),
            Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.25 / 8.0, l: (0,0,2) }),
            ],
        });


        m.insert("oxygen_advanced".into(), SimulationConfig {
            points_per_axis: 64,
            nuclei: vec![ Nucleus {
                species: "O".into(),
                atomic_number: 8,
                coordinates: [0.0,0.0,0.0],
            } ],
            num_electrons: 8,
            basis: vec![
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 130.070932 / 2.0, l: (0,0,0) }),
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 23.808861 / 2.0, l: (0,0,0) }),
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 6.4436083 / 2.0,  l: (0,0,0) }),

                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 1.1 / 2.0, l: (1,0,0) }), 
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 1.1 / 2.0, l: (0,1,0) }),
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 1.1 / 2.0, l: (0,0,1) }),

                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.1687144 / 2.0, l: (0,0,0) }),

                //optional
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.25 / 2.0, l: (2,0,0) }),
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.25 / 2.0, l: (0,2,0) }),
                Box::new(AngularGaussian { center: [0.0,0.0,0.0], alpha: 0.25 / 2.0, l: (0,0,2) }),
            ],
        });

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
        let state = SimulationState { total_time, status, dft_simulator, file_context: None };
        *SIMULATION_STATE.lock().unwrap() = state.clone();
        state
    }
    pub fn print_summary(&self) -> String {
        format!("Simulation State:\n  Total Time: {:.3} a.u.\n  Status: {:?}\n", self.total_time, self.status)
    }
}

impl Default for SimulationState {
    fn default() -> Self {
        SimulationState { total_time: 1.0, status: SimulationStatus::Completed, dft_simulator: DFTSolver::new(), file_context: None }
    }
}

#[derive(Clone)]
pub struct SimulationConfig {
    pub points_per_axis: usize,
    pub nuclei: Vec<Nucleus>,
    pub num_electrons: usize,
    pub basis: Vec<Box<dyn BasisFunction>>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            points_per_axis: 32,
            nuclei: vec![Nucleus { species: "H".into(), atomic_number: 1, coordinates: [0.0,0.0,0.0] }],
            num_electrons: 1,
            basis: vec![ Box::new(GaussianBasis { center: [0.0,0.0,0.0], alpha: 1.0 }) ],
        }
    }
}
