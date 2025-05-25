use std::{collections::HashMap, sync::Mutex};
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use crate::dft_simulator::DFTSolver;

lazy_static! {
    pub static ref SIMULATION_STATE: Mutex<SimulationState> = Mutex::new(SimulationState::default());

    pub static ref SIMULATION_CONFIGS: Mutex<HashMap<String, SimulationConfig>> = {
        let mut m = HashMap::new();
        m.insert(
            "hydrogen".to_string(),
            SimulationConfig {
                nuclei: vec![Nucleus {
                    species: "H".into(),
                    atomic_number: 1,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 1,
                basis: vec![GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.0 }],
                points_per_axis: 32,
            },
        );
        m.insert(
            "helium".to_string(),
            SimulationConfig {
                nuclei: vec![Nucleus {
                    species: "He".into(),
                    atomic_number: 2,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 2,
                basis: vec![GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.5 }],
                points_per_axis: 32,
            },
        );
        m.insert(
            "h2_molecule".to_string(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus {
                        species: "H".into(),
                        atomic_number: 1,
                        coordinates: [-0.37, 0.0, 0.0],
                    },
                    Nucleus {
                        species: "H".into(),
                        atomic_number: 1,
                        coordinates: [0.37, 0.0, 0.0],
                    },
                ],
                num_electrons: 2,
                basis: vec![
                    GaussianBasis { center: [-0.37, 0.0, 0.0], alpha: 1.0 },
                    GaussianBasis { center: [0.37, 0.0, 0.0], alpha: 1.0 },
                ],
                points_per_axis: 32,
            },
        );
        m.insert(
            "h2_molecule_low_resolution".to_string(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus {
                        species: "H".into(),
                        atomic_number: 1,
                        coordinates: [-1.0, 0.0, 0.0],
                    },
                    Nucleus {
                        species: "H".into(),
                        atomic_number: 1,
                        coordinates: [1.0, 0.0, 0.0],
                    },
                ],
                num_electrons: 2,
                basis: vec![
                    GaussianBasis { center: [-0.37, 0.0, 0.0], alpha: 1.0 },
                    GaussianBasis { center: [0.37, 0.0, 0.0], alpha: 1.0 },
                ],
                points_per_axis: 10,
            },
        );
        //EXPERIMENTAL - this is what I'm targeting as a proof of concept for the DFT solver
        m.insert(
            "oxygen".to_string(),
            SimulationConfig {
            points_per_axis: 32,
            nuclei: vec![Nucleus {
                species: "O".into(),
                atomic_number: 8,
                coordinates: [0.0, 0.0, 0.0],
            }],
            num_electrons: 8,
            basis: vec![
                GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.0 },
                GaussianBasis { center: [ 0.5, 0.0, 0.0], alpha: 0.5 },
                GaussianBasis { center: [-0.5, 0.0, 0.0], alpha: 0.5 },
                GaussianBasis { center: [0.0,  0.5, 0.0], alpha: 0.5 },
                GaussianBasis { center: [0.0, -0.5, 0.0], alpha: 0.5 },
                GaussianBasis { center: [0.0, 0.0,  0.5], alpha: 0.5 },
                GaussianBasis { center: [0.0, 0.0, -0.5], alpha: 0.5 },
                
                // GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 130.70932 },
                // GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 23.808861 },
                // GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 6.4436083 },
                // // Polarization functions
                // GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.0 }, // d-type function
                // // Diffuse functions
                // GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 0.1687144 },
            ],
            },
        );
        m.insert(
            "oxygen_low_res".to_string(),
            SimulationConfig {
            points_per_axis: 16,
            nuclei: vec![Nucleus {
                species: "O".into(),
                atomic_number: 8,
                coordinates: [0.0, 0.0, 0.0],
            }],
            num_electrons: 8,
            basis: vec![
                GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.0 },
                GaussianBasis { center: [ 0.5, 0.0, 0.0], alpha: 0.5 },
                GaussianBasis { center: [-0.5, 0.0, 0.0], alpha: 0.5 },
                GaussianBasis { center: [0.0,  0.5, 0.0], alpha: 0.5 },
                GaussianBasis { center: [0.0, -0.5, 0.0], alpha: 0.5 },
                GaussianBasis { center: [0.0, 0.0,  0.5], alpha: 0.5 },
                GaussianBasis { center: [0.0, 0.0, -0.5], alpha: 0.5 },
            ],
            },
        );


        Mutex::new(m)
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nucleus {
    pub species: String,
    pub atomic_number: u32,
    pub coordinates: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub total_time: f64,                  // Total simulation time in a.u. (irrelevant for now)
    pub status: SimulationStatus,         // Status of the simulation
    pub dft_simulator: DFTSolver,         // DFT solver instance
    pub file_context: Option<String>,     // Can be written to as if it were a file (this can be handled separately) LATER
}

impl SimulationState {
    pub fn new(
        total_time: f64,
        dft_simulator: DFTSolver,
        status: SimulationStatus,
    ) -> Self {
        let state = SimulationState {
            total_time,
            status,
            dft_simulator,
            file_context: None,
        };

        {
            let mut global_state = SIMULATION_STATE.lock().unwrap();
            *global_state = state.clone();
        }

        state
    }

    pub fn save_to_file_context(&mut self) -> Result<(), serde_json::Error> {
        let serialized = serde_json::to_string(self)?;
        self.file_context = Some(serialized);
        Ok(())
    }

    pub fn load_from_file_context(file_context: String) -> Result<Self, serde_json::Error> {
        let state: SimulationState = serde_json::from_str(&file_context)?;
        Ok(state)
    }

    pub fn print_summary(&self) -> String {
        let summary = format!(
            "Simulation State:\n  Total Time: {:.3} a.u.\n  Status: {:?}\n",
            self.total_time,
            self.status,
        );
        summary
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianBasis {
    pub center: [f32; 3],
    pub alpha:  f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub points_per_axis: usize,
    pub nuclei: Vec<Nucleus>,
    pub num_electrons: usize,
    pub basis: Vec<GaussianBasis>,
}

impl SimulationConfig {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
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
            basis: vec![ GaussianBasis { center: [0.0,0.0,0.0], alpha: 1.0 } ],
        }
    }
}


