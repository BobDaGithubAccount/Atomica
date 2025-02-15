use std::sync::Mutex;
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use crate::dft_simulator::DFTSolver;

lazy_static! {
    pub static ref SIMULATION_STATE: Mutex<SimulationState> = Mutex::new(SimulationState::default());
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
    pub file_context: Option<String>,     // Can be written to as if it were a file (this can be handled separately)
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

//TODO: Use these datastructures in the simulation code