use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref SIMULATION_STATE: Mutex<SimulationState> = Mutex::new(SimulationState::default());
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SimulationStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Nucleus {
    pub species: String,
    pub atomic_number: u32,
    pub coordinates: [f64; 3],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimulationState {
    pub simulation_time: f64, //Current simulation time in atomic units in a.u.
    pub time_step: f64,       //Time-step size in a.u.
    pub total_time: f64,      //Total simulation time in a.u.
    pub atomic_coordinates: Vec<Nucleus>, //Atomic coordinates with species and atomic number
    pub grid_spacing: [f64; 3],           //Grid spacing in R^3
    pub bounds: [[f64; 2]; 3],            //Bounds of the simulation in R^3
    pub density_matrix: Vec<Vec<f64>>,    //Time-dependent density matrix
    pub status: SimulationStatus,         //Status of the simulation
    pub file_context: Option<String>,     //Can be written to as if it were a file (this can be handled separately)
}

impl SimulationState {
    pub fn new(
        time_step: f64,
        total_time: f64,
        nucleus_locations: Vec<Nucleus>,
        grid_spacing: [f64; 3],
        bounds: [[f64; 2]; 3],
        density_matrix: Vec<Vec<f64>>,
        status: SimulationStatus,
    ) -> Self {
        let state = SimulationState {
            simulation_time: 0.0,
            time_step,
            total_time,
            atomic_coordinates: nucleus_locations,
            grid_spacing,
            bounds,
            density_matrix,
            status,
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

    pub fn validate_coordinates(&self) -> bool {
        for nucleus in &self.atomic_coordinates {
            if nucleus.coordinates[0] < self.bounds[0][0]
                || nucleus.coordinates[0] > self.bounds[0][1]
                || nucleus.coordinates[1] < self.bounds[1][0]
                || nucleus.coordinates[1] > self.bounds[1][1]
                || nucleus.coordinates[2] < self.bounds[2][0]
                || nucleus.coordinates[2] > self.bounds[2][1]
            {
                return false;
            }
        }
        return true;
    }

    pub fn print_summary(&self) -> String {
        let summary = format!(
            "Simulation State:\n  Simulation Time: {:.3} / {:.3} a.u.\n  Time Step: {:.3} a.u.\n  Status: {:?}\n  Grid Spacing: ({:.3}, {:.3}, {:.3}) a.u.\n  Bounds: x({:.3}, {:.3}), y({:.3}, {:.3}), z({:.3}, {:.3}) a.u.",
            self.simulation_time, self.total_time,
            self.time_step,
            self.status,
            self.grid_spacing[0], self.grid_spacing[1], self.grid_spacing[2],
            self.bounds[0][0], self.bounds[0][1],
            self.bounds[1][0], self.bounds[1][1],
            self.bounds[2][0], self.bounds[2][1]
        );
        summary
    }
}

impl Default for SimulationState {
    fn default() -> Self {
        SimulationState {
            simulation_time: 0.0,
            time_step: 0.01,
            total_time: 1.0,
            atomic_coordinates: Vec::new(),
            grid_spacing: [1.0, 1.0, 1.0],
            bounds: [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
            density_matrix: Vec::new(),
            status: SimulationStatus::Running,
            file_context: None,
        }
    }
}

//TODO: In future replace this with a state which is suitable as a placeholder
//This is just so there is always a state to work with