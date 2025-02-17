use crate::log;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use crate::simulation::*;

const TOLERANCE: f32 = 1e-3;
const MAX_ITERATIONS: usize = 1000;

// Exchange–correlation constant for the electron gas (LDA, exchange only).
const EXCHANGE_CORRELATION_CONSTANT: f32 = -0.738558766;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DFTSolver {
    pub final_density: Vec<f32>,
    pub eigenvalues: Option<Vec<f32>>,
}

impl DFTSolver {
    pub fn new() -> Self {
        DFTSolver {
            final_density: Vec::new(),
            eigenvalues: None,
        }
    }

    pub fn build_grid(&self, points_per_axis: usize) -> Vec<[f32; 3]> {
        let mut grid = Vec::with_capacity(points_per_axis * points_per_axis * points_per_axis);
        for x in 0..points_per_axis {
            let xf = -10.0 + 20.0 * (x as f32 / (points_per_axis as f32 - 1.0));
            for y in 0..points_per_axis {
                let yf = -10.0 + 20.0 * (y as f32 / (points_per_axis as f32 - 1.0));
                for z in 0..points_per_axis {
                    let zf = -10.0 + 20.0 * (z as f32 / (points_per_axis as f32 - 1.0));
                    grid.push([xf, yf, zf]);
                }
            }
        }
        grid
    }

    /// Run the self-consistent field (SCF) simulation.
    /// This function calls the SCF loop and stores the final electron density and eigenvalues.
    pub fn run_scf(
        &mut self,
        grid: &[[f32; 3]],
        centers: &[[f32; 3]],
        alphas: &[f32],
        num_electrons: usize,
    ) {
        let (density, eigenvalues) = scf_loop(grid, centers, alphas, num_electrons);
        self.final_density = density;
        self.eigenvalues = eigenvalues;
    }
}

//TODO: integrate into simulation
fn build_overlap_matrix(
    grid: &[[f32; 3]],
    centers: &[[f32; 3]],
    alphas: &[f32],
) -> DMatrix<f32> {
    let n_basis = centers.len();
    let n_grid = grid.len();
    let mut result = DMatrix::<f32>::zeros(n_basis, n_basis);

    for i in 0..n_basis {
        for j in 0..n_basis {
            let mut s = 0.0_f32;
            for k in 0..n_grid {
                // Gaussian centered at center i.
                let dx_i = grid[k][0] - centers[i][0];
                let dy_i = grid[k][1] - centers[i][1];
                let dz_i = grid[k][2] - centers[i][2];
                let g_i = (-alphas[i] * (dx_i * dx_i + dy_i * dy_i + dz_i * dz_i)).exp();

                // Gaussian centered at center j.
                let dx_j = grid[k][0] - centers[j][0];
                let dy_j = grid[k][1] - centers[j][1];
                let dz_j = grid[k][2] - centers[j][2];
                let g_j = (-alphas[j] * (dx_j * dx_j + dy_j * dy_j + dz_j * dz_j)).exp();

                s += g_i * g_j;
            }
            result[(i, j)] = s;
        }
    }
    result
}

fn build_full_hamiltonian(
    grid: &[[f32; 3]],
    centers: &[[f32; 3]],       // basis function centers
    alphas: &[f32],             // Gaussian exponents for each basis function
    atomic_centers: &[[f32; 3]],  // positions of nuclei
    atomic_charges: &[f32],       // nuclear charges (Z) for each atom
    density: &[f32],            // electron density on each grid point
    ) -> DMatrix<f32> {
    let n_basis = centers.len();
    let n_grid = grid.len();
    
    let volume = grid.len() as f32;
    let d_v = volume / n_grid as f32; // integration weight per grid point
    log::info!("Volume: {}, dV: {}", volume, d_v);
    
    // Small epsilon to avoid singularities in the Coulomb potential.
    let epsilon = 1e-6_f32;

    let mut hamiltonian = DMatrix::<f32>::zeros(n_basis, n_basis);

    // Loop over each pair of basis functions.
    for i in 0..n_basis {
        for j in 0..n_basis {
            let mut hij = 0.0_f32;

            // Integrate over grid points.
            for (k, point) in grid.iter().enumerate() {
                // Evaluate basis function i
                let dx_i = point[0] - centers[i][0];
                let dy_i = point[1] - centers[i][1];
                let dz_i = point[2] - centers[i][2];
                let r2_i = dx_i * dx_i + dy_i * dy_i + dz_i * dz_i;
                let phi_i = (-alphas[i] * r2_i).exp();

                // Evaluate basis function j
                let dx_j = point[0] - centers[j][0];
                let dy_j = point[1] - centers[j][1];
                let dz_j = point[2] - centers[j][2];
                let r2_j = dx_j * dx_j + dy_j * dy_j + dz_j * dz_j;
                let phi_j = (-alphas[j] * r2_j).exp();

                // Analytic Laplacian of a Gaussian:
                let laplacian_phi_j = (4.0 * alphas[j] * alphas[j] * r2_j - 6.0 * alphas[j]) * phi_j;

                // Kinetic energy term
                let kinetic = -0.5 * phi_i * laplacian_phi_j;

                // Nuclear Coulomb potential
                let mut v_nuc = 0.0_f32;
                for (a, atom_center) in atomic_centers.iter().enumerate() {
                    let dx_a = point[0] - atom_center[0];
                    let dy_a = point[1] - atom_center[1];
                    let dz_a = point[2] - atom_center[2];
                    let r_a = (dx_a * dx_a + dy_a * dy_a + dz_a * dz_a)
                        .sqrt()
                        .max(epsilon);
                    v_nuc += -atomic_charges[a] / r_a;
                }

                // Exchange–correlation potential:
                // V_xc(r) = EXCHANGE_CORRELATION_CONSTANT * ρ(r)^(1/3)
                // This constant is for an ideal electron gas. Non LDA approaches are more accurate but more complex.
                let density_val = density[k].max(1e-12);
                let v_xc = EXCHANGE_CORRELATION_CONSTANT * density_val.powf(1.0 / 3.0);

                let v_total = v_nuc + v_xc;

                // Potential energy term
                let potential = phi_i * v_total * phi_j;

                // Sum contributions with the integration weight (scale for volume element).
                hij += (kinetic + potential) * d_v;
            }
            hamiltonian[(i, j)] = hij;
        }
    }

    hamiltonian
}

/// Update the electron density on the grid from the occupied eigenfunctions.
fn update_density(
    grid: &[[f32; 3]],
    eigenvectors: &DMatrix<f32>, // Expected shape: (n_basis, n_occ)
    centers: &[[f32; 3]],
    alphas: &[f32],
    num_electrons: usize,
) -> Vec<f32> {
    let n_basis = centers.len();
    let n_grid = grid.len();
    let n_occ = num_electrons.min(eigenvectors.ncols());
    let mut density = vec![0.0_f32; n_grid];

    for (gid, point) in grid.iter().enumerate() {
        let mut total = 0.0;
        for i in 0..n_occ {
            let mut psi = 0.0_f32;
            for j in 0..n_basis {
                let dx = point[0] - centers[j][0];
                let dy = point[1] - centers[j][1];
                let dz = point[2] - centers[j][2];
                let g = (-alphas[j] * (dx * dx + dy * dy + dz * dz)).exp();
                psi += eigenvectors[(j, i)] * g;
            }
            total += psi * psi;
        }
        density[gid] = total;
    }

    // Normalise the density so that the sum equals the number of electrons.
    let sum: f32 = density.iter().sum();
    if sum > 0.0 {
        for d in &mut density {
            *d /= sum;
        }
    }
    density
}

/// The self-consistent field (SCF) loop.
/// 1. Build the Hamiltonian,
/// 2. Diagonalize it,
/// 3. Update the electron density using the occupied eigenfunctions,
/// 4. Apply damping, and
/// 5. Check for convergence.
fn scf_loop(
    grid: &[[f32; 3]],
    centers: &[[f32; 3]],
    alphas: &[f32],
    num_electrons: usize,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n_grid = grid.len();
    let n_basis = centers.len();
    // Initial guess for the density.
    let mut density = vec![0.1_f32; n_grid];
    let damping_factor = 0.2_f32;
    let mut eigenvalues_result = None;

    //TODO make this configurable
    let atomic_centers: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
    let atomic_charges: Vec<f32> = vec![1.0];

    for iter in 0..MAX_ITERATIONS {
        log(format!("Iteration {}...", iter + 1));

        // Build the full Hamiltonian.
        let hamiltonian = build_full_hamiltonian(
            grid,
            centers,
            alphas,
            &atomic_centers,
            &atomic_charges,
            &density,
        );

        // Diagonalize the Hamiltonian (symmetric eigen–problem).
        let eigen = hamiltonian.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues.clone();
        let eigenvectors = eigen.eigenvectors;
        log(format!(
            "Eigenvalues (first {}): {:?}",
            num_electrons,
            &eigenvalues.as_slice()[0..num_electrons.min(eigenvalues.len())]
        ));

        // Update the electron density from the occupied states.
        let new_density = update_density(grid, &eigenvectors, centers, alphas, num_electrons);

        // Apply damping.
        for k in 0..n_grid {
            density[k] = damping_factor * new_density[k] + (1.0 - damping_factor) * density[k];
        }

        // Check convergence.
        let diff: f32 = new_density
            .iter()
            .zip(density.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        if diff < TOLERANCE {
            log(format!("Converged in {} iterations.", iter + 1));
            eigenvalues_result = Some(eigenvalues.as_slice().to_vec());
            return (density, eigenvalues_result);
        }
    }
    println!("SCF did not converge.");
    (density, None)
}

pub fn run_scf_command(args: Vec<String>) {

    if args.len() != 1 {
        log("Invalid number of arguments for run_scf_command".to_string());
        return;
    }

    let points_per_axis = args[0].parse::<usize>().unwrap();

    log("Starting SCF simulation...".to_string());

    let mut solver = DFTSolver::new();

    let grid = solver.build_grid(points_per_axis);
    log(format!("Grid size: {}", grid.len()));

    // Single Gaussian basis function centered at the origin.
    let centers: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
    let alphas: Vec<f32> = vec![1.0];

    let num_electrons = 1;

    log("Running SCF simulation...".to_string());
    solver.run_scf(&grid, &centers, &alphas, num_electrons);

    log("Final electron density:".to_string());
    let final_density_string = solver
        .final_density
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join(", ");
    log(final_density_string);

    SIMULATION_STATE.lock().unwrap().dft_simulator = solver;
}