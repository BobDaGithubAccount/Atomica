use ndarray::prelude::*;
use nalgebra::{DMatrix, SymmetricEigen};

use std::sync::Mutex;
use lazy_static::lazy_static;
use crate::simulation::{SimulationState, SIMULATION_STATE, SimulationStatus};
use crate::log;

const TOLERANCE: f64 = 1e-6;
const MAX_ITERATIONS: usize = 100;

// convert ndarray::Array2<f64> to a nalgebra::DMatrix<f64>
fn to_dmatrix(arr: &Array2<f64>) -> Result<DMatrix<f64>, &'static str> {
    if let Some(slice) = arr.as_slice() {
        Ok(DMatrix::from_row_slice(arr.nrows(), arr.ncols(), slice))
    } else {
        Err("Non-contiguous data in Array2")
    }
}

// Define a Gaussian-type orbital
fn gaussian_basis(x: f64, center: f64, alpha: f64) -> f64 {
    (-alpha * (x - center).powi(2)).exp()
}

// Build the overlap matrix
fn build_overlap_matrix(grid: &Array1<f64>, centers: &[f64], alphas: &[f64]) -> Array2<f64> {
    let n_basis = centers.len();
    let mut s = Array2::<f64>::zeros((n_basis, n_basis));
    for i in 0..n_basis {
        for j in 0..n_basis {
            let integral = grid
                .iter()
                .map(|&x| {
                    gaussian_basis(x, centers[i], alphas[i])
                        * gaussian_basis(x, centers[j], alphas[j])
                })
                .sum::<f64>();
            s[[i, j]] = integral;
        }
    }
    s
}

// LDA Exchange-Correlation Potential
fn lda_exchange_correlation(density: &Array1<f64>) -> Array1<f64> {
    density.mapv(|rho| {
        if rho > 0.0 {
            -0.738558766 * rho.powf(1.0 / 3.0)
        } else {
            0.0
        }
    })
}

fn build_hamiltonian(
    grid: &Array1<f64>,
    density: &Array1<f64>,
    centers: &[f64],
    alphas: &[f64],
    external_potential: &dyn Fn(f64) -> f64,
) -> Array2<f64> {
    let n_basis = centers.len();
    let xc_potential = lda_exchange_correlation(density);

    // Coulomb potential calculation (Hartree potential)
    let mut coulomb_potential = Array1::<f64>::zeros(grid.len());
    for i in 0..grid.len() {
        let mut potential_sum = 0.0;
        for j in 0..grid.len() {
            if i != j {
                let distance = (grid[i] - grid[j]).abs();
                // Simple 1 / |r - r'|
                potential_sum += density[j] / distance;
            }
        }
        coulomb_potential[i] = potential_sum;
    }

    // Build Hamiltonian: ***H[i,j] = ∫ φ_i(x) * [V_ext(x) + V_xc(x) + V_coulomb(x)] * φ_j(x) dx***
    let mut hamiltonian = Array2::<f64>::zeros((n_basis, n_basis));
    for i in 0..n_basis {
        for j in 0..n_basis {
            let integral = grid
                .iter()
                .enumerate()
                .map(|(idx, &x)| {
                    gaussian_basis(x, centers[i], alphas[i])
                        * gaussian_basis(x, centers[j], alphas[j])
                        * (external_potential(x)
                            + xc_potential[idx]
                            + coulomb_potential[idx])
                })
                .sum::<f64>();
            hamiltonian[[i, j]] = integral;
        }
    }

    hamiltonian
}

// SELF CONSISTENT FIELD (SCF) LOOP
fn scf_loop() -> Result<(), &'static str> {
    // Grid and basis set
    let grid: Array1<f64> = Array::linspace(-5.0, 5.0, 100);
    let centers = vec![-1.0, 1.0]; // Basis function centers
    let alphas = vec![1.0, 1.0];  // Basis function exponents

    // Initial density (guess)
    let mut density = Array1::<f64>::ones(grid.len()) * 0.1;

    // Example external potential
    let external_potential = |x: f64| -1.0 / (x.abs() + 0.1);

    // Build overlap matrix
    let s = build_overlap_matrix(&grid, &centers, &alphas);

    for iteration in 0..MAX_ITERATIONS {
        // Build Hamiltonian
        let hamiltonian: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = build_hamiltonian(&grid, &density, &centers, &alphas, &external_potential);

        // Convert to DMatrix for diagonalization
        let h_mat = to_dmatrix(&hamiltonian)?;
        // (Currently ignoring S in the diagonalization; a full generalized solve would require S^-1 * H)

        // Perform symmetric eigen-decomposition on H
        let eigen = SymmetricEigen::new(h_mat);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Calculate new density from the lowest-energy eigenvector
        let mut new_density = Array1::<f64>::zeros(grid.len());
        // Summation of squares of the coefficients in the lowest-energy column
        // (This is a simplified approach that doesn't integrate basis functions on the grid.)
        let lowest_column = eigenvectors.column(0);
        let sum_of_squares: f64 = lowest_column.iter().map(|&c| c.powi(2)).sum();
        for i in 0..grid.len() {
            // Using the same contribution at each grid point (placeholder approach)
            new_density[i] = sum_of_squares;
        }

        // Check for convergence
        let diff = (&new_density - &density).mapv(|x| x.abs()).sum();
        if diff < TOLERANCE {
            log(format!("SCF converged in {} iterations", iteration));
            log(format!("Final eigenvalues: {:?}", eigenvalues));
            log(format!("Final density: {:?}", new_density));

            {
                let mut state = SIMULATION_STATE.lock().unwrap();
                // Store the final density in the simulation state
                state.density_matrix = vec![new_density.to_vec()];
                state.status = SimulationStatus::Completed;
            }

            return Ok(());
        }

        density = new_density;
    }

    Err("SCF did not converge within the maximum number of iterations.")
}

pub fn run_scf_command(_args: Vec<String>) {
    log("Starting SCF simulation...".to_string());
    match scf_loop() {
        Ok(_) => log("SCF simulation completed successfully.".to_string()),
        Err(e) => log(format!("Error during SCF simulation: {}", e)),
    }
}