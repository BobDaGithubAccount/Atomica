use nalgebra::{DMatrix, Dynamic, MatrixMN, U1};
use std::f32::consts::E;
use crate::log;

const TOLERANCE: f32 = 1e-3;
const MAX_ITERATIONS: usize = 1000;

#[derive(Debug)]
pub struct DFTSolver {
    pub frames: Vec<Vec<f32>>,
    pub final_density: Vec<f32>,
    pub eigenvalues: Option<Vec<f32>>,
}

impl DFTSolver {
    pub fn new() -> Self {
        DFTSolver {
            frames: Vec::new(),
            final_density: Vec::new(),
            eigenvalues: None,
        }
    }

    pub fn build_grid(&self, points_per_axis: usize) -> Vec<[f32; 3]> {
        let mut grid = Vec::new();
        for x in 0..points_per_axis {
            for y in 0..points_per_axis {
                for z in 0..points_per_axis {
                    let xf = -10.0 + 20.0 * (x as f32 / (points_per_axis - 1) as f32);
                    let yf = -10.0 + 20.0 * (y as f32 / (points_per_axis - 1) as f32);
                    let zf = -10.0 + 20.0 * (z as f32 / (points_per_axis - 1) as f32);
                    grid.push([xf, yf, zf]);
                }
            }
        }
        grid
    }

    pub fn run_scf(
        &mut self,
        grid: &Vec<[f32; 3]>,
        centers: &Vec<[f32; 3]>,
        alphas: &Vec<f32>,
        external_potential: &dyn Fn(&[[f32; 3]]) -> Vec<f32>,
        num_electrons: usize,
    ) {
        let (dens, evals) = scf_loop(
            grid,
            centers,
            alphas,
            external_potential,
            num_electrons,
            &mut |d| {
                self.frames.push(d.to_vec());
            },
        );
        self.final_density = dens;
        self.eigenvalues = evals;
    }
}

/// Build the overlap matrix.
fn build_overlap_matrix(grid: &Vec<[f32; 3]>, centers: &Vec<[f32; 3]>, alphas: &Vec<f32>) -> DMatrix<f32> {
    let n_basis = centers.len();
    let n_grid = grid.len();
    let mut result = DMatrix::<f32>::zeros(n_basis, n_basis);

    for i in 0..n_basis {
        for j in 0..n_basis {
            let mut s = 0.0_f32;
            for k in 0..n_grid {
                let dx_i = grid[k][0] - centers[i][0];
                let dy_i = grid[k][1] - centers[i][1];
                let dz_i = grid[k][2] - centers[i][2];
                let g_i = (-alphas[i] * (dx_i * dx_i + dy_i * dy_i + dz_i * dz_i)).exp();

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

/// Update electron density.
fn update_density(
    grid: &Vec<[f32; 3]>,
    eigenvectors: &DMatrix<f32>,
    centers: &Vec<[f32; 3]>,
    alphas: &Vec<f32>,
    num_electrons: usize,
) -> Vec<f32> {
    let n_basis = centers.len();
    let n_grid = grid.len();
    let mut density = vec![0.0_f32; n_grid];

    for (gid, point) in grid.iter().enumerate() {
        let mut total_density = 0.0_f32;
        let n_occ = num_electrons.min(eigenvectors.ncols());
        for i in 0..n_occ {
            let mut wavefunction_i = 0.0_f32;
            for j in 0..n_basis {
                let dx = point[0] - centers[j][0];
                let dy = point[1] - centers[j][1];
                let dz = point[2] - centers[j][2];
                let g = (-alphas[j] * (dx * dx + dy * dy + dz * dz)).exp();
                wavefunction_i += eigenvectors[(j, i)] * g;
            }
            total_density += wavefunction_i * wavefunction_i;
        }
        density[gid] = total_density;
    }

    let sum_dens: f32 = density.iter().sum();
    if sum_dens > 0.0 {
        for d in density.iter_mut() {
            *d /= sum_dens;
        }
    }
    density
}

/// SCF loop.
pub fn scf_loop(
    grid: &Vec<[f32; 3]>,
    centers: &Vec<[f32; 3]>,
    alphas: &Vec<f32>,
    external_potential: &dyn Fn(&[[f32; 3]]) -> Vec<f32>,
    num_electrons: usize,
    frame_callback: &mut dyn FnMut(&[f32]),
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n_basis = centers.len();
    let n_grid = grid.len();
    let mut density = vec![0.1_f32; n_grid];
    let damping_factor = 0.2_f32;

    for iteration in 0..MAX_ITERATIONS {
        log(format!("Iteration {}: Starting SCF loop", iteration));
        
        let mut hamiltonian = build_overlap_matrix(grid, centers, alphas);
        for r in 0..n_basis {
            hamiltonian[(r, r)] += 1.0;
        }

        let eigen = hamiltonian.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues.as_slice().to_vec();
        let eigenvectors = eigen.eigenvectors;

        log(format!("Iteration {}: Eigenvalues calculated: {:?}", iteration, eigenvalues));

        let new_density = update_density(grid, &eigenvectors, centers, alphas, num_electrons);

        for i in 0..n_grid {
            density[i] = damping_factor * new_density[i] + (1.0 - damping_factor) * density[i];
        }

        frame_callback(&density);

        let diff_sum: f32 = new_density
            .iter()
            .zip(density.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        
        log(format!("Iteration {}: Density difference sum: {}", iteration, diff_sum));

        if diff_sum < TOLERANCE {
            log(format!("Iteration {}: Convergence achieved", iteration));
            return (density, Some(eigenvalues));
        }
    }
    log("Maximum iterations reached without convergence".to_string());
    (density, None)
}

pub fn run_scf_command(_args: Vec<String>) {
    log("Starting SCF simulation...".to_string());
    let mut solver = DFTSolver::new();
    let grid = solver.build_grid(10);
    let centers = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
    let alphas = vec![0.5, 0.5, 0.5];
    let external_potential = |grid: &[[f32; 3]]| -> Vec<f32> {
        grid.iter().map(|p| p[0] + p[1] + p[2]).collect()
    };
    solver.run_scf(&grid, &centers, &alphas, &external_potential, 2);
    log("SCF simulation finished".to_string());
}