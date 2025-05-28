use crate::log;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use crate::simulation::*;
use rustfft::{num_complex::Complex, FftPlanner};

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
    // pub fn run_scf(
    //     &mut self,
    //     grid: &[[f32; 3]],
    //     centers: &[[f32; 3]],
    //     alphas: &[f32],
    //     num_electrons: usize,
    // ) {
    //     let (density, eigenvalues) = scf_loop(grid, centers, alphas, num_electrons);
    //     self.final_density = density;
    //     self.eigenvalues = eigenvalues;
    // }
    pub fn run_scf(&mut self, config: &SimulationConfig) {
        let grid = self.build_grid(config.points_per_axis);
        let (atom_centers, atom_charges): (Vec<[f32;3]>, Vec<f32>) = config
            .nuclei
            .iter()
            .map(|n| ([n.coordinates[0] as f32, n.coordinates[1] as f32, n.coordinates[2] as f32], n.atomic_number as f32))
            .unzip();

        let (density, eigenvalues) = scf_loop(
            &grid,
            &config.basis,
            config.num_electrons,
            &atom_centers,
            &atom_charges,
        );
        self.final_density = density;
        self.eigenvalues   = eigenvalues;
    }
}

fn build_overlap_matrix(
    grid: &[[f32; 3]],
    basis: &[Box<dyn BasisFunction>],
) -> DMatrix<f32> {
    let n = basis.len();
    let mut S = DMatrix::<f32>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for &pt in grid {
                sum += basis[i].value(&pt) * basis[j].value(&pt);
            }
            S[(i, j)] = sum;
        }
    }

    S
}

///// Poisson solver for the electron density. (FFT scaling instead of O(N^2) for large systems)

fn compute_hartree_potential_fft(
    density: &[f32],
    points_per_axis: usize,
) -> Vec<f32> {
    let n = points_per_axis;
    let n_grid = n * n * n;
    let l = points_per_axis as f32;
    let delta = (l as f32) / (n as f32); // assume grid spacing. NOTE: BE CAREFUL WITH THIS!
    log(format!("Computing Hartree potential on {}³ grid with delta = {}", n, delta));
    if density.len() != n_grid {
        panic!("Density length {} does not match grid size {}", density.len(), n_grid);
    }
    // let vol = l * l * l;

    // 1) pack density into complex array for FFT
    let mut data: Vec<Complex<f32>> = density
        .iter()
        .map(|&r| Complex::new(r, 0.0))
        .collect();

    // 2) forward 3D FFT (we do it as n^2 row‐FFTs on each axis)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // helper to index (i,j,k) -> flat
    let idx = |i, j, k| (i * n + j) * n + k;

    // FFT along k
    for i in 0..n {
        for j in 0..n {
            let mut row: Vec<_> = (0..n).map(|k| data[idx(i, j, k)]).collect();
            fft.process(&mut row);
            for k in 0..n { data[idx(i, j, k)] = row[k]; }
        }
    }
    // FFT along j
    for i in 0..n {
        for k in 0..n {
            let mut row: Vec<_> = (0..n).map(|j| data[idx(i, j, k)]).collect();
            fft.process(&mut row);
            for j in 0..n { data[idx(i, j, k)] = row[j]; }
        }
    }
    // FFT along i
    for j in 0..n {
        for k in 0..n {
            let mut row: Vec<_> = (0..n).map(|i| data[idx(i, j, k)]).collect();
            fft.process(&mut row);
            for i in 0..n { data[idx(i, j, k)] = row[i]; }
        }
    }

    // 3) Solve in k‐space: V(k) = (4π / |k|²) ρ(k), with k = 2π·m/L (thanks copilot for the unicode :D)
    for i in 0..n {
        let ki = if i <= n/2 { i as f32 } else { (i as f32) - (n as f32) };
        let kx = 2.0 * std::f32::consts::PI * ki / l;
        for j in 0..n {
            let kj = if j <= n/2 { j as f32 } else { (j as f32) - (n as f32) };
            let ky = 2.0 * std::f32::consts::PI * kj / l;
            for k in 0..n {
                let kk = if k <= n/2 { k as f32 } else { (k as f32) - (n as f32) };
                let kz = 2.0 * std::f32::consts::PI * kk / l;

                let k2 = kx*kx + ky*ky + kz*kz;
                let index = idx(i, j, k);
                if k2.abs() < 1e-12 {
                    data[index] = Complex::new(0.0, 0.0); // zero‐mode
                } else {
                    // Poisson: ∇²V = -4πρ  ⇒  V(k) = 4π ρ(k) / k²
                    data[index] *= Complex::new(4.0 * std::f32::consts::PI / k2, 0.0);
                }
            }
        }
    }

    // 4) inverse 3D FFT
    // along i
    for j in 0..n {
        for k in 0..n {
            let mut row: Vec<_> = (0..n).map(|i| data[idx(i, j, k)]).collect();
            ifft.process(&mut row);
            for i in 0..n { data[idx(i, j, k)] = row[i]; }
        }
    }
    // along j
    for i in 0..n {
        for k in 0..n {
            let mut row: Vec<_> = (0..n).map(|j| data[idx(i, j, k)]).collect();
            ifft.process(&mut row);
            for j in 0..n { data[idx(i, j, k)] = row[j]; }
        }
    }
    // along k
    for i in 0..n {
        for j in 0..n {
            let mut row: Vec<_> = (0..n).map(|k| data[idx(i, j, k)]).collect();
            ifft.process(&mut row);
            for k in 0..n { data[idx(i, j, k)] = row[k]; }
        }
    }

    // 5) normalize and extract real part
    data
        .into_iter()
        .map(|c| c.re / (n_grid as f32))  // rustfft doesn’t normalize inverse
        .collect()
}

fn build_full_hamiltonian(
    grid: &[[f32;3]],
    basis: &[Box<dyn BasisFunction>],
    atomic_centers: &[[f32;3]],
    atomic_charges: &[f32],
    density: &[f32],
) -> DMatrix<f32> {
    let n = basis.len();
    let n_grid = grid.len();
    // let dv = 1.0 / n_grid as f32;
    let dv = 1.0 as f32 / (n_grid as f32).powi(3);
    let v_h = compute_hartree_potential_fft(density, (n_grid as f32).cbrt() as usize);

    let mut H = DMatrix::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut hij = 0.0;
            for (k, &pt) in grid.iter().enumerate() {
                let φi = basis[i].value(&pt);
                let φj = basis[j].value(&pt);

                // nuclear attraction
                let mut v_nuc = 0.0;
                for (ac, &charge) in atomic_centers.iter().zip(atomic_charges) {
                    let dx = pt[0] - ac[0];
                    let dy = pt[1] - ac[1];
                    let dz = pt[2] - ac[2];
                    let r  = (dx*dx + dy*dy + dz*dz).sqrt().max(1e-6);
                    v_nuc += -charge / r;
                }

                // exchange–correlation
                let ρ   = density[k].max(1e-12);
                let v_xc = EXCHANGE_CORRELATION_CONSTANT * ρ.powf(1.0/3.0);

                hij += (φi * (v_nuc + v_h[k] + v_xc) * φj) * dv;
            }
            H[(i,j)] = hij;
        }
    }

    H
}

/////

/// Update the electron density on the grid from the occupied eigenfunctions.
fn update_density(
    grid: &[[f32;3]],
    eigenvectors: &DMatrix<f32>,
    basis: &[Box<dyn BasisFunction>],
    num_electrons: usize,
) -> Vec<f32> {
    let n_grid = grid.len();
    let n_occ  = num_electrons.min(eigenvectors.ncols());
    let mut density = vec![0.0; n_grid];

    for (k, &pt) in grid.iter().enumerate() {
        let mut ρ_sum = 0.0;
        for occ in 0..n_occ {
            let mut ψ = 0.0;
            for b in 0..basis.len() {
                ψ += eigenvectors[(b, occ)] * basis[b].value(&pt);
            }
            ρ_sum += ψ * ψ;
        }
        density[k] = ρ_sum;
    }

    // normalize to N electrons
    let total: f32 = density.iter().sum();
    if total > 0.0 {
        for d in &mut density {
            *d /= total;
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
    grid: &[[f32;3]],
    basis: &[Box<dyn BasisFunction>],
    num_electrons: usize,
    atomic_centers: &[[f32;3]],
    atomic_charges: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut density = vec![1.0 / (grid.len() as f32); grid.len()];

    for iter in 0..MAX_ITERATIONS {
        log(format!("SCF iteration {}", iter + 1));

        let H = build_full_hamiltonian(grid, basis, atomic_centers, atomic_charges, &density);
        let S = build_overlap_matrix(grid, basis);

        // generalized eigenproblem H C = S C ε
        let L    = S.clone().cholesky().expect("Overlap not PD").l();
        let Linv = L.clone().try_inverse().unwrap();
        let Ht   = &Linv * &H * &Linv.transpose();
        let eig  = Ht.symmetric_eigen();
        let C    = Linv.transpose() * eig.eigenvectors;

        let new_d = update_density(grid, &C, basis, num_electrons);

        // simple linear mixing
        let diff: f32 = new_d.iter()
            .zip(density.iter())
            .map(|(a,b)| (a - b).abs())
            .sum();
        for (d_old, &d_new) in density.iter_mut().zip(&new_d) {
            *d_old = 0.2 * d_new + 0.8 * *d_old;
        }

        if diff < TOLERANCE {
            log(format!("Converged in {} iterations", iter + 1));
            return (density, Some(eig.eigenvalues.iter().cloned().collect()));
        }
    }

    log("SCF did not converge".to_string());
    (density, None)
}

pub fn run_scf_command(args: Vec<String>) {
    if args.len() != 1 {
        log("Usage: run_dft <config>".into());
        return;
    }
    let key  = &args[0];
    let cfgs = SIMULATION_CONFIGS.lock().unwrap();
    let cfg  = match cfgs.get(key) {
        Some(c) => c.clone(),
        None    => {
                log(format!("Unknown config '{}'", key));
                if !cfgs.is_empty() {
                    log(format!(
                        "Available configs: {}",
                        cfgs.keys().cloned().collect::<Vec<_>>().join(", ")
                    ));
                }
                return;
            }
    };

    log(format!("Running SCF on '{}'", key));
    let mut solver = DFTSolver::new();
    solver.run_scf(&cfg);

    *SIMULATION_STATE.lock().unwrap() =SimulationState::new(0.0, solver, SimulationStatus::Completed);

    log("SCF complete".into());
}
