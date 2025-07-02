use crate::log;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use crate::simulation::*;
use rustfft::{num_complex::Complex, FftPlanner};

const TOLERANCE: f32 = 1e-3;
const MAX_ITERATIONS: usize = 1000;

// Exchange–correlation constant for the electron gas (LDA, exchange only).
const EXCHANGE_CORRELATION_CONSTANT: f32 = -0.738558766;

// Physical box from –10 → +10 in each direction
const BOX_LENGTH: f32 = 20.0;

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
            let xf = -10.0 + BOX_LENGTH * (x as f32 / (points_per_axis as f32 - 1.0));
            for y in 0..points_per_axis {
                let yf = -10.0 + BOX_LENGTH * (y as f32 / (points_per_axis as f32 - 1.0));
                for z in 0..points_per_axis {
                    let zf = -10.0 + BOX_LENGTH * (z as f32 / (points_per_axis as f32 - 1.0));
                    grid.push([xf, yf, zf]);
                }
            }
        }
        grid
    }

    /// Run the self-consistent field (SCF) simulation.
    pub fn run_scf(&mut self, config: &SimulationConfig) {
        let points_per_axis = config.points_per_axis;
        let grid = self.build_grid(points_per_axis);
        let (atom_centers, atom_charges): (Vec<[f32; 3]>, Vec<f32>) = config
            .nuclei
            .iter()
            .map(|n| ([n.coordinates[0] as f32, n.coordinates[1] as f32, n.coordinates[2] as f32], n.atomic_number as f32))
            .unzip();

        let (density, eigenvalues) = scf_loop(
            &grid,
            points_per_axis,
            &config.basis,
            config.num_electrons,
            &atom_centers,
            &atom_charges,
        );
        self.final_density = density;
        self.eigenvalues = eigenvalues;
    }
}

pub fn volume_element(points_per_axis: usize) -> f32 {
    let n_grid = (points_per_axis * points_per_axis * points_per_axis) as f32;
    BOX_LENGTH.powi(3) / n_grid
}


fn build_overlap_matrix(
    grid: &[[f32; 3]],
    basis: &[Box<dyn BasisFunction>],
    points_per_axis: usize,
) -> DMatrix<f32> {
    let n = basis.len();
    let mut s = DMatrix::<f32>::zeros(n, n);

    let dv = volume_element(points_per_axis);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for &pt in grid {
                sum += basis[i].value(&pt) * basis[j].value(&pt) * dv;
            }
            s[(i, j)] = sum;
        }
    }
    s
}

/// Poisson solver for the electron density via FFT (periodic boundary conditions).
fn compute_hartree_potential_fft(
    density: &[f32],
    points_per_axis: usize,
) -> Vec<f32> {
    let n = points_per_axis;
    let n_grid = n * n * n;

    if density.len() != n_grid {
        panic!("Density length {} does not match grid size {}", density.len(), n_grid);
    }

    // Physical box length
    let l = BOX_LENGTH;
    // Grid spacing
    let n_f = n as f32;
    let delta = BOX_LENGTH / (n_f - 1.0);
    log(format!("Computing Hartree potential on {}³ grid with delta = {}", n, delta));

    // 1) pack density into complex array for FFT
    let mut data: Vec<Complex<f32>> = density
        .iter()
        .map(|&r| Complex::new(r, 0.0))
        .collect();

    // 2) forward 3D FFT (n² row‐FFTs on each axis)
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

    // 3) Solve Poisson in k‐space: V(k) = 4π ρ(k) / k², with k = 2π·m/L
    for i in 0..n {
        let ki = if i <= n / 2 { i as f32 } else { (i as f32) - (n as f32) };
        let kx = 2.0 * std::f32::consts::PI * ki / l;
        for j in 0..n {
            let kj = if j <= n / 2 { j as f32 } else { (j as f32) - (n as f32) };
            let ky = 2.0 * std::f32::consts::PI * kj / l;
            for k in 0..n {
                let kk = if k <= n / 2 { k as f32 } else { (k as f32) - (n as f32) };
                let kz = 2.0 * std::f32::consts::PI * kk / l;

                let k2 = kx * kx + ky * ky + kz * kz;
                let index = idx(i, j, k);
                if k2.abs() < 1e-12 {
                    data[index] = Complex::new(0.0, 0.0); // zero‐mode
                } else {
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

    let mut v = data
    .into_iter()
    .map(|c| c.re / (n_grid as f32))
    .collect::<Vec<f32>>();

    // Enforce Dirichlet BC: V = 0 on any boundary face (this will be coupled with DST)
    let n = points_per_axis;
    let idx = |i, j, k| (i * n + j) * n + k;
    for i in 0..n {
        for j in 0..n {
            v[idx(0,    i, j)] = 0.0;
            v[idx(n-1,  i, j)] = 0.0;
            v[idx(i,  0,   j)] = 0.0;
            v[idx(i, n-1, j)] = 0.0;
            v[idx(i,  j,   0)] = 0.0;
            v[idx(i,  j, n-1)] = 0.0;
        }
    }

    v
}

fn build_full_hamiltonian(
    grid: &[[f32; 3]],
    points_per_axis: usize,
    basis: &[Box<dyn BasisFunction>],
    atomic_centers: &[[f32; 3]],
    atomic_charges: &[f32],
    neutral_density: &[f32],
) -> DMatrix<f32> {
    let n = basis.len();
    let mut h = DMatrix::<f32>::zeros(n, n);

    let dv = volume_element(points_per_axis);
    let v_h = compute_hartree_potential_fft(neutral_density, points_per_axis);

    for i in 0..n {
        for j in 0..n {
            let mut hij = 0.0;
            for (k_idx, &pt) in grid.iter().enumerate() {
                let φi = basis[i].value(&pt);
                let φj = basis[j].value(&pt);

                // nuclear attraction + exchange-correlation
                let v_nuc = atomic_centers.iter().zip(atomic_charges).map(|(ac, &q)| {
                    let dx = pt[0] - ac[0];
                    let dy = pt[1] - ac[1];
                    let dz = pt[2] - ac[2];
                    -q / dx.hypot(dy).hypot(dz).max(1e-6)
                }).sum::<f32>();
                let ρ = neutral_density[k_idx].max(1e-12);
                let v_xc = EXCHANGE_CORRELATION_CONSTANT * ρ.powf(1.0 / 3.0);

                hij += φi * (v_nuc + v_h[k_idx] + v_xc) * φj * dv;
            }
            h[(i, j)] = hij;
        }
    }

    h
}

/// Update the electron density on the grid from the occupied eigenfunctions.
fn update_density(
    grid: &[[f32; 3]],
    eigenvectors: &DMatrix<f32>,
    basis: &[Box<dyn BasisFunction>],
    num_electrons: usize,
) -> Vec<f32> {
    let n_grid = grid.len() as f32;
    let vol = BOX_LENGTH.powi(3);
    let dv = vol / n_grid;
    let n_occ = num_electrons.min(eigenvectors.ncols());
    let mut density = vec![0.0; n_grid as usize];

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

    // Drift correction: normalize the density to match the number of electrons
    let total_raw: f32 = density.iter().sum();
    let physical_total = total_raw * dv;
    let n_e = num_electrons as f32;
    if physical_total > 1e-12 {
        let scale = n_e / physical_total;
        for d in &mut density {
            *d *= scale;
        }
    }
    density
}

/// The self-consistent field (SCF) loop.
/// 1. Build the neutralized density,
/// 2. Build the Hamiltonian,
/// 3. Diagonalize it,
/// 4. Update the electron density,
/// 5. Apply mixing, and
/// 6. Check for convergence.
fn scf_loop(
    grid: &[[f32; 3]],
    points_per_axis: usize,
    basis: &[Box<dyn BasisFunction>],
    num_electrons: usize,
    atomic_centers: &[[f32; 3]],
    atomic_charges: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    // Initialize with a uniform density: n_e / volume
    let initial_density_value = num_electrons as f32 / BOX_LENGTH.powi(3);
    let mut density = vec![initial_density_value; grid.len()];

    for iter in 0..MAX_ITERATIONS {
        log(format!("SCF iteration {}", iter + 1));

        // 1) Build neutralized density (ρ - ρ_bg)
        let vol = BOX_LENGTH.powi(3);
        let dv = vol / (grid.len() as f32);
        let n_e: f32 = density.iter().map(|&ρ| ρ * dv).sum();
        let z: f32 = atomic_charges.iter().sum();
        let rho_bg = (n_e - z) / vol;
        let mut neutral_density = Vec::with_capacity(density.len());
        for &value in &density {
            neutral_density.push(value - rho_bg);
        }
        log(format!(
            " SCF iter {}: N_e={:.6}, Z={:.6}, ρ_bg={:.3e}",
            iter + 1,
            n_e,
            z,
            rho_bg
        ));

        // 2) Build H and S
        let h = build_full_hamiltonian(
            grid,
            points_per_axis,
            basis,
            atomic_centers,
            atomic_charges,
            &neutral_density,
        );
        let s = build_overlap_matrix(grid, basis, points_per_axis);

        // 3) Solve generalized eigenproblem H C = S C ε
        let l_factor = s.clone().cholesky().expect("Overlap not positive-definite").l();
        let linv = l_factor.clone().try_inverse().unwrap();
        let ht = &linv * &h * &linv.transpose();
        let eig = ht.symmetric_eigen();
        let c = linv.transpose() * eig.eigenvectors;

        // 4) Update density from occupied eigenvectors
        let new_density = update_density(grid, &c, basis, num_electrons);

        // 5) Simple linear mixing
        let alpha = 0.2;
        let mut diff = 0.0;
        for (d_old, &d_new) in density.iter_mut().zip(&new_density) {
            diff += (d_new - *d_old).abs();
            *d_old = alpha * d_new + (1.0 - alpha) * *d_old;
        }

        // 6) Check for convergence
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
    let key = &args[0];
    let cfgs = SIMULATION_CONFIGS.lock().unwrap();
    let cfg = match cfgs.get(key) {
        Some(c) => c.clone(),
        None => {
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

    *SIMULATION_STATE.lock().unwrap() =
        SimulationState::new(0.0, solver, SimulationStatus::Completed);

    log("SCF complete".into());
}
