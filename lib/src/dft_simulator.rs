use core::str;

use crate::{log, generate_cube_string};
use log::info;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use crate::simulation::*;
use rustfft::{num_complex::Complex, FftPlanner};
use statrs::function::erf::erfc;


const TOLERANCE: f32 = 1e-1;
const MAX_ITERATIONS: usize = 100;

// Exchange–correlation constant for the electron gas (LDA, exchange only).
const EXCHANGE_CORRELATION_CONSTANT: f32 = -0.738558766;

// Physical box from –50 → +50 in each direction
const BOX_LENGTH: f32 = 48.0;

#[derive(Debug, Serialize, Deserialize)]
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
            let half = BOX_LENGTH / 2.0;
            let xf   = -half + BOX_LENGTH * (x as f32 / (points_per_axis as f32 - 1.0));
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

        ///////////////////////////
        // Find the index of your px basis in `config.basis`

//        let px_index = config.basis.iter().enumerate().find_map(|(i, b)| {
//            if let Some(g) = b.as_any().downcast_ref::<AngularGaussian>() {
//                if g.l == (1, 0, 0)
//                    && g.center == [0.0, 0.0, 0.0]
//                    && (g.alpha - (1.1 / 12.0)).abs() < 1e-6
//                {
//                    Some(i)
//                } else {
//                    None
//                }
//            } else {
//                None
//            }
//        }).expect("Couldn't find px orbital in basis set");
//
//        let dx = BOX_LENGTH / (points_per_axis as f32 - 1.0);
//        info!("Sampling φ_px along +x axis:");
//        for i in 0..points_per_axis {
//            let x = -BOX_LENGTH/2.0 + i as f32 * dx;
//            // grid index at (i, mid, mid)
//            let idx = (i * points_per_axis + points_per_axis/2) * points_per_axis
//                    + points_per_axis/2;
//            let φx = config.basis[px_index].value(&[x, 0.0, 0.0]);
//            info!("  φ_px({:+.3}) = {:.6}", x, φx);
//        }
//        info!("— end px diagnostic —\n");

        // -----------------------------------

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

impl Clone for DFTSolver {
    fn clone(&self) -> Self {
        DFTSolver {
            final_density: self.final_density.clone(),
            eigenvalues: self.eigenvalues.clone(),
        }
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

pub fn compute_hartree_potential_fft_padded(
    neutral_density: &[f32],
    points_per_axis: usize,
    pad: usize,
) -> Vec<f32> {
    let n = points_per_axis;
    let N = n + 2 * pad;
    let N3 = N * N * N;
    assert_eq!(neutral_density.len(), n*n*n,
        "Density len {} != {}³", neutral_density.len(), n);
    let delta = BOX_LENGTH / (n as f32 - 1.0);
    let dv = delta.powi(3);

    log(format!("Padded FFT Poisson: {}³ → {}³ (pad={})", n, N, pad));

    // --- short‐range real‐space erfc sum on original grid ---
    let alpha = 3.0 / delta;
    let r_cut2 = (4.0 * delta).powi(2);
    let max_i = (r_cut2.sqrt() / delta).ceil() as isize;
    let mut neighbor = Vec::new();
    for dx in -max_i..=max_i {
        for dy in -max_i..=max_i {
            for dz in -max_i..=max_i {
                let r2 = (dx*dx + dy*dy + dz*dz) as f32;
                if r2 > 0.0 && r2 * delta * delta <= r_cut2 {
                    neighbor.push((dx, dy, dz, r2));
                }
            }
        }
    }
    let mut v_short = vec![0.0f32; n*n*n];
    let idx_opt = |x:isize, y:isize, z:isize| {
        if x<0 || x>=n as isize || y<0 || y>=n as isize || z<0 || z>=n as isize {
            None
        } else {
            Some((x as usize * n + y as usize) * n + z as usize)
        }
    };
    for ix in 0..n as isize {
        for iy in 0..n as isize {
            for iz in 0..n as isize {
                let dst = (ix as usize * n + iy as usize) * n + iz as usize;
                let mut sum = 0.0f32;
                for &(dx,dy,dz,r2) in &neighbor {
                    if let Some(src) = idx_opt(ix+dx, iy+dy, iz+dz) {
                        let r = r2.sqrt() * delta;
                        sum += neutral_density[src] * (erfc((alpha * r) as f64) / (r as f64)) as f32;
                    }
                }
                v_short[dst] = sum * dv;
            }
        }
    }

    // --- pack neutral_density * dv into padded complex array ---
    let mut data = vec![Complex::new(0.0,0.0); N3];
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let src = (ix * n + iy) * n + iz;
                let dst = ((ix+pad) * N + (iy+pad)) * N + (iz+pad);
                data[dst] = Complex::new(neutral_density[src] * dv, 0.0);
            }
        }
    }

    // --- 3D FFT forward ---
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let ifft = planner.plan_fft_inverse(N);
    let idx3 = |i,j,k| ((i * N + j) * N + k) as usize;

    for i in 0..N {
        for j in 0..N {
            let mut row = (0..N).map(|k| data[idx3(i,j,k)]).collect::<Vec<_>>();
            fft.process(&mut row);
            for k in 0..N { data[idx3(i,j,k)] = row[k]; }
        }
    }
    for i in 0..N {
        for k in 0..N {
            let mut row = (0..N).map(|j| data[idx3(i,j,k)]).collect::<Vec<_>>();
            fft.process(&mut row);
            for j in 0..N { data[idx3(i,j,k)] = row[j]; }
        }
    }
    for j in 0..N {
        for k in 0..N {
            let mut row = (0..N).map(|i| data[idx3(i,j,k)]).collect::<Vec<_>>();
            fft.process(&mut row);
            for i in 0..N { data[idx3(i,j,k)] = row[i]; }
        }
    }

    // --- apply screened Green's function on padded box length ---
    let Lpad = BOX_LENGTH + 2.0 * pad as f32 * delta;
    let four_pi = 4.0 * std::f32::consts::PI;
    for i in 0..N {
        let ki = if i <= N/2 { i as f32 } else { i as f32 - N as f32 };
        let kx = two_pi() * ki / Lpad;
        for j in 0..N {
            let kj = if j <= N/2 { j as f32 } else { j as f32 - N as f32 };
            let ky = two_pi() * kj / Lpad;
            for k in 0..N {
                let kk = if k <= N/2 { k as f32 } else { k as f32 - N as f32 };
                let kz = two_pi() * kk / Lpad;
                let k2 = kx*kx + ky*ky + kz*kz;
                let gk = if k2 > 1e-12 {
                    four_pi / k2 * (-k2 / (4.0*alpha*alpha)).exp()
                } else { 0.0 };
                data[idx3(i,j,k)] *= Complex::new(gk, 0.0);
            }
        }
    }

    // --- inverse 3D FFT ---
    for j in 0..N {
        for k in 0..N {
            let mut row = (0..N).map(|i| data[idx3(i,j,k)]).collect::<Vec<_>>();
            ifft.process(&mut row);
            for i in 0..N { data[idx3(i,j,k)] = row[i]; }
        }
    }
    for i in 0..N {
        for k in 0..N {
            let mut row = (0..N).map(|j| data[idx3(i,j,k)]).collect::<Vec<_>>();
            ifft.process(&mut row);
            for j in 0..N { data[idx3(i,j,k)] = row[j]; }
        }
    }
    for i in 0..N {
        for j in 0..N {
            let mut row = (0..N).map(|k| data[idx3(i,j,k)]).collect::<Vec<_>>();
            ifft.process(&mut row);
            for k in 0..N { data[idx3(i,j,k)] = row[k]; }
        }
    }

    // --- crop back, normalize and combine ---
    let mut v = vec![0.0f32; n*n*n];
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let src = ((ix+pad)*N + (iy+pad))*N + (iz+pad);
                let dst = (ix * n + iy) * n + iz;
                // inv_N3 applied implicitly by FFT library, dv already in data
                v[dst] = data[src].re + v_short[dst];
            }
        }
    }

    v
}

// two_pi helper
#[inline(always)]
fn two_pi() -> f32 {
    std::f32::consts::PI * 2.0
}

///// Pure direct‐sum Hartree (no FFT).  Very slow - only for diagnosis!
pub fn compute_hartree_potential_direct(
    density: &[f32],
    points_per_axis: usize,
) -> Vec<f32> {
    let n = points_per_axis;
    let N3 = n * n * n;
    assert_eq!(density.len(), N3,
        "Density len {} != {}³", density.len(), n);
    let delta = BOX_LENGTH / (n as f32 - 1.0);
    let dv = delta.powi(3);

    log(format!("Direct‐sum Poisson: {}³ grid, Δ={:.6}", n, delta));

    // collect coordinates of each grid point
    let mut coords = Vec::with_capacity(N3);
    for i in 0..n {
        let x = -BOX_LENGTH/2.0 + delta * (i as f32);
        for j in 0..n {
            let y = -BOX_LENGTH/2.0 + delta * (j as f32);
            for k in 0..n {
                let z = -BOX_LENGTH/2.0 + delta * (k as f32);
                coords.push([x, y, z]);
            }
        }
    }

    let mut v = vec![0.0f32; N3];
    for a in 0..N3 {
        let pa = coords[a];
        let mut sum = 0.0f32;
        for b in 0..N3 {
            if a == b { continue; }
            let pb = coords[b];
            let dx = pa[0] - pb[0];
            let dy = pa[1] - pb[1];
            let dz = pa[2] - pb[2];
            let r = (dx*dx + dy*dy + dz*dz).sqrt().max(1e-6);
            sum += density[b] * (1.0 / r);
        }
        v[a] = sum * dv;
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
    let v_h = compute_hartree_potential_fft_padded(neutral_density, points_per_axis, 1);

//    let v_h = compute_hartree_potential_direct(neutral_density, points_per_axis);

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

        //// Diagnostics ////

        let m = (&c.transpose() * &s) * &c;
        for i in 0..3 {
            for j in 0..3 {
                info!("M[{},{}] = {:.3e}", i, j, m[(i,j)]);
            }
        }

        for (k,&ev) in eig.eigenvalues.iter().enumerate() {
            info!("  ε[{}] = {:.6}", k, ev);
        }

        for (i, b) in basis.iter().enumerate() {

            let gauss: &AngularGaussian = b
                .as_any()
                .downcast_ref::<AngularGaussian>()
                .expect("Expected AngularGaussian");
            info!(
                "Basis[{}] at {:?}, l=({},{},{}), alpha={}",
                i, gauss.center, gauss.l.0, gauss.l.1, gauss.l.2, gauss.alpha
            );
        }

        ///////////////////

        // 4) Update density from occupied eigenvectors
        let new_density = update_density(grid, &c, basis, num_electrons);

        // 5) Simple linear mixing
        let alpha = 0.2;
        let mut diff = 0.0;
        for (d_old, &d_new) in density.iter_mut().zip(&new_density) {
            diff += (d_new - *d_old).abs();
            *d_old = alpha * d_new + (1.0 - alpha) * *d_old;
        }

        log(String::from(format!(" SCF iter {}: Density updated, diff={:.3e}", iter + 1, diff)));

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
        SimulationState::new(0.0, solver.clone(), SimulationStatus::Completed);

    log("SCF complete".into());

    info!("{}", generate_cube_string(&solver.final_density, cfg.points_per_axis, BOX_LENGTH));
}