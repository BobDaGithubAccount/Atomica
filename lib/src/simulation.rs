use std::{collections::HashMap, sync::Mutex, any::Any};
use lazy_static::lazy_static;
use crate::dft_simulator::DFTSolver;

////////////////////////////////////////////////////////////////////////
/// Basis functions
////////////////////////////////////////////////////////////////////////

pub trait BasisFunction: Send + Sync {
    fn center(&self) -> [f32; 3];
    fn value(&self, point: &[f32; 3]) -> f32;
    fn clone_box(&self) -> Box<dyn BasisFunction>;
    fn as_any(&self) -> &dyn Any;
}

impl Clone for Box<dyn BasisFunction> {
    fn clone(&self) -> Box<dyn BasisFunction> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct GaussianBasis {
    pub center: [f32; 3],
    pub alpha: f32,
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
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A Cartesian Gaussian of arbitrary angular momentum (ℓₓ,ℓᵧ,ℓ𝓏).
#[derive(Debug, Clone)]
pub struct AngularGaussian {
    pub center: [f32; 3],
    pub alpha: f32,
    pub l: (usize, usize, usize),
}

impl BasisFunction for AngularGaussian {
    fn center(&self) -> [f32; 3] {
        self.center
    }

    fn value(&self, point: &[f32; 3]) -> f32 {
        let (lx, ly, lz) = self.l;
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;

        let alpha = self.alpha;
        let lsum = (lx + ly + lz) as f32;

        // avoid underflow by treating lx==0 specially
        let df_x_arg = if lx == 0 { 0 } else { 2 * lx - 1 };
        let df_y_arg = if ly == 0 { 0 } else { 2 * ly - 1 };
        let df_z_arg = if lz == 0 { 0 } else { 2 * lz - 1 };

        let norm_prefactor = (2.0 * alpha / std::f32::consts::PI).powf(0.75);
        let poly_prefactor = (4.0 * alpha).powf(lsum / 2.0);

        let norm_denom = (
            double_factorial(df_x_arg) *
            double_factorial(df_y_arg) *
            double_factorial(df_z_arg)
        ).sqrt();

        let norm = norm_prefactor * poly_prefactor / norm_denom;

        norm
            * dx.powi(lx as i32)
            * dy.powi(ly as i32)
            * dz.powi(lz as i32)
            * (-alpha * r2).exp()
    }
    fn clone_box(&self) -> Box<dyn BasisFunction> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

}

fn double_factorial(n: usize) -> f32 {
    if n == 0 || n == 1 {
        1.0
    } else {
        (n as f32) * double_factorial(n - 2)
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
                nuclei: vec![Nucleus { species: "H".into(), atomic_number: 1, coordinates: [0.0, 0.0, 0.0] }],
                num_electrons: 1,
                basis: vec![Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.0})],
                points_per_axis: 32,
                tolerance: 1e-1
            },
        );

        // Helium
        m.insert(
            "helium".into(),
            SimulationConfig {
                nuclei: vec![Nucleus { species: "He".into(), atomic_number: 2, coordinates: [0.0, 0.0, 0.0] }],
                num_electrons: 2,
                basis: vec![Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 1.0})],
                points_per_axis: 32,
                tolerance: 1e-1
            },
        );

        // H2 molecule
        m.insert(
            "h2_molecule".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.0, 0.0, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [1.0, 0.0, 0.0] },
                ],
                num_electrons: 2,
                basis: vec![
                    Box::new(GaussianBasis { center: [-1.0, 0.0, 0.0], alpha: 4.0}),
                    Box::new(GaussianBasis { center: [1.0, 0.0, 0.0], alpha: 4.0,}),
                ],
                points_per_axis: 32,
                tolerance: 1e-1
            },
        );

        // H2 low-resolution
        m.insert(
            "h2_molecule_low_resolution".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.0, 0.0, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [1.0, 0.0, 0.0] },
                ],
                num_electrons: 2,
                basis: vec![
                    Box::new(GaussianBasis { center: [-0.37, 0.0, 0.0], alpha: 1.0}),
                    Box::new(GaussianBasis { center: [0.37, 0.0, 0.0], alpha: 1.0}),
                ],
                points_per_axis: 10,
                tolerance: 1e-1
            },
        );

        m.insert(
            "oxygen".into(),
            SimulationConfig {
                points_per_axis: 48,
                nuclei: vec![Nucleus {
                    species: "O".into(),
                    atomic_number: 8,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 8,
                basis: vec![
                    // Core 1s STO-3G-like
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 130.07, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 23.81, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 6.44, l: (0, 0, 0) }),

                    // 2s valence
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.1687, l: (0, 0, 0) }),

                    // 2p valence (px, py, pz)
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1, l: (1, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1, l: (0, 1, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1, l: (0, 0, 1) }),

                    // Optional diffuse d-functions
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25, l: (2, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25, l: (0, 2, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25, l: (0, 0, 2) }),
                ],
                tolerance: 1e-1
            },
        );
        m.insert(
            "oxygen_low_res".into(),
            SimulationConfig {
                points_per_axis: 20,
                nuclei: vec![Nucleus {
                    species: "O".into(),
                    atomic_number: 8,
                    coordinates: [0.0, 0.0, 0.0],
                }],
                num_electrons: 8,
                basis: vec![
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 130.070932 / 12.0, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 23.808861 / 12.0, l: (0, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 6.4436083 / 12.0, l: (0, 0, 0) }),

                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1 / 12.0, l: (1, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1 / 12.0, l: (0, 1, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.1 / 12.0, l: (0, 0, 1) }),

                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.1687144 / 12.0, l: (0, 0, 0) }),

                    // optional d‐functions
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25 / 12.0, l: (2, 0, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25 / 12.0, l: (0, 2, 0) }),
                    Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 0.25 / 12.0, l: (0, 0, 2) }),
                ],
                tolerance: 1e-1
            },
        );


        m.insert(
            "benzene".into(),
            SimulationConfig {
                // Carbon ring: radius ~1.397 Å; hydrogens ~1.09 Å further out (C-H ≈ 1.09 Å)
                nuclei: vec![
                    // Carbons (6)
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [ 1.397,  0.0,     0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [ 0.6985,  1.209837,0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [-0.6985,  1.209837,0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [-1.397,  0.0,     0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [-0.6985, -1.209837,0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [ 0.6985, -1.209837,0.0] },

                    // Hydrogens (6)
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 2.487,   0.0,      0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 1.2435,  2.153805, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.2435,  2.153805, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-2.487,   0.0,     0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.2435, -2.153805, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 1.2435, -2.153805, 0.0] },
                ],
                num_electrons: 42,
                // Basis: simple contracted-like sets built from isotropic Gaussians only (no angular Gaussians)
                // Carbon STO-3G-like exponents: 71.6168370, 13.0450960, 3.5305122
                // Hydrogen STO-3G-like exponents: 3.42525091, 0.62391373, 0.1688554
                basis: vec![
                    // Carbon centers (3 gaussians per carbon)
                    Box::new(GaussianBasis { center: [ 1.397,   0.0,      0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [ 1.397,   0.0,      0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [ 1.397,   0.0,      0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [ 0.6985,  1.209837, 0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [ 0.6985,  1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [ 0.6985,  1.209837, 0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [-0.6985,  1.209837, 0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [-0.6985,  1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [-0.6985,  1.209837, 0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [-1.397,   0.0,      0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [-1.397,   0.0,      0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [-1.397,   0.0,      0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [-0.6985, -1.209837, 0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [-0.6985, -1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [-0.6985, -1.209837, 0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [ 0.6985, -1.209837, 0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [ 0.6985, -1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [ 0.6985, -1.209837, 0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [ 2.487,   0.0,      0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [ 2.487,   0.0,      0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 2.487,   0.0,      0.0], alpha: 0.1688554 }),

                    Box::new(GaussianBasis { center: [ 1.2435,  2.153805, 0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [ 1.2435,  2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 1.2435,  2.153805, 0.0], alpha: 0.1688554 }),

                    Box::new(GaussianBasis { center: [-1.2435,  2.153805, 0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [-1.2435,  2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-1.2435,  2.153805, 0.0], alpha: 0.1688554 }),

                    Box::new(GaussianBasis { center: [-2.487,   0.0,      0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [-2.487,   0.0,      0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-2.487,   0.0,      0.0], alpha: 0.1688554 }),

                    Box::new(GaussianBasis { center: [-1.2435, -2.153805, 0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [-1.2435, -2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-1.2435, -2.153805, 0.0], alpha: 0.1688554 }),

                    Box::new(GaussianBasis { center: [ 1.2435, -2.153805, 0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [ 1.2435, -2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 1.2435, -2.153805, 0.0], alpha: 0.1688554 }),
                ],
                // finer real-space sampling because aromatic pi-system benefits from resolution
                points_per_axis: 64,
                tolerance: 1e-2,
            },
        );

        // Lower-resolution / faster debug version (will actually run well in a browser)
        m.insert(
            "benzene_low_res".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [ 1.397,  0.0,     0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [ 0.6985,  1.209837,0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [-0.6985,  1.209837,0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [-1.397,  0.0,     0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [-0.6985, -1.209837,0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [ 0.6985, -1.209837,0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 2.487,   0.0,      0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 1.2435,  2.153805, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.2435,  2.153805, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-2.487,   0.0,     0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-1.2435, -2.153805, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 1.2435, -2.153805, 0.0] },
                ],
                num_electrons: 42,
                basis: vec![
                    Box::new(GaussianBasis { center: [ 1.397,   0.0,      0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [ 0.6985,  1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [-0.6985,  1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [-1.397,   0.0,      0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [-0.6985, -1.209837, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [ 0.6985, -1.209837, 0.0], alpha: 13.0450960 }),

                    Box::new(GaussianBasis { center: [ 2.487,   0.0,      0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 1.2435,  2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-1.2435,  2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-2.487,   0.0,      0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-1.2435, -2.153805, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 1.2435, -2.153805, 0.0], alpha: 0.62391373 }),
                ],
                points_per_axis: 32,
                tolerance: 1e-1,
            },
        );

        m.insert(
            "co2".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "O".into(), atomic_number: 8, coordinates: [-1.16, 0.0, 0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [0.0,  0.0, 0.0]   },
                    Nucleus { species: "O".into(), atomic_number: 8, coordinates: [ 1.16, 0.0, 0.0] },
                ],
                num_electrons: 22,
                basis: vec![
                    Box::new(GaussianBasis { center: [-1.16, 0.0, 0.0], alpha: 130.70932 }),
                    Box::new(GaussianBasis { center: [-1.16, 0.0, 0.0], alpha: 23.808861 }),
                    Box::new(GaussianBasis { center: [-1.16, 0.0, 0.0], alpha: 6.4436083 }),

                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 3.5305122 }),

                    Box::new(GaussianBasis { center: [1.16, 0.0, 0.0], alpha: 130.70932 }),
                    Box::new(GaussianBasis { center: [1.16, 0.0, 0.0], alpha: 23.808861 }),
                    Box::new(GaussianBasis { center: [1.16, 0.0, 0.0], alpha: 6.4436083 }),
                ],
                points_per_axis: 56,
                tolerance: 1e-1,
            },
        );

        m.insert(
            "co2_low_res".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "O".into(), atomic_number: 8, coordinates: [-1.16, 0.0, 0.0] },
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [0.0,  0.0, 0.0]   },
                    Nucleus { species: "O".into(), atomic_number: 8, coordinates: [ 1.16, 0.0, 0.0] },
                ],
                num_electrons: 22,
                basis: vec![
                    Box::new(GaussianBasis { center: [-1.16, 0.0, 0.0], alpha: 23.808861 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0],    alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [1.16, 0.0, 0.0],   alpha: 23.808861 }),
                ],
                points_per_axis: 28,
                tolerance: 1e-1,
            },
        );

        m.insert(
            "water".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "O".into(), atomic_number: 8, coordinates: [0.0, 0.0, 0.0] },
                    // coordinates computed with OH = 0.9572 Å, H-O-H = 104.45
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.7565922909,  0.5863445619, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.7565922909,  0.5863445619, 0.0] },
                ],
                num_electrons: 10,
                basis: vec![
                    // O
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 130.70932 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 23.808861 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 6.4436083 }),
                    // H (each)
                    Box::new(GaussianBasis { center: [ 0.7565922909, 0.5863445619, 0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [ 0.7565922909, 0.5863445619, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 0.7565922909, 0.5863445619, 0.0], alpha: 0.1688554 }),

                    Box::new(GaussianBasis { center: [-0.7565922909, 0.5863445619, 0.0], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [-0.7565922909, 0.5863445619, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-0.7565922909, 0.5863445619, 0.0], alpha: 0.1688554 }),
                ],
                points_per_axis: 48,
                tolerance: 1e-1,
            },
        );

        m.insert(
            "water_low_res".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "O".into(), atomic_number: 8, coordinates: [0.0, 0.0, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.7565922909,  0.5863445619, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.7565922909,  0.5863445619, 0.0] },
                ],
                num_electrons: 10,
                basis: vec![
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 23.808861 }),
                    Box::new(GaussianBasis { center: [ 0.7565922909, 0.5863445619, 0.0], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-0.7565922909, 0.5863445619, 0.0], alpha: 0.62391373 }),
                ],
                points_per_axis: 20,
                tolerance: 1e-1,
            },
        );

        // --- Methane (CH4) ---
        m.insert(
            "methane".into(),
            SimulationConfig {
                nuclei: {
                    let mut v = Vec::new();
                    v.push(Nucleus { species: "C".into(), atomic_number: 6, coordinates: [0.0, 0.0, 0.0] });
                    // Tetrahedral H positions (CH = 1.09 Å) (approx)
                    v.push(Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.6293117934,  0.6293117934,  0.6293117934] });
                    v.push(Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.6293117934, -0.6293117934, -0.6293117934] });
                    v.push(Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.6293117934,  0.6293117934, -0.6293117934] });
                    v.push(Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.6293117934, -0.6293117934,  0.6293117934] });
                    v
                },
                num_electrons: 10,
                basis: vec![
                    // Carbon
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 71.6168370 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [0.0, 0.0, 0.0], alpha: 3.5305122 }),
                    // Hydrogens
                    Box::new(GaussianBasis { center: [ 0.6293117934,  0.6293117934,  0.6293117934], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [ 0.6293117934, -0.6293117934, -0.6293117934], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [-0.6293117934,  0.6293117934, -0.6293117934], alpha: 3.42525091 }),
                    Box::new(GaussianBasis { center: [-0.6293117934, -0.6293117934,  0.6293117934], alpha: 3.42525091 }),
                ],
                points_per_axis: 48,
                tolerance: 1e-1,
            },
        );

        m.insert(
            "methane_low_res".into(),
            SimulationConfig {
                nuclei: vec![
                    Nucleus { species: "C".into(), atomic_number: 6, coordinates: [0.0, 0.0, 0.0] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.6293117934,  0.6293117934,  0.6293117934] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [ 0.6293117934, -0.6293117934, -0.6293117934] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.6293117934,  0.6293117934, -0.6293117934] },
                    Nucleus { species: "H".into(), atomic_number: 1, coordinates: [-0.6293117934, -0.6293117934,  0.6293117934] },
                ],
                num_electrons: 10,
                basis: vec![
                    Box::new(GaussianBasis { center: [0.0,0.0,0.0], alpha: 13.0450960 }),
                    Box::new(GaussianBasis { center: [ 0.6293117934,  0.6293117934,  0.6293117934], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [ 0.6293117934, -0.6293117934, -0.6293117934], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-0.6293117934,  0.6293117934, -0.6293117934], alpha: 0.62391373 }),
                    Box::new(GaussianBasis { center: [-0.6293117934, -0.6293117934,  0.6293117934], alpha: 0.62391373 }),
                ],
                points_per_axis: 24,
                tolerance: 1e-1,
            },
        );

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
        let state = SimulationState {
            total_time,
            status,
            dft_simulator,
            file_context: None,
        };
        *SIMULATION_STATE.lock().unwrap() = state.clone();
        state
    }
    pub fn print_summary(&self) -> String {
        format!(
            "Simulation State:\n  Total Time: {:.3} a.u.\n  Status: {:?}\n",
            self.total_time, self.status
        )
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

#[derive(Clone)]
pub struct SimulationConfig {
    pub points_per_axis: usize,
    pub nuclei: Vec<Nucleus>,
    pub num_electrons: usize,
    pub basis: Vec<Box<dyn BasisFunction>>,
    pub tolerance: f32,
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
            basis: vec![Box::new(AngularGaussian { center: [0.0, 0.0, 0.0], alpha: 1.0, l: (0,0,0)})],
            tolerance: 1e-5
        }
    }
}