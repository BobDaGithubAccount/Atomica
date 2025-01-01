import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Constants
TOLERANCE = 1e-3
MAX_ITERATIONS = 1000

def gaussian_basis(r, center, alpha):
    """Define a 3D Gaussian-type orbital."""
    return np.exp(-alpha * np.sum((r - center)**2, axis=-1))

def build_overlap_matrix(grid, centers, alphas):
    """Build the overlap matrix."""
    n_basis = len(centers)
    s = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            s[i, j] = np.sum(
                gaussian_basis(grid, centers[i], alphas[i]) *
                gaussian_basis(grid, centers[j], alphas[j])
            )
    return s

def lda_exchange_correlation(density):
    """LDA exchange-correlation potential."""
    return np.where(density > 0, -0.738558766 * density**(1.0 / 3.0), 0.0)

def build_hamiltonian(grid, density, centers, alphas, external_potential):
    """Build the Hamiltonian matrix."""
    n_basis = len(centers)
    xc_potential = lda_exchange_correlation(density)

    # Coulomb potential (Hartree potential)
    coulomb_potential = np.zeros_like(density)
    for i in range(len(grid)):
        coulomb_potential[i] = np.sum(density / (np.linalg.norm(grid[i] - grid, axis=-1) + 1e-10))

    # Hamiltonian
    hamiltonian = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            hamiltonian[i, j] = np.sum(
                gaussian_basis(grid, centers[i], alphas[i]) *
                gaussian_basis(grid, centers[j], alphas[j]) *
                (external_potential(grid) + xc_potential + coulomb_potential)
            )
    return hamiltonian

def calculate_density(grid, eigenvectors, centers, alphas, num_electrons):
    """Calculate the electron density considering orbital occupancy."""
    new_density = np.zeros(len(grid))
    occupancies = np.zeros(len(eigenvectors[0]))  # Initialize occupancy for each orbital

    # Determine occupancies based on the number of electrons
    for i in range(num_electrons):
        occupancies[i % len(occupancies)] += 1

    # Calculate the density
    for orbital_idx, occupancy in enumerate(occupancies):
        if occupancy == 0:
            continue
        orbital_vector = eigenvectors[:, orbital_idx]
        for i, r in enumerate(grid):
            for j, center in enumerate(centers):
                new_density[i] += occupancy * orbital_vector[j] * gaussian_basis(r, center, alphas[j])
    
    new_density **= 2  # Square to get density
    new_density /= np.sum(new_density)  # Normalize so integral of density corresponds with number of electrons
    return new_density

def scf_loop(grid, centers, alphas, external_potential, num_electrons, frame_callback):
    """Self-consistent field loop."""
    density = np.ones(len(grid)) * 0.1  # Initial density guess
    damping_factor = 0.25  # Damping to stabilize convergence
    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}...")
        hamiltonian = build_hamiltonian(grid, density, centers, alphas, external_potential)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        print(f"Eigenvalues: {eigenvalues[:num_electrons]}")

        # Update density
        new_density = calculate_density(grid, eigenvectors, centers, alphas, num_electrons)

        # Apply damping to stabilize SCF convergence
        density = damping_factor * new_density + (1 - damping_factor) * density

        frame_callback(density)

        # Check for convergence
        if np.sum(np.abs(new_density - density)) < TOLERANCE:
            print(f"Converged in {iteration + 1} iterations.")
            return density, eigenvalues

    print("SCF did not converge.")
    return density, None

def create_animation(atomic_number, num_electrons):
    grid_points = 10
    grid = np.array([(x, y, z) for x in np.linspace(-5, 5, grid_points)
                                for y in np.linspace(-5, 5, grid_points)
                                for z in np.linspace(-5, 5, grid_points)])

    centers = np.array([[0, 0, 0]])
    alphas = [1.0]

    external_potential = lambda r: -atomic_number / (np.linalg.norm(r, axis=-1) + 1e-10)

    frames = []

    def frame_callback(density):
        frames.append(density.reshape(grid_points, grid_points, grid_points))

    density, _ = scf_loop(grid, centers, alphas, external_potential, num_electrons, frame_callback)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        ax.set_title(f"Electron Density Frame {frame_idx}")
        density = frames[frame_idx]
        mean = np.mean(density)
        std = np.std(density)
        threshold = mean + 2 * std  # 2sigma threshold
        ax.voxels(density > threshold, facecolors='blue', edgecolors='gray')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100)

    writer = FFMpegWriter(fps=10, bitrate=1800)
    ani.save("electron_density.mp4", writer=writer)
    print("Animation saved as electron_density.mp4.")

if __name__ == "__main__":
    create_animation(atomic_number=1, num_electrons=2)
