import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
TOLERANCE = 1e-6
MAX_ITERATIONS = 1000

def gaussian_basis(x, center, alpha):
    """Define a Gaussian-type orbital."""
    return np.exp(-alpha * (x - center)**2)

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
    coulomb_potential = np.zeros_like(grid)
    for i in range(len(grid)):
        coulomb_potential[i] = np.sum(density / np.abs(grid[i] - grid + 1e-10))

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

def scf_loop(grid, centers, alphas, external_potential, num_electrons, frame_callback):
    """Self-consistent field loop."""
    density = np.ones_like(grid) * 0.1  # Initial density guess
    damping_factor = 0.1  # Damping to stabilize convergence

    for iteration in range(MAX_ITERATIONS):
        # Build the Hamiltonian matrix
        hamiltonian = build_hamiltonian(grid, density, centers, alphas, external_potential)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

        # Debugging: Print eigenvalues
        print(f"Iteration {iteration + 1}: Eigenvalues = {eigenvalues[:num_electrons]}")
        print(f"Hamiltonian shape: {hamiltonian.shape}, Eigenvectors shape: {eigenvectors.shape}")

        # Ensure there are enough eigenvectors for all electrons
        if eigenvectors.shape[1] < num_electrons:
            raise ValueError("Not enough eigenvectors for the specified number of electrons. Increase the basis set size.")

        # Update density using the lowest-energy eigenvectors
        new_density = np.zeros_like(grid)
        for electron_idx in range(num_electrons):
            lowest_vector = eigenvectors[:, electron_idx]
            print(f"lowest_vector norm (electron {electron_idx}): {np.linalg.norm(lowest_vector)}")
            for i, x in enumerate(grid):
                for j, center in enumerate(centers):
                    new_density[i] += lowest_vector[j] * gaussian_basis(x, center, alphas[j])
        new_density **= 2  # Square to get the density
        new_density /= np.sum(new_density)  # Normalize

        # Damping to stabilise convergence
        density = damping_factor * new_density + (1 - damping_factor) * density

        frame_callback(density)

        # Check convergence
        if np.sum(np.abs(new_density - density)) < TOLERANCE:
            print(f"Converged in {iteration + 1} iterations.")
            return density, eigenvalues

    print("SCF did not converge.")
    return density, None

def create_animation(atomic_number, num_electrons):
    fig, ax = plt.subplots()
    grid = np.linspace(-5.0, 5.0, 100)

    # Ensure enough basis functions for the number of electrons (by ensuring there's enough eigenstates)
    num_basis_functions = max(num_electrons, 1)
    centers = np.linspace(-2.0, 2.0, num_basis_functions)
    alphas = [1.0] * len(centers)

    # Define external potential based on the atomic number (Z) (coloumb potential)
    external_potential = lambda x: -atomic_number / (np.abs(x) + 1e-10)

    # Initialize visualization
    im = ax.imshow(
        np.zeros((100, 100)),
        extent=(-5, 5, -5, 5),
        origin="lower",
        cmap="coolwarm",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Electron Density")

    ax.set_title(f"Atomic Density: Z={atomic_number}, Electrons={num_electrons}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    frames = []

    def frame_callback(density):
        frames.append(density.copy())

    def update(frame):
        # Scale the color limits based on the current frame's statistics
        current_density = frames[frame]
        mean = np.mean(current_density)
        std = np.std(current_density)
        vmin = max(0, mean - 2 * std)
        vmax = mean + 2 * std
        im.set_clim(vmin, vmax)
        im.set_array(np.outer(current_density, current_density))

    # Run the SCF loop
    density, _ = scf_loop(grid, centers, alphas, external_potential, num_electrons, frame_callback)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)
    ani.save(f"atom_Z{atomic_number}_e{num_electrons}_density_animation.mp4", fps=10)
    plt.show()

if __name__ == "__main__":
    # Example: Carbon atom (Z=6) with 6 electrons
    create_animation(atomic_number=1, num_electrons=2)



    ##TODO ISSUE WITH CENTERS AND EIGENSTATES CAUSING WEIRD BEHAVIOUR