import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.reduction as cl_reduction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

TOLERANCE = 1e-3
MAX_ITERATIONS = 1000

platforms = cl.get_platforms()
devices = platforms[0].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context)
mf = cl.mem_flags

kernel_code = """
__kernel void gaussian_basis(
    __global const float *r,
    __global const float *center,
    float alpha,
    __global float *result
) {
    int gid = get_global_id(0);
    float dx = r[3*gid] - center[0];
    float dy = r[3*gid+1] - center[1];
    float dz = r[3*gid+2] - center[2];
    float dist_sq = dx*dx + dy*dy + dz*dz;
    result[gid] = exp(-alpha * dist_sq);
}

__kernel void overlap_matrix(
    __global const float *grid,
    __global const float *centers,
    __global const float *alphas,
    __global float *result,
    int n_basis,
    int n_grid
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float s = 0.0f;
    for (int k = 0; k < n_grid; k++) {
        float dx_i = grid[3*k] - centers[3*i];
        float dy_i = grid[3*k+1] - centers[3*i+1];
        float dz_i = grid[3*k+2] - centers[3*i+2];
        float g_i = exp(-alphas[i] * (dx_i*dx_i + dy_i*dy_i + dz_i*dz_i));
        
        float dx_j = grid[3*k] - centers[3*j];
        float dy_j = grid[3*k+1] - centers[3*j+1];
        float dz_j = grid[3*k+2] - centers[3*j+2];
        float g_j = exp(-alphas[j] * (dx_j*dx_j + dy_j*dy_j + dz_j*dz_j));
        
        s += g_i * g_j;
    }
    result[i*n_basis + j] = s;
}

__kernel void density_update(
    __global const float *grid,
    __global const float *eigenvectors,
    __global const float *centers,
    __global const float *alphas,
    __global float *density,
    int n_basis,
    int n_grid
) {
    int gid = get_global_id(0);
    float new_density = 0.0f;
    for (int j = 0; j < n_basis; j++) {
        float dx = grid[3*gid] - centers[3*j];
        float dy = grid[3*gid+1] - centers[3*j+1];
        float dz = grid[3*gid+2] - centers[3*j+2];
        float g = exp(-alphas[j] * (dx*dx + dy*dy + dz*dz));
        new_density += eigenvectors[j] * g;
    }
    density[gid] = new_density * new_density;
}
"""

program = cl.Program(context, kernel_code).build()

def calculate_gaussian_basis(grid, center, alpha):
    """Calculate Gaussian basis using OpenCL."""
    r_gpu = cl_array.to_device(queue, grid.astype(np.float32))
    center_gpu = cl_array.to_device(queue, center.astype(np.float32))
    result_gpu = cl_array.zeros(queue, grid.shape[0], dtype=np.float32)

    program.gaussian_basis(queue, (grid.shape[0],), None, r_gpu.data, center_gpu.data, np.float32(alpha), result_gpu.data)

    return result_gpu.get()

def build_overlap_matrix(grid, centers, alphas):
    """Build the overlap matrix using OpenCL."""
    n_basis = len(centers)
    n_grid = grid.shape[0]
    grid_gpu = cl_array.to_device(queue, grid.astype(np.float32).flatten())
    centers_gpu = cl_array.to_device(queue, np.array(centers, dtype=np.float32).flatten())
    alphas_gpu = cl_array.to_device(queue, np.array(alphas, dtype=np.float32))
    result_gpu = cl_array.zeros(queue, (n_basis, n_basis), dtype=np.float32)

    program.overlap_matrix(queue, (n_basis, n_basis), None, grid_gpu.data, centers_gpu.data, alphas_gpu.data, result_gpu.data, np.int32(n_basis), np.int32(n_grid))

    return result_gpu.get()

def update_density(grid, eigenvectors, centers, alphas):
    """Update density using OpenCL."""
    n_basis = len(centers)
    n_grid = grid.shape[0]
    grid_gpu = cl_array.to_device(queue, grid.astype(np.float32).flatten())
    eigenvectors_gpu = cl_array.to_device(queue, eigenvectors.flatten().astype(np.float32))
    centers_gpu = cl_array.to_device(queue, np.array(centers, dtype=np.float32).flatten())
    alphas_gpu = cl_array.to_device(queue, np.array(alphas, dtype=np.float32))
    density_gpu = cl_array.zeros(queue, n_grid, dtype=np.float32)

    program.density_update(queue, (n_grid,), None, grid_gpu.data, eigenvectors_gpu.data, centers_gpu.data, alphas_gpu.data, density_gpu.data, np.int32(n_basis), np.int32(n_grid))

    density = density_gpu.get()
    density /= np.sum(density)  # Normalize
    return density

def scf_loop(grid, centers, alphas, external_potential, num_electrons, frame_callback):
    """Self-consistent field loop using OpenCL."""
    n_basis = len(centers)
    n_grid = grid.shape[0]
    
    # Initialize density
    density = np.ones(n_grid, dtype=np.float32) * 0.1
    damping_factor = 0.2

    # Flatten grid and centers for OpenCL
    grid_flat = grid.astype(np.float32).flatten()
    centers_flat = np.array(centers, dtype=np.float32).flatten()

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}...")

        # Build Hamiltonian matrix
        hamiltonian = build_overlap_matrix(grid, centers, alphas)
        hamiltonian += np.eye(n_basis)  # Add identity for simplicity in this demo

        # Transfer Hamiltonian to OpenCL
        hamiltonian_gpu = cl_array.to_device(queue, hamiltonian.astype(np.float32))

        # Solve eigenvalues and eigenvectors
        h_eigenvalues, h_eigenvectors = np.linalg.eigh(hamiltonian_gpu.get())
        print(f"Eigenvalues: {h_eigenvalues[:num_electrons]}")

        # Calculate new density
        new_density = update_density(grid, h_eigenvectors[:, :num_electrons], centers, alphas)

        # Apply damping
        density = damping_factor * new_density + (1 - damping_factor) * density

        # Save density for animation
        frame_callback(density)

        # Check for convergence
        if np.sum(np.abs(new_density - density)) < TOLERANCE:
            print(f"Converged in {iteration + 1} iterations.")
            return density, h_eigenvalues

    print("SCF did not converge.")
    return density, None


def create_animation(atomic_number, num_electrons):
    """Create animation of electron density evolution."""
    grid_points = 40
    grid = np.array([(x, y, z) for x in np.linspace(-10, 10, grid_points)
                     for y in np.linspace(-10, 10, grid_points)
                     for z in np.linspace(-10, 10, grid_points)], dtype=np.float32)
    
    centers = np.array([[0, 0, 0]], dtype=np.float32)  # Single atom at origin (this needs to correspond with the potential)
    alphas = [1.0]  # Simple Gaussian width TODO: Use multiple Gaussians for better results

    # External potential for hydrogen atom (pretty good approximation for small Z and to an extent large Z)
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
        threshold = mean + (2*std)  # Display only high-density regions (statistically significant - 2 std)
        ax.voxels(density > threshold, facecolors='blue', edgecolors='gray')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100)

    writer = FFMpegWriter(fps=10, bitrate=1800, extra_args=['-c:v', 'h264_amf'])
    ani.save("electron_density.mp4", writer=writer)
    print("Animation saved as electron_density.mp4.")


if __name__ == "__main__":
    create_animation(atomic_number=6, num_electrons=6)
