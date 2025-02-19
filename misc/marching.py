import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Reinsert as required - ruining github statistics
density_string = ""

# Convert string to a NumPy array
density_values = np.array([float(val) for val in density_string.split(", ")], dtype=np.float32)

# Define the grid size (adjust based on your dataset)
grid_points = int(round(len(density_values) ** (1/3)))  # Assuming a cubic grid
if grid_points ** 3 != len(density_values):
    raise ValueError("Density values do not fit into a perfect 3D cube.")

# Reshape density values into a 3D grid
density_grid = density_values.reshape((grid_points, grid_points, grid_points))

# Create figure and 3D axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Final Electron Density Distribution")

# Define a threshold for visualization (e.g., mean + 2 std deviations)
mean = np.mean(density_grid)
std = np.std(density_grid)
threshold = mean + (2 * std)

# Plot voxels where density is above the threshold
ax.voxels(density_grid > threshold, facecolors='blue', edgecolors='gray')

# Show the plot
plt.savefig("final_density_distribution.png")