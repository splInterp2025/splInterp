# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script generates a comparison showing:
#   • (left) Standard Voronoi diagram
#   • (center) First-order power diagram with linear distance weighting
#   • (right) Second-order power diagram
#
# This progression demonstrates how higher-order power diagrams can capture
# more sophisticated spatial relationships.
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.patches as patches

def power_distance(x, centers, alphas):
    """Calculate power distance from point x to all centers with offsets alphas"""
    return np.sum((x - centers)**2, axis=1) - alphas

def find_cell(x, centers, alphas):
    """Find which cell the point x belongs to"""
    distances = power_distance(x, centers, alphas)
    return np.argmin(distances)

def plot_power_diagram(ax, centers, alphas, title, bounds=(-4, 4, -4, 4), resolution=400):
    """Plot a power diagram"""
    x_min, x_max, y_min, y_max = bounds
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Determine cell for each point in grid
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = find_cell(point, centers, alphas)
    
    # Plot the diagram
    ax.imshow(Z, origin='lower', extent=bounds, aspect='auto', cmap='tab20', 
             interpolation='nearest', alpha=0.5)
    
    # Plot the centers
    ax.scatter(centers[:, 0], centers[:, 1], color='black', s=90, zorder=2)
    
    # Add circles with radius related to sqrt(alpha) to illustrate power
    for i, (center, alpha) in enumerate(zip(centers, alphas)):
        if alpha > 0:
            radius = np.sqrt(alpha)
            circle = patches.Circle(center, radius, fill=False, 
                                   edgecolor='red', linestyle='-', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
        # Add text label with alpha value
        ax.text(center[0], center[1]+0.3, f"α={alpha:.1f}", 
               ha='center', va='bottom', fontsize=15)
    
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(title, fontsize=22)
    ax.grid(alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_second_order_power_diagram(ax, centers, alphas, title, bounds=(-4, 4, -4, 4), resolution=400):
    """Plot a 2nd-order power diagram (each cell is closest to 2 centers)"""
    x_min, x_max, y_min, y_max = bounds
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    k = len(centers)
    num_cells = k * (k-1) // 2  # Number of 2-element subsets
    
    # Determine cell for each point in grid
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            distances = power_distance(point, centers, alphas)
            
            # Get indices of two closest centers
            closest_two = np.argsort(distances)[:2]
            # Assign a unique ID for each pair of centers
            cell_id = closest_two[0] * k + closest_two[1] if closest_two[0] < closest_two[1] else closest_two[1] * k + closest_two[0]
            Z[i, j] = cell_id % 20  # Modulo 20 to fit with colormap
    
    # Plot the diagram
    ax.imshow(Z, origin='lower', extent=bounds, aspect='auto', cmap='tab20', 
             interpolation='nearest', alpha=0.5)
    
    # Plot the centers
    ax.scatter(centers[:, 0], centers[:, 1], color='black', s=90)
    
    # Add circles with radius related to sqrt(alpha)
    for i, (center, alpha) in enumerate(zip(centers, alphas)):
        if alpha > 0:
            radius = np.sqrt(alpha)
            circle = patches.Circle(center, radius, fill=False, 
                                   edgecolor='red', linestyle='-', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            ax.text(center[0], center[1]+0.3, f"α={alpha:.1f}", 
            ha='center', va='bottom', fontsize=15)
    
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(title, fontsize=22)
    ax.grid(alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

# Set up centers and different alpha values
np.random.seed(42)
k = 7  # Number of centers
centers = np.random.rand(k, 2) * 6 - 3  # Random centers in [-3, 3]×[-3, 3]

# Zero alphas for Voronoi
alphas_voronoi = np.zeros(k)

# Variable alphas for power diagram
alphas_power = np.random.rand(k) * 2  # Random alphas in [0, 2]
alphas_power[0] += 4.0  # Make the first center much more powerful

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_power_diagram(axes[0], centers, alphas_voronoi, "Voronoi Diagram\n(all αᵢ = 0)")
plot_power_diagram(axes[1], centers, alphas_power, "Power Diagram\n(variable αᵢ values)")
plot_second_order_power_diagram(axes[2], centers, alphas_power, "2nd-order Power Diagram\n(closest to 2 centers)")

plt.suptitle("Power Diagrams and Voronoi Diagrams", fontsize=28)
plt.tight_layout()
plt.show()