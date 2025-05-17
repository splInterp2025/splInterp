# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script projects 3D Voronoi diagrams onto 2D power diagrams.
# The resulting power cells demonstrate the preservation of topological
# structure through projection.
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Use this instead of PolyCollection
import numpy as np
import torch
from matplotlib.colors import to_rgba

# Parameters
K = 12  # Number of dictionary elements
n_points = 100

# data
np.random.seed(42)
X = torch.zeros(n_points, 2)

# Cluster 1: Horizontal line
n1 = int(n_points * 0.4)
X[:n1, 0] = torch.linspace(-1.5, 0, n1) + torch.randn(n1) * 0.1
X[:n1, 1] = -0.8 + torch.randn(n1) * 0.1

# Cluster 2: Square
n2 = int(n_points * 0.3)
X[n1:n1+n2, 0] = torch.rand(n2) * 0.8 - 0.4
X[n1:n1+n2, 1] = torch.rand(n2) * 0.8 + 0.4

# Cluster 3: Diagonal
n3 = n_points - n1 - n2
t = torch.linspace(0, 1, n3)
X[n1+n2:, 0] = 0.8 + t + torch.randn(n3) * 0.15
X[n1+n2:, 1] = t - 0.5 + torch.randn(n3) * 0.15

# Setup for visualization
X_np = X.detach().numpy()

# Manually position dictionary elements for clarity
dict_positions = np.array([
    [-1.2, -0.8],  
    [-0.6, -0.8],  
    [0.0, -0.8],   
    [-0.2, 0.6],   
    [0.2, 0.6],   
    [-0.2, 0.2],   
    [0.2, 0.2],   
    [0.8, -0.5],   
    [1.2, -0.3],   
    [1.7, 0.0],    
    [-1.0, 0.0],   
    [1.0, 1.0],    
])

# set biases - varied values for better visualization
dict_biases = np.array([1.5, 0.8, 0.3, 1.9, 0.2, 0.6, 1.0, 0.4, 1.2, 0.7, 0.1, 1.8])

# Function to compute power distance for grid visualization
def power_distance(x, centers, biases):
    return np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2) - biases[np.newaxis, :]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 2D grid for power diagram visualization
resolution = 50
x1 = np.linspace(-2, 2, resolution)
x2 = np.linspace(-2, 2, resolution)
X1, X2 = np.meshgrid(x1, x2)
grid_points = np.column_stack((X1.ravel(), X2.ravel()))

# power diagram
distances = power_distance(grid_points, dict_positions, dict_biases)
grid_assignments = np.argmin(distances, axis=1)
grid_assignments = grid_assignments.reshape(resolution, resolution)

# Calculate normalized heights for visualization
# Higher bias = lower height 
bias_min = np.min(dict_biases)
bias_range = np.max(dict_biases) - bias_min
normalized_heights = 1.0 - (dict_biases - bias_min) / bias_range  # Between 0 and 1

# Colors for consistent visualization
colors = plt.cm.tab20(np.linspace(0, 1, 20))

# 1. Draw 2D projection plane
xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
z_plane = np.zeros_like(xx) - 0.2  
ax.plot_surface(xx, yy, z_plane, alpha=0.1, color='gray')

# semi-transparent colored overlay showing the power diagram
for i in range(resolution-1):  # -1 to avoid edge issues
    for j in range(resolution-1):
        color_idx = grid_assignments[i, j] % 20
        color = list(colors[color_idx])
        color[3] = 0.5  # Set alpha low
        
        # four corners of a grid cell
        x = [x1[i], x1[i+1], x1[i+1], x1[i]]
        y = [x2[j], x2[j], x2[j+1], x2[j+1]]
        z = [-0.19, -0.19, -0.19, -0.19]  # Slightly above the plane
        
        # vertices for a single polygon (quad)
        vertices = [[x[0], y[0], z[0]], 
                   [x[1], y[1], z[1]], 
                   [x[2], y[2], z[2]], 
                   [x[3], y[3], z[3]]]
        
        poly = Poly3DCollection([vertices])
        poly.set_facecolor(color)
        poly.set_edgecolor(None)
        ax.add_collection3d(poly)

# 2. lifted dictionary elements in 3D
for i in range(len(dict_positions)):
    pos = dict_positions[i]
    height = normalized_heights[i]
    color_idx = i % 20
    color = colors[color_idx]
    
    # Plot lifted point
    ax.scatter(pos[0], pos[1], height, color=color, s=100, marker='o', edgecolors='black', zorder=3)
    
    # vertical line connecting to ground projection
    ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [height, -0.2], 
           color=color, linestyle='--', alpha=0.7, zorder=2)
    
    # the projection point on the ground plane
    ax.scatter(pos[0], pos[1], -0.2, color=color, s=100, marker='x', alpha=1.0, linewidth=2, edgecolors='black',zorder=2)
    
    ax.text(pos[0], pos[1], height + 0.05, f"α={dict_biases[i]:.1f}", 
           fontsize=15, ha='center', zorder=4)

# 3. Visualize some power cells in 3D
if len(dict_positions) >= 4:
    try:
        # For each pair of nearby dictionary elements, visualize their boundary
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(dict_positions, dict_positions)
        np.fill_diagonal(dist_matrix, np.inf)
        
        for i in range(len(dict_positions)):
            # Find 2-3 nearest neighbors
            nearest = np.argsort(dist_matrix[i])[:3]
            
            for j in nearest:
                # Draw line connecting points in 3D
                ax.plot([dict_positions[i,0], dict_positions[j,0]],
                       [dict_positions[i,1], dict_positions[j,1]],
                       [normalized_heights[i], normalized_heights[j]],
                       'k-', alpha=0.3, zorder=1)
                       
                # For select pairs, visualize the power diagram boundary
                if i < j:  # Avoid duplicates
                    # Get biases
                    bias_i = dict_biases[i]
                    bias_j = dict_biases[j]
                    
                    # Calculate midpoint between the two centers
                    midpoint = (dict_positions[i] + dict_positions[j]) / 2
                    
                    # Calculate direction from i to j
                    direction_ij = dict_positions[j] - dict_positions[i]
                    direction_ij_norm = np.linalg.norm(direction_ij)
                    
                    # Normalized direction vector
                    if direction_ij_norm > 0:
                        direction_ij = direction_ij / direction_ij_norm
                    
                    # Calculate shift due to bias difference
                    # The boundary shifts toward the center with higher bias
                    bias_diff = bias_j - bias_i
                    shift = bias_diff / (2 * direction_ij_norm) if direction_ij_norm > 0 else 0
                    
                    # Adjust midpoint based on biases
                    adjusted_midpoint = midpoint + shift * direction_ij
                    
                    # Calculate perpendicular direction
                    perp_direction = np.array([-direction_ij[1], direction_ij[0]])
                    
                    # Draw perpendicular line at adjusted midpoint
                    line_length = 0.8
                    start = adjusted_midpoint - perp_direction * line_length
                    end = adjusted_midpoint + perp_direction * line_length
                    
                    # Use blended colors for the boundary
                    color_i = colors[i % 20]
                    color_j = colors[j % 20]
                    blended_color = [(color_i[0]+color_j[0])/2, (color_i[1]+color_j[1])/2, 
                                    (color_i[2]+color_j[2])/2, 0.5]
                                    
    
    except Exception as e:
        print(f"Skipping some 3D visualizations due to error: {e}")

# 4. Draw some example "equidistant" 3D vectors
for p_idx in range(len(dict_positions)):
    center = dict_positions[p_idx]
    height = normalized_heights[p_idx]
    radius = np.sqrt(dict_biases[p_idx] - bias_min) * 0.4  # Scale for visibility
    
    # distance circle in 2D
    theta = np.linspace(0, 2*np.pi, 50)
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    z_circle = np.ones_like(theta) * -0.19
    ax.plot(x_circle, y_circle, z_circle, color=colors[p_idx % 20], linewidth=2, alpha=0.9, zorder=2)
    
    # a few 3D distance vectors
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        point_2d = center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        # the 3D distance vector
        ax.plot([center[0], point_2d[0]], [center[1], point_2d[1]], 
               [height, -0.19], color=colors[p_idx % 20], linestyle='-', 
               alpha=0.7, linewidth=1.5, zorder=2)

# data points on the projection plane
ax.scatter(X_np[:, 0], X_np[:, 1], np.ones_like(X_np[:, 0]) * -0.2,
          color='black', s=15, alpha=0.5, zorder=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height (Higher Centers = Lower Bias)',fontsize=18)
ax.set_title('Power Diagram as Projection of 3D Voronoi Diagram\nHigher Points → Lower Bias (α)', fontsize=22)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-0.3, 1.2)

ax.text2D(0.5, 0.01, 
         "The power diagram (colored regions) is the projection of a Voronoi diagram in 3D\n"
         "where centers are lifted based on their bias values (α)", 
         transform=ax.transAxes, ha='center', fontsize=18, 
         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

ax.view_init(elev=25, azim=-35)

plt.tight_layout()
plt.show()