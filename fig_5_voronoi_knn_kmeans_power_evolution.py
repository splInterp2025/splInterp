# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script illustrates spatial partitioning methods:
#   (a) Standard Voronoi diagram where space is partitioned based on the nearest
#       generator point using Euclidean distance.
#   (b) Nearest-neighbor (𝑘 = 1) classification showing how Voronoi cells define
#       decision boundaries for point classification.
#   (c) 𝑘-means clustering with 𝑘 = 3 demonstrating how cluster centroids generate
#       Voronoi cells that define cluster boundaries.
#   (d) Power diagram (weighted Voronoi) where each generator has an associated
#       weight, creating curved boundaries between regions.
#
# Power diagrams generalize Voronoi diagrams and provide additional flexibility
# for modeling spatial relationships.
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
from scipy.spatial import Voronoi
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

fig, axs = plt.subplots(2, 2, figsize=(13, 11))
axs = axs.flatten()

# random 2D points for the first three plots
num_points = 12
points = np.random.rand(num_points, 2) * 8 - 4  

# Assign random classes for k-NN visualization
classes = np.random.choice([0, 1, 2], size=num_points)

# Assign random weights for power diagram
weights = np.random.uniform(0.5, 2.5, size=num_points)

# Generate a grid of points to visualize decision boundaries
x_min, x_max = -4.5, 4.5
y_min, y_max = -4.5, 4.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Colors for visualization
class_colors = ['#ff7f0e', '#1f77b4', '#2ca02c']  
point_colors = [class_colors[c] for c in classes]
cmap = plt.cm.viridis

# VORONOI DIAGRAM 
ax = axs[0]
ax.set_title("a) Standard Voronoi Diagram", fontsize=22, pad=24)  

# Compute Voronoi diagram
# Add far away points to close the regions
far_points = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]])
vor = Voronoi(np.vstack([points, far_points]))

# Plot Voronoi regions
for i, point in enumerate(points):
    region_idx = vor.point_region[i]
    region = vor.regions[region_idx]
    
    if -1 not in region:  # If the region is closed
        polygon = [vor.vertices[v] for v in region]
        ax.fill(*zip(*polygon), alpha=0.3, color=cmap(i / num_points))
        ax.plot(*zip(*polygon + [polygon[0]]), 'k--', alpha=0.3)

# Plot the points
ax.scatter(points[:, 0], points[:, 1], c=[cmap(i / num_points) for i in range(num_points)], 
          s=80, edgecolors='k', zorder=10)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
#ax.set_xlabel("X", fontsize=14)
#ax.set_ylabel("Y", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])

ax.text(-4.33, 3.0, "• Each point owns the region closest to it\n• Creates cell boundaries that are equidistant\n  between neighboring points",
       fontsize=16, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'), zorder=20)

# K-NN (k=1) 
ax = axs[1]
ax.set_title("b) K-Nearest Neighbors (k=1)", fontsize=22, pad=24)  # Increased padding

# Train a KNN model with k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(points, classes)

# Predict on grid to visualize decision boundaries
Z = knn.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot decision regions
ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.brg)

# Plot the training points
scatter = ax.scatter(points[:, 0], points[:, 1], c=classes, 
                    cmap=plt.cm.brg, s=80, edgecolors='k', zorder=10)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
#ax.set_xlabel("X", fontsize=14)
#ax.set_ylabel("Y", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])


legend1 = ax.legend(*scatter.legend_elements(),
                   loc="lower right", title="Classes")
ax.add_artist(legend1)

ax.text(-4.33, 3.0, "• Assigns each point to nearest neighbor\n• With k=1, the boundaries form a Voronoi diagram\n• Colors represent different classes (e.g., categories)",
       fontsize=16, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'), zorder=20)

# K-MEANS 
ax = axs[2]
ax.set_title("c) K-means Clustering (k=3)", fontsize=22, pad=24)  # Increased padding

# Generate new random data (more points) for K-means
num_data_points = 200
data = np.random.randn(num_data_points, 2)  # Standard normal distribution
data[:num_data_points//3, :] = data[:num_data_points//3, :] - 2  # Shift 1/3 of points
data[num_data_points//3:2*num_data_points//3, :] = data[num_data_points//3:2*num_data_points//3, :] + np.array([2, -2])  # Shift 1/3 of points

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)
centroids = kmeans.cluster_centers_

# Predict on grid to visualize decision boundaries
kmeans_labels = kmeans.predict(grid_points)
kmeans_labels = kmeans_labels.reshape(xx.shape)

# Plot decision regions
ax.contourf(xx, yy, kmeans_labels, alpha=0.3, cmap=plt.cm.brg)

# Plot the data points
ax.scatter(data[:, 0], data[:, 1], c=clusters, cmap=plt.cm.brg, 
          s=30, alpha=0.7, edgecolors='none')

# Plot the centroids
ax.scatter(centroids[:, 0], centroids[:, 1], c=range(3), cmap=plt.cm.brg,
          s=180, marker='*', edgecolors='k', linewidth=2, zorder=10)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
#ax.set_xlabel("X", fontsize=14)
#ax.set_ylabel("Y", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])


ax.text(-4.33, 3.0, "• Finds optimal centers (stars) given data\n• Cell boundaries form a Voronoi diagram\n• Centers ≠ data points (unlike k-NN)",
       fontsize=16, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'), zorder=20)

# POWER DIAGRAM 
ax = axs[3]
ax.set_title("d) Power Diagram", fontsize=22, pad=24)  # Increased padding

# Compute power distance - squared Euclidean distance minus weight
def power_distance(x, centers, weights):
    return np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2) - weights[np.newaxis, :]

# Compute power diagram (closest point by power distance)
distances = power_distance(grid_points, points, weights)
power_assignments = np.argmin(distances, axis=1)
power_assignments = power_assignments.reshape(xx.shape)

# Create a colormap for the power diagram
power_cmap = plt.cm.viridis

# Plot the power diagram
ax.contourf(xx, yy, power_assignments, alpha=0.3, cmap=power_cmap, levels=np.arange(num_points+1)-0.5)

# Show the circles representing the weights
for i, (point, weight) in enumerate(zip(points, weights)):
    # Circle radius based on weight
    radius = np.sqrt(weight) * 0.4  # Scale for visualization
    circle = Circle(point, radius, fill=False, color=cmap(i / num_points), linewidth=1.5)
    ax.add_patch(circle)

# Make one circle stand out more
special_index = 2  # Choose which point to highlight
special_point = points[special_index]
special_weight = weights[special_index]
special_radius = np.sqrt(special_weight) * 0.4
special_circle = Circle(special_point, special_radius, fill=False, 
                       color='red', linewidth=2.5, linestyle='-')
ax.add_patch(special_circle)

# Plot the center points
ax.scatter(points[:, 0], points[:, 1], c=[cmap(i / num_points) for i in range(num_points)], 
          s=80, edgecolors='k', zorder=10)

# Add text annotations for weights - now for more points (every other point)
for i, (point, weight) in enumerate(zip(points, weights)):
    if i % 2 == 0:  # Changed from i % 3 == 0 to show more weight labels
        ax.text(point[0], point[1] + 0.35, f"w={weight:.1f}", 
               ha='center', va='center', fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.2'), zorder=15)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
#ax.set_xlabel("X", fontsize=14)
#ax.set_ylabel("Y", fontsize=14)
ax.set_xticks([])
ax.set_yticks([])


ax.text(-4.33, 2.7, "• Generalizes Voronoi diagrams with weights/biases\n• Similar to how GMMs generalize k-means\n• Weights act as 'priors' over regions\n• Equivalent to projecting a higher-D Voronoi",
       fontsize=16, bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'), zorder=20)

plt.suptitle("From Voronoi to Power Diagrams", 
             fontsize=28, y=0.99)  # Moved title up slightly

plt.tight_layout()

plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.25, wspace=0.2)  

plt.show()
