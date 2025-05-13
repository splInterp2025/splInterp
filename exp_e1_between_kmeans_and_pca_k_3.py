# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script visualizes the SAE bridge between 𝑘-means clustering and PCA.
# Configuration: k = 3 (soft).
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.spatial import Voronoi
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def on_key(event):
    if event.key == 'q':
        plt.close('all')
        sys.exit(0)

# Parameters
K = 80  # Number of clusters/dictionary elements
n_points = 100
k_active = 3  # Number of active dictionary elements per input

# Generate data
X = torch.zeros(n_points, 2)

# Cluster 1: Horizontal line 
n1 = int(n_points * 0.4)
X[:n1, 0] = torch.linspace(-1.5, 0, n1) + torch.randn(n1) * 0.1
X[:n1, 1] = -0.8 + torch.randn(n1) * 0.1

# Cluster 2: Dense square 
n2 = int(n_points * 0.3)
X[n1:n1+n2, 0] = torch.rand(n2) * 0.8 - 0.4  # Square from -0.4 to 0.4
X[n1:n1+n2, 1] = torch.rand(n2) * 0.8 + 0.4  # Square from 0.4 to 1.2

# Cluster 3: Scattered diagonal 
n3 = n_points - n1 - n2
t = torch.linspace(0, 1, n3)
X[n1+n2:, 0] = 0.8 + t + torch.randn(n3) * 0.15
X[n1+n2:, 1] = t - 0.5 + torch.randn(n3) * 0.15

resolution = 95
x1 = torch.linspace(-2, 2, resolution)
x2 = torch.linspace(-2, 2, resolution)
grid_points = torch.stack(torch.meshgrid(x1, x2, indexing='ij'), -1).reshape(-1, 2)

# Convert to numpy
X_np = X.detach().numpy()
grid_np = grid_points.detach().numpy()

# Setup for sparse autoencoder
W1 = torch.nn.Parameter(torch.randn(K, 2))
with torch.no_grad():
    # Initialize dictionary elements near data points
    random_indices = torch.randint(0, n_points, (K,))
    W1.data = X[random_indices] + torch.randn(K, 2) * 0.1
W2 = torch.nn.Parameter(torch.randn(2, K))
b = torch.nn.Parameter(torch.randn(K) * 1.0)  # Larger bias for visible polytopes

# Optimizer
optim = torch.optim.Adam([W1, W2, b], lr=8e-3)

plt.ion()  # interactive mode
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.canvas.mpl_connect('key_press_event', on_key)
print(f"Press 'q' to quit - Using top-{k_active} SAE")

# Training Params
num_steps = 7500
visualization_interval = 10

Kr = 3

# Run K-means once at the start
kmeans = KMeans(n_clusters=Kr, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_np)
kmeans_labels = kmeans.predict(X_np)
grid_kmeans_labels = kmeans.predict(grid_np)

# Initialize PCA directions for each cluster
pca_directions = np.zeros((Kr, 2))
for k in range(Kr):
    cluster_points = X_np[kmeans_labels == k]
    if len(cluster_points) >= 2:  # Need at least 2 points for PCA
        # Center the data
        centered = cluster_points - np.mean(cluster_points, axis=0)
        # Skip empty clusters
        if centered.shape[0] > 1:
            # Calculate the covariance matrix
            cov_matrix = np.cov(centered, rowvar=False)
            # Get eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Use the eigenvector with the largest eigenvalue
            pca_directions[k] = eigenvectors[:, -1]

# Pure k-means reconstruction (assign to nearest centroid)
kmeans_reconstruction = np.zeros_like(X_np)
for i in range(n_points):
    kmeans_reconstruction[i] = kmeans.cluster_centers_[kmeans_labels[i]]

# For PCA extension of k-means
pca_reconstructions = np.zeros_like(X_np)
for k in range(Kr):
    cluster_points = X_np[kmeans_labels == k]
    if len(cluster_points) >= 2:
        center = kmeans.cluster_centers_[k]
        for i in np.where(kmeans_labels == k)[0]:
            # Project onto principal component
            centered = X_np[i] - center
            projection = np.dot(centered, pca_directions[k]) * pca_directions[k]
            pca_reconstructions[i] = center + projection
    else:
        # If cluster has < 2 points, just use centroid
        for i in np.where(kmeans_labels == k)[0]:
            pca_reconstructions[i] = kmeans.cluster_centers_[k]

# Compute losses for static plots
kmeans_mse = np.mean((X_np - kmeans_reconstruction) ** 2)
pca_mse = np.mean((X_np - pca_reconstructions) ** 2)

# Temperature parameter for top-k softmax
temperature = 0.3  # Lower = harder selection, higher = softer

for step in range(num_steps):
    # Train the Top-k SAE (modified from Soft SAE)
    W1_norm = W1.square().sum(1).sqrt()
    W2_norm = W2.square().sum(0).sqrt()
    
    code = (X @ W1.T / W1_norm) + b
    
    # Explicit top-k selection instead of softmax
    topk_values, topk_indices = torch.topk(code, k=k_active, dim=1)
    
    # Create sparse weights using only top-k elements
    sparse_weights = torch.zeros_like(code)
    for i in range(X.shape[0]):
        # Apply softmax only to the top-k values
        sparse_weights[i, topk_indices[i]] = torch.softmax(topk_values[i] / temperature, dim=0)
    
    # Compute reconstruction as weighted sum of dictionary elements
    reconstruction = torch.mm(sparse_weights, W2.T / W2_norm.unsqueeze(1))

    loss = torch.nn.functional.mse_loss(reconstruction, X)
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 100 == 0:
        print(f"Step {step}/{num_steps}: Top-{k_active} SAE Loss = {loss.item():.6f}, K-means MSE = {kmeans_mse:.6f}, PCA MSE = {pca_mse:.6f}")
    
    if step % visualization_interval == 0:
        # Clear all axes
        for ax in axes:
            ax.clear()

        with torch.no_grad():
            # 1. K-means visualization
            axes[0].imshow(grid_kmeans_labels.reshape(resolution, resolution),
                         extent=[-2, 2, -2, 2], alpha=0.6, aspect="auto", cmap='tab20',
                         interpolation='nearest')
            
            axes[0].scatter(X_np[:, 0], X_np[:, 1], color='black', alpha=0.8, label="Data Points")
            axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                          color='red', alpha=0.9, marker='o', s=50, label="Centroids")
            
            # Draw lines to show which centroid each point belongs to
            for i in range(n_points):
                center = kmeans.cluster_centers_[kmeans_labels[i]]
                axes[0].plot([X_np[i, 0], center[0]], [X_np[i, 1], center[1]], 
                            'k-', alpha=0.1)
                
            axes[0].set_title(f"K-means (MSE: {kmeans_mse:.6f})")
            axes[0].set_xlim(-2, 2)
            axes[0].set_ylim(-2, 2)
            axes[0].grid(alpha=0.3)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            axes[0].set_xticklabels([])
            axes[0].set_yticklabels([])
            axes[0].legend(loc='upper right')
            
            # 2. Top-k SAE visualization
            Z = (grid_points @ W1.T / W1_norm) + b
            # For visualization, compute top-k for grid points too
            grid_topk_values, grid_topk_indices = torch.topk(Z, k=k_active, dim=1)
            grid_sparse_weights = torch.zeros_like(Z)
            for i in range(grid_points.shape[0]):
                grid_sparse_weights[i, grid_topk_indices[i]] = torch.softmax(grid_topk_values[i] / temperature, dim=0)
            
            # For coloring, use the most active element
            partition_indices = torch.argmax(grid_sparse_weights, dim=1)
            partition_map = partition_indices.reshape(x1.size(0), x2.size(0))
            
            axes[1].imshow(partition_map.detach().cpu().numpy(), extent=[-2, 2, -2, 2],
                         alpha=0.6, aspect="auto", cmap='tab20', interpolation='nearest')
            
            axes[1].scatter(X[:, 0].detach().cpu().numpy(), X[:, 1].detach().cpu().numpy(),
                          color='black', alpha=0.8, label="Data Points")
            
            # Calculate dictionary element usage based on sparse weights
            dict_usage = torch.sum(sparse_weights, dim=0)
            
            min_size = 30
            max_size = 300
            if dict_usage.max() > 0:
                marker_sizes = min_size + (max_size - min_size) * dict_usage / dict_usage.max()
            else:
                marker_sizes = torch.ones(K) * min_size
            
            axes[1].scatter(W1[:, 0].detach().cpu().numpy(), W1[:, 1].detach().cpu().numpy(),
                          color='red', alpha=0.9, marker='x', s=marker_sizes.detach().cpu().numpy(),
                          label="Dictionary Elements")
            
            # Draw lines to show dictionary element usage with line thickness based on weight
            '''
            for i in range(n_points):
                # Use the actual top-k indices for visualization
                for j, idx in enumerate(topk_indices[i]):
                    w = W1[idx].detach().cpu().numpy()
                    weight = sparse_weights[i, idx].item()
                    # Line width and alpha based on weight
                    line_alpha = min(weight * 0.8, 0.15)  # Scale up alpha but cap it
                    axes[1].plot([X_np[i, 0], w[0]], [X_np[i, 1], w[1]], 
                                'k-', alpha=line_alpha)
            '''
            axes[1].set_title(f"Top-{k_active} SAE (Temperature: {temperature}, Loss: {loss.item():.6f})")
            axes[1].set_xlim(-2, 2)
            axes[1].set_ylim(-2, 2)
            axes[1].grid(alpha=0.3)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].set_xticklabels([])
            axes[1].set_yticklabels([])
            axes[1].legend(loc='upper right')
            
            # 3. PCA extension of k-means visualization
            axes[2].imshow(grid_kmeans_labels.reshape(resolution, resolution),
                         extent=[-2, 2, -2, 2], alpha=0.6, aspect="auto", cmap='tab20',
                         interpolation='nearest')
            
            axes[2].scatter(X_np[:, 0], X_np[:, 1], color='black', alpha=0.8, label="Data Points")
            axes[2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                          color='red', alpha=0.9, marker='o', s=50, label="Centroids")
            
            # Draw PCA directions as lines
            for k in range(Kr):  # Use Kr here
                if np.sum(kmeans_labels == k) > 0:  # Only visualize used clusters
                    # Draw principal component as a line through the centroid
                    length = 0.6  # Increased line length for better visibility
                    direction = pca_directions[k] * length
                    center = kmeans.cluster_centers_[k]
                    axes[2].arrow(center[0], center[1], direction[0], direction[1],
                                head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.8)
            
            # Draw lines to show projection onto principal component
            for i in range(n_points):
                center = kmeans.cluster_centers_[kmeans_labels[i]]
                projected = pca_reconstructions[i]
                axes[2].plot([X_np[i, 0], projected[0]], [X_np[i, 1], projected[1]], 
                            'g-', alpha=0.3)
                axes[2].plot([center[0], projected[0]], [center[1], projected[1]], 
                            'b-', alpha=0.3)
            
            axes[2].set_title(f"PCA Extension of K-means (MSE: {pca_mse:.6f})")
            axes[2].set_xlim(-2, 2)
            axes[2].set_ylim(-2, 2)
            axes[2].grid(alpha=0.3)
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            axes[2].set_xticklabels([])
            axes[2].set_yticklabels([])
            axes[2].legend(loc='upper right')
            
        plt.suptitle(f"Step {step}: K-means vs. Top-{k_active} SAE vs. PCA Extension")
        plt.tight_layout()
        plt.pause(0.01)

plt.ioff()
plt.show()