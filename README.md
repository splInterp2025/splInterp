# SplInterp: Sparse Autoencoders through the Lens of Spline Theory

This repository contains the code accompanying the paper **"SplInterp: Improving our Understanding and Training of Sparse Autoencoders"**.

## Overview

Sparse Autoencoders (SAEs) are widely used in mechanistic interpretability, but their theoretical underpinnings remain underexplored. This work reframes SAEs using **spline theory**, yielding both new insights and practical algorithms:

- **Theory**: We show that SAEs are piecewise affine splines and can be geometrically characterized via **power diagrams**. TopK SAEs in particular correspond exactly to **K-th order power diagrams**.
- **Connection to ML classics**: SAEs are shown to generalize **k-means** and connect with **PCA**, sacrificing some reconstruction accuracy for interpretability.
- **Training Innovation**: We propose **PAM-SGD**, a Proximal Alternating Method for training SAEs. It alternates between SGD updates for encoding and closed-form decoding updates, offering superior **sample efficiency** and **code sparsity** vs. standard SGD—especially in low-data settings.

## Features


- PAM-SGD training algorithm with theoretical backing
- Experiments on:
  - MNIST for visual representation learning
  - Gemma-2-2B LLM activations
- Visualizations of power diagrams and spline regions

## Installation

First, install the required Python libraries:

```
pip install torch numpy matplotlib tqdm scikit-learn scipy pyvista

```
---

## File Descriptions

- **exp_e2_MNIST_SGD_PAM_ablations.py**  
  Compares standard SGD and PAM-SGD (Algorithm 3.1) for training sparse autoencoders on MNIST. Includes ablation studies for both TopK and ReLU activations.

- **exp_e3_LLM_SGD_PAM_ablations.py**  
  Compares SGD and PAM-SGD for training sparse autoencoders on high-dimensional LLM activations (Gemma-2-2B). Supports extensive ablations and visualization.

- **fig_5_voronoi_knn_kmeans_power_evolution.py**  
  Illustrates spatial partitioning: standard Voronoi diagrams, k-nearest neighbor classification, k-means clustering, and power diagrams. Shows how power diagrams generalize Voronoi diagrams.

- **fig_6_voronoi_to_power.py**  
  Generates a visual comparison of standard Voronoi diagrams and first/second-order power diagrams, highlighting the progression to more flexible spatial partitions.

- **fig_6_voronoi_to_power_variation.py**  
  Variant of the above, providing additional visualizations of the transition from Voronoi to power diagrams.

- **fig_7_voronoi_3d_to_power_2d.py**  
  Projects 3D Voronoi diagrams onto 2D power diagrams, demonstrating the preservation of topological structure through projection.

- **fig_7_voronoi_3d_to_power_2d_variation.py**  
  Variation of the above, with alternative visualizations for 3D-to-2D Voronoi-to-power diagram projections.

- **exp_e1_between_kmeans_and_pca_k_1.py**  
  Visualizes the connection between k-means clustering (k=1) and PCA using sparse autoencoders. Demonstrates how SAEs interpolate between these two classic methods.

- **exp_e1_between_kmeans_and_pca_k_3.py**  
  Similar to the above, but for k=3 (soft assignment). Shows the transition from k-means to PCA in a multi-cluster setting.

- **exp_e1_between_kmeans_and_pca_k_3hard.py**  
  Like the previous, but with hard assignment for k=3. Useful for understanding the effect of hard vs. soft clustering in the SAE framework.

---

For more details on each experiment or script, see the comments at the top of each file. The scripts are self-contained and can be run directly after installing the dependencies.
