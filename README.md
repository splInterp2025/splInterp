# SplInterp: Sparse Autoencoders through the Lens of Spline Theory

This repository contains the code accompanying the NeurIPS 2025 paper **"SplInterp: Improving our Understanding and Training of Sparse Autoencoders"**.

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