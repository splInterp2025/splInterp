# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script compares the benefits of our PAM-SGD method (Algorithm 3.1) versus
# standard SGD for training sparse autoencoders (SAEs) on simple visual domains,
# using the MNIST dataset.
#
# Instructions: Running this file will perform all the ablations first for TopK
# and then for ReLU. Results will be produced in different subfolders.
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------
# In the comments, PAM-SGD is sometimes abbreviated as EM for brevity.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
import time
import os
from datetime import datetime

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
batch_size = 128
em_batch_size = 1024
learning_rate = 0.003
num_epochs = 50
k = 15  
latent_dim = 256 
image_size = 28 * 28  

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

em_train_loader = DataLoader(
    train_dataset,
    batch_size=em_batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Base model architecture (shared between SGD and PAM-SGD) 
class SparseAutoencoderBase(nn.Module):
    def __init__(self, input_dim, latent_dim, k=15, activation_type="topk"):
        super(SparseAutoencoderBase, self).__init__()
        # Initialize weights with small random values
        self.encoder_weights = nn.Parameter(
            torch.randn(input_dim, latent_dim) * 0.1
        )
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.activation_type = activation_type
        
    def encode(self, x):
        # Linear encoder (no activation function as per the paper)
        return x @ self.encoder_weights + self.encoder_bias
    
    def apply_sparsity(self, z):
        if self.activation_type == "topk":
            # Apply k-sparsity: keep only the k largest activations per sample
            values, indices = torch.topk(torch.abs(z), k=self.k, dim=1)
            mask = torch.zeros_like(z).scatter_(1, indices, 1)
            return z * mask
        elif self.activation_type == "relu":
            return torch.relu(z)
        else:
            raise ValueError(f"Unknown activation_type: {self.activation_type}")
        
    def decode(self, z):
        # Tied weights: use transpose of encoder weights
        return z @ self.encoder_weights.t() + self.decoder_bias
    
    def forward(self, x):
        # Flatten the input
        x_flat = x.view(x.size(0), -1)
        
        # Encode (linear)
        z = self.encode(x_flat)
        
        # Apply sparsity (TopK or ReLU)
        z_sparse = self.apply_sparsity(z)
        
        recon = self.decode(z_sparse)
               
        return recon, z_sparse

# SGD Version (standard autodiff) 
class SGDAutoencoder(SparseAutoencoderBase):
    def __init__(self, input_dim, latent_dim, k=15, activation_type="topk"):
        super(SGDAutoencoder, self).__init__(input_dim, latent_dim, k, activation_type)

# EM Version 
class EMAutoencoder(SparseAutoencoderBase):
    def __init__(self, input_dim, latent_dim, k=15, activation_type="topk"):
        super(EMAutoencoder, self).__init__(input_dim, latent_dim, k, activation_type)
        # For PAM-SGD, handle the decoder weights separately (not as Parameter)
        self.decoder_weights = torch.randn(latent_dim, input_dim, device=device) * 0.01
    
    def decode(self, z):
        # explicitly maintained decoder weights instead of tied weights
        return z @ self.decoder_weights + self.decoder_bias
    
    def update_decoder_analytically(self, data_loader, num_batches=None):
        """
        Solve for optimal decoder weights using the pseudo-inverse
        W_dec = (Z^T Z)^(-1) Z^T X
        """
        self.eval()  # Set to eval mode for consistent sparsity application
        
        # Collect activations and targets
        all_z_sparse = []
        all_inputs = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if num_batches is not None and i >= num_batches:
                    break
                
                data = data.to(device)
                x_flat = data.view(data.size(0), -1)
                z = self.encode(x_flat)
                z_sparse = self.apply_sparsity(z)
                
                all_z_sparse.append(z_sparse)
                all_inputs.append(x_flat)
        
        # Concatenate all batches
        Z = torch.cat(all_z_sparse, dim=0)  # [N, latent_dim]
        X = torch.cat(all_inputs, dim=0)     # [N, input_dim]
        
        # Compute Z^T Z and Z^T X
        ZTZ = Z.t() @ Z
        ZTX = Z.t() @ X
        
        # Add small regularization for numerical stability
        reg = 1e-6 * torch.eye(ZTZ.shape[0], device=device)
        ZTZ_reg = ZTZ + reg
        
        # Solve for optimal weights using pseudo-inverse
        # W_dec = (Z^T Z)^(-1) Z^T X
        self.decoder_weights = torch.linalg.solve(ZTZ_reg, ZTX)
        
        # Update bias term (mean residual)
        pred = Z @ self.decoder_weights
        self.decoder_bias.data = torch.mean(X - pred, dim=0)

# ===== Training Functions =====

# SGD training loop
def train_sgd(model, train_loader, test_loader, optimizer, criterion, epochs):
    train_losses = []
    test_losses = []
    recon_examples = []
    
    # some fixed test examples for visualization
    test_examples, _ = next(iter(test_loader))
    test_examples = test_examples[:10].to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            x_flat = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            recon, _ = model(data)
            loss = criterion(recon, x_flat)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'SGD - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                x_flat = data.view(data.size(0), -1)
                recon, _ = model(data)
                loss = criterion(recon, x_flat)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        time_elapsed = time.time() - start_time
        
        print(f'SGD - Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, Time: {time_elapsed:.2f}s')
        
        # Save reconstruction examples
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                recon, _ = model(test_examples)
                recon_examples.append((epoch, deepcopy(recon.detach())))
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "recon_examples": recon_examples
    }

# PAM-SGD training loop with configurable SGD updates per batch
def train_em(model, train_loader, em_train_loader, test_loader, encoder_optimizer, criterion, epochs, sgd_updates_per_batch=1):
    train_losses = []
    test_losses = []
    recon_examples = []
    
    # some fixed test examples for visualization
    test_examples, _ = next(iter(test_loader))
    test_examples = test_examples[:10].to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        # E-step: Optimize encoder with fixed decoder (use regular train_loader)
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            x_flat = data.view(data.size(0), -1)
            
            # Perform multiple SGD updates on the same batch
            for update_idx in range(sgd_updates_per_batch):
                encoder_optimizer.zero_grad()
                recon, _ = model(data)
                loss = criterion(recon, x_flat)
                loss.backward()
                encoder_optimizer.step()
                
                # Only record loss for the last update
                if update_idx == sgd_updates_per_batch - 1:
                    epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'PAM-SGD (E-step) - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Updates/batch: {sgd_updates_per_batch}, Loss: {loss.item():.6f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # M-step: Update decoder weights analytically (em_train_loader with larger batch size)
        print(f'PAM-SGD (M-step) - Epoch {epoch+1}/{epochs}, Computing analytic decoder solution...')
        model.update_decoder_analytically(em_train_loader, num_batches=None)
        
        # Evaluate after M-step
        model.eval()
        train_loss_after_m = 0
        test_loss = 0
        
        # Re-check training loss after M-step
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device)
                x_flat = data.view(data.size(0), -1)
                recon, _ = model(data)
                loss = criterion(recon, x_flat)
                train_loss_after_m += loss.item()
        
        avg_train_loss_after_m = train_loss_after_m / len(train_loader)
        train_losses.append(avg_train_loss_after_m)
        
        # Check test loss
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                x_flat = data.view(data.size(0), -1)
                recon, _ = model(data)
                loss = criterion(recon, x_flat)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        time_elapsed = time.time() - start_time
        
        print(f'PAM-SGD - Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss_after_m:.6f} (before M-step: {avg_train_loss:.6f}), '
              f'Test Loss: {avg_test_loss:.6f}, Time: {time_elapsed:.2f}s')
        
        # Save reconstruction examples
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                recon, _ = model(test_examples)
                recon_examples.append((epoch, deepcopy(recon.detach())))
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "recon_examples": recon_examples
    }

# Visualization utilities

def create_plots_folder():
    # Always use a subfolder inside 'plots' (for consistency with LLM results)
    base_folder = "plots"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(base_folder, timestamp)
    os.makedirs(subfolder, exist_ok=True)
    print(f"Plots will be saved to: {subfolder}")
    return subfolder

# Plot loss curves
def plot_losses(sgd_results, em_results, output_folder):
    plt.figure(figsize=(12, 6))
    
    plt.plot(sgd_results["train_losses"], 'b-', label='SGD Train')
    plt.plot(sgd_results["test_losses"], 'b--', label='SGD Test')
    plt.plot(em_results["train_losses"], 'r-', label='PAM-SGD Train')
    plt.plot(em_results["test_losses"], 'r--', label='PAM-SGD Test')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Comparison')
    plt.legend()
    plt.grid(True)
    # Ensure output directory exists before saving
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'loss_comparison.png'))
    plt.close()  # Close instead of show

# ===== Weight Decay and Mu/Nu Ablation Functions =====

def run_weight_decay_ablation(output_folder, activation_type="topk"):
    """Compare different weight decay values for both SGD and PAM-SGD"""
    weight_decay_values = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    sgd_results = []
    em_results = []
    for wd in weight_decay_values:
        print(f"\n===== Training with weight decay {wd} (activation: {activation_type}) =====")
        # SGD
        sgd_model = SGDAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        # Separate weight decay for weights and no decay for bias
        sgd_optimizer = optim.Adam([
            {'params': sgd_model.encoder_weights, 'weight_decay': wd},
            {'params': sgd_model.encoder_bias, 'weight_decay': 0.0},
            {'params': sgd_model.decoder_bias, 'weight_decay': 0.0},
        ], lr=learning_rate)
        criterion = nn.MSELoss()
        sgd_res = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, num_epochs)
        sgd_results.append((wd, sgd_res))
        # PAM-SGD
        em_model = EMAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        em_optimizer = optim.Adam([
            {'params': em_model.encoder_weights, 'weight_decay': wd},
            {'params': em_model.encoder_bias, 'weight_decay': 0.0},
        ], lr=learning_rate)
        em_res = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, num_epochs)
        em_results.append((wd, em_res))
    # Plot final test loss vs weight decay
    plt.figure(figsize=(10, 6))
    wd_labels = [str(wd) for wd, _ in sgd_results]
    sgd_losses = [r["test_losses"][-1] for _, r in sgd_results]
    em_losses = [r["test_losses"][-1] for _, r in em_results]
    x = np.arange(len(weight_decay_values))
    width = 0.35
    plt.bar(x - width/2, sgd_losses, width, label='SGD')
    plt.bar(x + width/2, em_losses, width, label='PAM-SGD')
    plt.xlabel('Weight Decay')
    plt.ylabel('Final Test Loss')
    plt.title(f'Effect of Weight Decay on Model Performance ({activation_type})')
    plt.xticks(x, wd_labels)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_folder, f'weight_decay_comparison_{activation_type}.png'))
    plt.close()
    return sgd_results, em_results

def run_mu_nu_ablation(output_folder, activation_type="topk"):
    """Compare different mu/nu values for both SGD and PAM-SGD (encoder L2 regularization)"""
    mu_nu_values = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    sgd_results = []
    em_results = []
    for mu_nu in mu_nu_values:
        print(f"\n===== Training with mu/nu {mu_nu} (activation: {activation_type}) =====")
        # SGD
        sgd_model = SGDAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        # Add L2 penalty manually in loss
        def sgd_loss_fn(recon, x_flat, model=sgd_model):
            base = criterion(recon, x_flat)
            l2 = mu_nu * (torch.norm(model.encoder_weights) ** 2 + torch.norm(model.encoder_bias) ** 2)
            return base + l2
        sgd_res = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, sgd_loss_fn, num_epochs)
        sgd_results.append((mu_nu, sgd_res))
        # PAM-SGD
        em_model = EMAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=learning_rate)
        def em_loss_fn(recon, x_flat, model=em_model):
            base = criterion(recon, x_flat)
            l2 = mu_nu * (torch.norm(model.encoder_weights) ** 2 + torch.norm(model.encoder_bias) ** 2)
            return base + l2
        em_res = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, em_loss_fn, num_epochs)
        em_results.append((mu_nu, em_res))
    # Plot final test loss vs mu/nu
    plt.figure(figsize=(10, 6))
    mu_nu_labels = [str(mu) for mu, _ in sgd_results]
    sgd_losses = [r["test_losses"][-1] for _, r in sgd_results]
    em_losses = [r["test_losses"][-1] for _, r in em_results]
    x = np.arange(len(mu_nu_values))
    width = 0.35
    plt.bar(x - width/2, sgd_losses, width, label='SGD')
    plt.bar(x + width/2, em_losses, width, label='PAM-SGD')
    plt.xlabel('Mu/Nu (L2 Penalty)')
    plt.ylabel('Final Test Loss')
    plt.title(f'Effect of Mu/Nu (L2) on Model Performance ({activation_type})')
    plt.xticks(x, mu_nu_labels)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_folder, f'mu_nu_comparison_{activation_type}.png'))
    plt.close()
    return sgd_results, em_results


# Plot reconstructions over time
def plot_reconstruction_comparison(sgd_results, em_results, test_examples, output_folder):
    # Get reconstruction examples from results
    sgd_recons = sgd_results["recon_examples"]
    em_recons = em_results["recon_examples"]
    
    # Get the latest epoch reconstructions
    latest_sgd_epoch, latest_sgd_recons = sgd_recons[-1]
    latest_em_epoch, latest_em_recons = em_recons[-1]
    
    # Create a figure with 3 rows: original, SGD recon, PAM-SGD recon
    # Use 5 columns for 5 different samples
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Get labels for the test examples for better identification
    _, labels = next(iter(test_loader))
    labels = labels[:5]
    
    # Row 0: Original images
    for i in range(5):
        orig_img = test_examples[i].cpu().view(28, 28).numpy()
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f"Original Digit: {labels[i].item()}")
        axes[0, i].axis('off')
    
    # Row 1: SGD reconstructions (different examples)
    for i in range(5):
        # clamping for proper visualization
        recon_img = torch.clamp(latest_sgd_recons[i].cpu(), 0, 1).view(28, 28).numpy()
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title(f"SGD Recon {i+1}")
        axes[1, i].axis('off')
    
    # Row 2: EM reconstructions (different examples)
    for i in range(5):
        # clamping for proper visualization
        recon_img = torch.clamp(latest_em_recons[i].cpu(), 0, 1).view(28, 28).numpy()
        axes[2, i].imshow(recon_img, cmap='gray')
        axes[2, i].set_title(f"PAM-SGD Recon {i+1}")
        axes[2, i].axis('off')
    
    # Add row labels
    fig.text(0.05, 0.75, 'Original', fontsize=14, rotation='vertical')
    fig.text(0.05, 0.5, 'SGD', fontsize=14, rotation='vertical')
    fig.text(0.05, 0.25, 'PAM-SGD', fontsize=14, rotation='vertical')
    
    plt.suptitle(f'Reconstruction Comparison (Epoch {latest_sgd_epoch+1})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.9)
    plt.savefig(os.path.join(output_folder, 'reconstruction_comparison.png'))
    plt.close()  # Close instead of show
    
    # progression view for the first sample
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Select equally spaced epochs
    display_epochs = np.linspace(0, len(sgd_recons)-1, 5, dtype=int)
    
    # SGD progression over epochs
    for col, epoch_idx in enumerate(display_epochs):
        epoch, recons = sgd_recons[epoch_idx]
        recon_img = torch.clamp(recons[0].cpu(), 0, 1).view(28, 28).numpy()
        axes[0, col].imshow(recon_img, cmap='gray')
        axes[0, col].set_title(f"SGD (Epoch {epoch+1})")
        axes[0, col].axis('off')
    
    # EM progression over epochs
    for col, epoch_idx in enumerate(display_epochs):
        epoch, recons = em_recons[epoch_idx]
        recon_img = torch.clamp(recons[0].cpu(), 0, 1).view(28, 28).numpy()
        axes[1, col].imshow(recon_img, cmap='gray')
        axes[1, col].set_title(f"PAM-SGD (Epoch {epoch+1})")
        axes[1, col].axis('off')
    
    plt.suptitle(f'Reconstruction Progress (First Sample)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(output_folder, 'reconstruction_progress.png'))
    plt.close()  # Close instead of show

    
# Visualize encoder and decoder weights as filters
def visualize_filters(sgd_model, em_model, output_folder, num_filters=10):
    plt.figure(figsize=(16, 12))
    
    # fewer filters to fit in the grid
    filter_indices = range(num_filters)
    
    # SGD encoder filters (top row)
    for i, idx in enumerate(filter_indices):
        plt.subplot(4, num_filters, i+1)
        filter_img = sgd_model.encoder_weights.data[:, idx].cpu().view(28, 28).numpy()
        plt.imshow(filter_img, cmap='viridis')
        plt.title(f"SGD Enc {idx+1}")
        plt.axis('off')
    
    # EM encoder filters (second row)
    for i, idx in enumerate(filter_indices):
        plt.subplot(4, num_filters, i+1+num_filters)
        filter_img = em_model.encoder_weights.data[:, idx].cpu().view(28, 28).numpy()
        plt.imshow(filter_img, cmap='viridis')
        plt.title(f"PAM-SGD Enc {idx+1}")
        plt.axis('off')
    
    # SGD decoder filters (third row) - tied weights transposed
    for i, idx in enumerate(filter_indices):
        plt.subplot(4, num_filters, i+1+num_filters*2)
        filter_img = sgd_model.encoder_weights.data[:, idx].cpu().view(28, 28).numpy()
        plt.imshow(filter_img, cmap='viridis')  
        plt.title(f"SGD Dec {idx+1}")
        plt.axis('off')
    
    # EM decoder filters (bottom row) - analytically computed
    for i, idx in enumerate(filter_indices):
        plt.subplot(4, num_filters, i+1+num_filters*3)
        filter_img = em_model.decoder_weights[idx, :].cpu().view(28, 28).numpy()
        plt.imshow(filter_img, cmap='viridis')
        plt.title(f"PAM-SGD Dec {idx+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'filter_comparison.png'))
    plt.close()  # Close instead of show

# Run experiments with different training set sizes
def run_training_size_ablation(output_folder, activation_type="topk"):
    # Training set sizes to try (fraction of full dataset)
    sizes = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    sgd_results = []
    em_results = []
    
    for size in sizes:
        print(f"\n===== Training with {size*100:.1f}% of data =====")
        # Create subset of training data
        subset_size = int(len(train_dataset) * size)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        subset = torch.utils.data.Subset(train_dataset, indices)
        
        # Create data loaders with appropriate batch sizes
        subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        subset_em_loader = DataLoader(subset, batch_size=min(em_batch_size, subset_size), shuffle=True)
        
        # Train both models with the same subset
        sgd_model = SGDAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        em_model = EMAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        
        criterion = nn.MSELoss()
        sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=learning_rate)
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=learning_rate)
        
        # Set fewer epochs for very small datasets to avoid overfitting
        epochs = min(20, max(10, int(20 * size)))
        
        # Train the models
        sgd_res = train_sgd(sgd_model, subset_loader, test_loader, sgd_optimizer, criterion, epochs)
        em_res = train_em(em_model, subset_loader, subset_em_loader, test_loader, em_optimizer, criterion, epochs)
        
        # Store results
        sgd_results.append((size, sgd_res))
        em_results.append((size, em_res))
    
    # Plot final test loss vs dataset size
    plt.figure(figsize=(10, 6))
    sizes_str = [f"{s*100:.1f}%" for s, _ in sgd_results]
    sgd_losses = [r["test_losses"][-1] for _, r in sgd_results]
    em_losses = [r["test_losses"][-1] for _, r in em_results]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    plt.bar(x - width/2, sgd_losses, width, label='SGD')
    plt.bar(x + width/2, em_losses, width, label='PAM-SGD')
    plt.xlabel('Training Set Size')
    plt.ylabel('Final Test Loss')
    plt.title('Effect of Training Set Size on Model Performance')
    plt.xticks(x, sizes_str)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_folder, 'training_size_comparison.png'))
    plt.close()  # Close instead of show
    
    # Plot loss curves for each size
    plt.figure(figsize=(15, 10))
    for i, (size, res) in enumerate(sgd_results):
        plt.subplot(2, 3, i+1)
        plt.plot(res["train_losses"], 'b-', label='SGD Train')
        plt.plot(res["test_losses"], 'b--', label='SGD Test')
        plt.plot(em_results[i][1]["train_losses"], 'r-', label='PAM-SGD Train')
        plt.plot(em_results[i][1]["test_losses"], 'r--', label='PAM-SGD Test')
        plt.title(f'Training with {size*100:.1f}% data')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if i == 0:
            plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_size_loss_curves.png'))
    plt.close() 
    
    return sgd_results, em_results

# Run experiments with different numbers of SGD updates per batch
def run_sgd_updates_ablation(output_folder, activation_type="topk"):
    # Number of SGD updates per batch to try
    sgd_updates_list = [1, 3, 5, 10]
    em_results = []
    
    for sgd_updates in sgd_updates_list:
        print(f"\n===== Training PAM-SGD with {sgd_updates} SGD updates per batch =====")
        em_model = EMAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        criterion = nn.MSELoss()
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=learning_rate)
        # Train with specified number of SGD updates per batch
        em_res = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, num_epochs, sgd_updates)
        em_results.append((sgd_updates, em_res))
    
    # Plot final test loss vs number of SGD updates
    plt.figure(figsize=(10, 6))
    sgd_updates_str = [str(u) for u, _ in em_results]
    em_losses = [r["test_losses"][-1] for _, r in em_results]
    
    plt.bar(sgd_updates_str, em_losses)
    plt.xlabel('SGD Updates per Batch')
    plt.ylabel('Final Test Loss')
    plt.title('Effect of SGD Updates per Batch on PAM-SGD Performance')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_folder, 'sgd_updates_comparison.png'))
    plt.close()  
    
    # Plot loss curves for each number of updates
    plt.figure(figsize=(15, 10))
    for i, (updates, res) in enumerate(em_results):
        plt.subplot(2, 2, i+1)
        plt.plot(res["train_losses"], 'r-', label='Train')
        plt.plot(res["test_losses"], 'r--', label='Test')
        plt.title(f'{updates} SGD Updates per Batch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sgd_updates_loss_curves.png'))
    plt.close()  
    
    return em_results


def compare_activations(sgd_model, em_model, data_loader, device, output_folder):
    sgd_model.eval()
    em_model.eval()
    
    # Get a single batch
    data, _ = next(iter(data_loader))
    data = data[:5].to(device)
    
    # Get latent activations
    with torch.no_grad():
        _, sgd_z = sgd_model(data)
        _, em_z = em_model(data)
    
    # Plot
    fig, axes = plt.subplots(5, 2, figsize=(16, 10))
    
    for i in range(5):
        # SGD activations
        sgd_act = sgd_z[i].cpu().numpy()
        axes[i, 0].stem(np.arange(len(sgd_act)), sgd_act)
        axes[i, 0].set_title(f"Sample {i+1}: SGD Latent Activations")
        axes[i, 0].set_xlabel(f"Active: {(sgd_act != 0).sum()}/{len(sgd_act)}")
        
        # EM activations
        em_act = em_z[i].cpu().numpy()
        axes[i, 1].stem(np.arange(len(em_act)), em_act)
        axes[i, 1].set_title(f"Sample {i+1}: PAM-SGD Latent Activations")
        axes[i, 1].set_xlabel(f"Active: {(em_act != 0).sum()}/{len(em_act)}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'activation_comparison.png'))
    plt.close()  

# Main area

def run_comparison():
    print("Initializing models...")
    # Initialize models
    sgd_model = SGDAutoencoder(image_size, latent_dim, k).to(device)
    em_model = EMAutoencoder(image_size, latent_dim, k).to(device)
    
    # Initialize optimizers and loss
    criterion = nn.MSELoss()
    sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=learning_rate)
    em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=learning_rate)
    
    # Train SGD model
    print("\n===== Training SGD Model =====")
    sgd_results = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, num_epochs)
    
    # Train EM model
    print("\n===== Training PAM-SGD Model =====")
    em_results = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, num_epochs)
    
    # Plot comparison results
    plot_losses(sgd_results, em_results)
    
    # Get some fixed test examples for visualization
    test_examples, _ = next(iter(test_loader))
    test_examples = test_examples[:10].to(device)
    
    plot_reconstruction_comparison(sgd_results, em_results, test_examples)
    compare_activations(sgd_model, em_model, test_loader, device)
    
    return sgd_results, em_results

def run_full_experiments():
    # Create output folder for plots
    output_folder = create_plots_folder()
    activation_types = ["topk", "relu"]
    all_results = {}
    for activation_type in activation_types:
        print(f"\n===== Running experiments with activation: {activation_type.upper()} =====")
        # Initialize default models
        sgd_model = SGDAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        em_model = EMAutoencoder(image_size, latent_dim, k, activation_type=activation_type).to(device)
        
        # Initialize optimizers and loss
        criterion = nn.MSELoss()
        sgd_optimizer = optim.Adam([
            {'params': sgd_model.encoder_weights, 'weight_decay': 0.0},
            {'params': sgd_model.encoder_bias, 'weight_decay': 0.0},
            {'params': sgd_model.decoder_bias, 'weight_decay': 0.0},
        ], lr=learning_rate)
        em_optimizer = optim.Adam([
            {'params': em_model.encoder_weights, 'weight_decay': 0.0},
            {'params': em_model.encoder_bias, 'weight_decay': 0.0},
        ], lr=learning_rate)
        
        # Print encoder bias before training (SGD)
        print(f"[DEBUG] SGD encoder bias (before): mean={sgd_model.encoder_bias.data.mean().item():.6f}, std={sgd_model.encoder_bias.data.std().item():.6f}")
        # Train SGD model
        print("\n===== Training Base SGD Model =====")
        sgd_results = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, num_epochs)
        # Print encoder bias after training (SGD)
        print(f"[DEBUG] SGD encoder bias (after): mean={sgd_model.encoder_bias.data.mean().item():.6f}, std={sgd_model.encoder_bias.data.std().item():.6f}")
        
        # Train PAM-SGD model
        print("\n===== Training Base PAM-SGD Model =====")
        em_results = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, num_epochs)
        
        # Get some fixed test examples for visualization
        test_examples, _ = next(iter(test_loader))
        test_examples = test_examples[:10].to(device)
        
        # Plot and compare results
        print("\n===== Visualizing Base Results =====")
        plot_losses(sgd_results, em_results, output_folder + f"/{activation_type}")
        plot_reconstruction_comparison(sgd_results, em_results, test_examples, output_folder + f"/{activation_type}")
        compare_activations(sgd_model, em_model, test_loader, device, output_folder + f"/{activation_type}")
        visualize_filters(sgd_model, em_model, output_folder + f"/{activation_type}")
        
        # Save base results
        os.makedirs(output_folder + f"/{activation_type}", exist_ok=True)
        torch.save({
            'sgd_model': sgd_model.state_dict(),
            'em_model': em_model.state_dict(),
            'sgd_results': sgd_results,
            'em_results': em_results
        }, os.path.join(output_folder, activation_type, 'base_comparison_results.pt'))
        
        print(f"Base results saved to '{os.path.join(output_folder, activation_type, 'base_comparison_results.pt')}'")
        
        # Run ablation studies
        print("\n===== Running SGD Updates Ablation =====")
        sgd_updates_results = run_sgd_updates_ablation(output_folder + f"/{activation_type}", activation_type=activation_type)
        
        print("\n===== Running Training Size Ablation =====")
        training_size_results = run_training_size_ablation(output_folder + f"/{activation_type}", activation_type=activation_type)
        
        print("\n===== Running Weight Decay Ablation =====")
        weight_decay_results = run_weight_decay_ablation(output_folder + f"/{activation_type}", activation_type=activation_type)
        
        print("\n===== Running Mu/Nu Ablation =====")
        mu_nu_results = run_mu_nu_ablation(output_folder + f"/{activation_type}", activation_type=activation_type)
        
        # Save ablation results
        torch.save({
            'sgd_updates_results': sgd_updates_results,
            'training_size_results': training_size_results,
            'weight_decay_results': weight_decay_results,
            'mu_nu_results': mu_nu_results
        }, os.path.join(output_folder, activation_type, 'ablation_results.pt'))
        
        print(f"Ablation results saved to '{os.path.join(output_folder, activation_type, 'ablation_results.pt')}'")
        
        all_results[activation_type] = {
            'base_comparison': (sgd_results, em_results),
            'sgd_updates_results': sgd_updates_results,
            'training_size_results': training_size_results,
            'weight_decay_results': weight_decay_results,
            'mu_nu_results': mu_nu_results,
            'output_folder': output_folder + f"/{activation_type}"
        }
    return all_results


if __name__ == "__main__":
    results = run_full_experiments()