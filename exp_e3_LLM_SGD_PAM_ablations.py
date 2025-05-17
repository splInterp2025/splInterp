# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script compares the benefits of our PAM-SGD method (Algorithm 3.1) versus
# standard SGD for training sparse autoencoders (SAEs) on high-dimensional LLM
# activations, using Google DeepMind’s Gemma-2-2B model.
#
# Instructions: Running this file will perform all the ablations either for TopK
# or for ReLU. You can change that setting and all other hyperparameters using
# command-line arguments or by editing the defaults at the bottom of the file in
# the parser section.
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------
# In the comments, PAM-SGD is sometimes abbreviated as EM for brevity.

# -----------------------------------------------------------------------------
# Command-Line Usage:
#
# To run this script with default settings:
#   python exp_e3_LLM_SGD_PAM_ablations.py
#
# To customize settings, use command-line arguments like:
#   python exp_e3_LLM_SGD_PAM_ablations.py --activation-type relu --latent-dim 2048 --num-epochs 50
#
# Available arguments:
#   --latent-dim            Latent dimension size (default: 4096)
#   --activation-type       Activation function type: 'topk' or 'relu' (default: topk)
#   --k                     TopK value (only used if activation-type is 'topk') (default: 320)
#   --l1-lambda             L1 regularization strength (default: 0.01)
#   --l1-lambda-pam-relu    L1 regularization for PAM-SGD with ReLU (default: 0.00001)
#   --learning-rate         Learning rate for optimizers (default: 0.001)
#   --num-epochs            Number of training epochs (default: 100)
#   --batch-size            Batch size for standard SGD (default: 256)
#   --em-batch-size         Batch size for EM/PAM-SGD decoder update (default: 2048)
#   --weight-decay          Weight decay for optimizers (default: 0.0)
#   --mu-enc                Encoder weight update penalty μ (PAM-SGD) (default: 0.001)
#   --nu-enc                Encoder bias update penalty ν (PAM-SGD) (default: 0.001)
#   --mu-dec                Decoder weight update penalty μ (ablation) (default: 0.001)
#   --nu-dec                Decoder bias update penalty ν (ablation) (default: 0.001)
#   --decoder-reg           L2 regularization for decoder weights (default: 1e-5)
#   --alpha-w               Decoder weight decay alpha_w (default: 1e-6)
#   --beta-b                Decoder bias decay beta_b (default: 1e-6)
#   --decoder-stay-close    Enable stay-close and decay in decoder update (flag)
#
# -----------------------------------------------------------------------------

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Set larger font sizes for better readability in figures
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Set seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################
# Parameters
###############################
# === Training Parameters ===
BATCH_SIZE = 256
EM_BATCH_SIZE = 2048
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
K = 320
LATENT_DIM = 4096
NUM_SAMPLES = 10000
SEQUENCE_LENGTH = 128
GEMMA_LAYER_WIDTH = 2304
TARGET_LAYER = 12

# === Regularization parameters (further increased for stability) ===
# Regularization parameters
DECODER_REG = 1e-5  # Regularization for analytical decoder solution (reduced from 1e-3)
ALPHA_W = 1e-6  # Weight decay for weights in decoder (α term) (reduced from 1e-3)
BETA_B = 1e-6  # Weight decay for biases in decoder (β term) (reduced from 1e-3)
L1_LAMBDA = 0.01  # L1 sparsity coefficient for TopK activations
L1_LAMBDA_PAM_RELU = 0.00001  # Greatly reduced L1 coefficient for ReLU activations (both SGD and PAM-SGD)

# ReLU naturally produces ~50% non-zero values while TopK only produces K values (~1.5% for K=15),
# so ReLU needs much lower L1 regularization to avoid excessive penalties

# Quadratic update cost parameters (μ and ν terms)
MU_ENC = 0.001  # Encoder weights "stay-close" penalty (μ_enc) (reduced from 0.01)
NU_ENC = 0.001  # Encoder bias "stay-close" penalty (ν_enc) (reduced from 0.01)
MU_DEC = 0.001  # Decoder weights "stay-close" penalty (μ_dec) (reduced from 0.01)
NU_DEC = 0.001  # Decoder bias "stay-close" penalty (ν_dec) (reduced from 0.01)

# Activation type
ACTIVATION_TYPE = "topk"  # Options: "topk", "relu"

# Paths
DATA_FOLDER = "gemma_activations"
RESULTS_FOLDER = os.path.join("plots", f"llm_ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

###############################
# Utility Functions
###############################
def ensure_folder(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def save_figure(fig, folder, filename):
    """Save a figure to a specific folder, ensuring the directory exists."""
    ensure_folder(folder)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    return path
    plt.close(fig)
    print(f"Saved figure to: {path}")
    return path

###############################
# Data Loading Functions
###############################
def get_text_samples(num_samples=NUM_SAMPLES):
    """Get text samples from stable sources with explicit error handling"""
    print(f"Loading {num_samples} text samples...")
    
    samples = []
    
    # Attempt to load from multiple datasets
    datasets_to_try = [
        ("EleutherAI/pile", "train"),
        ("wikitext", "wikitext-103-v1", "train"),
        ("wikipedia", "20220301.en", "train")
    ]
    
    for dataset_info in datasets_to_try:
        if len(samples) >= num_samples:
            break
            
        try:
            if len(dataset_info) == 2:
                dataset_name, split = dataset_info
                print(f"\nAttempting to load {dataset_name} ({split}) dataset...")
                dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
            else:
                dataset_name, config, split = dataset_info
                print(f"\nAttempting to load {dataset_name} ({config}, {split}) dataset...")
                dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                
            iterator = iter(dataset)
            
            # Keep track of consecutive failures
            failures = 0
            max_failures = 10
            
            while len(samples) < num_samples and failures < max_failures:
                try:
                    item = next(iterator)
                    
                    # Get text based on dataset schema
                    if "text" in item:
                        text = item["text"]
                    elif "page" in item and "text" in item["page"]:
                        text = item["page"]["text"]
                    else:
                        # Try to find a text field
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 50:
                                text = value
                                break
                        else:
                            raise ValueError(f"Could not find text field in {dataset_name} item")
                    
                    # Validate text content
                    if text and isinstance(text, str) and len(text.strip()) > 50:
                        # Trim to reasonable length
                        text = text[:SEQUENCE_LENGTH*10]
                        samples.append(text)
                        failures = 0  # Reset failure counter on success
                        
                        # Display sample previews
                        if len(samples) <= 3 or len(samples) % 100 == 0:
                            preview = text[:100].replace('\n', ' ')
                            print(f"\nSample {len(samples)} from {dataset_name}: {preview}...")
                    else:
                        failures += 1
                        if failures % 5 == 0:
                            print(f"Skipped {failures} items with insufficient text content")
                
                except Exception as e:
                    failures += 1
                    if failures >= max_failures:
                        print(f"Too many failures ({max_failures}) with {dataset_name}. Trying next dataset.")
                        print(f"Last error: {str(e)}")
                    
            if len(samples) >= num_samples:
                print(f"\nSuccessfully loaded {len(samples)} samples from {dataset_name}")
                break
                
        except Exception as e:
            print(f"Error loading {dataset_info[0]}: {str(e)}")
    
    # Check if we have enough samples
    if len(samples) < num_samples * 0.9:  # If we have less than 90% of requested samples
        print(f"WARNING: Only managed to collect {len(samples)} samples, expected at least {int(num_samples * 0.9)}")
    
    # Trim to exact number requested
    samples = samples[:min(len(samples), num_samples)]
    print(f"Successfully loaded {len(samples)} diverse text samples")
    
    return samples

def extract_gemma_activations(texts, output_file):
    """Extract layer 12 activations from Gemma-2-2B for a list of text samples"""
    ensure_folder(DATA_FOLDER)
    
    # Print some examples of the text being processed through Gemma
    print("\n=== Example texts being processed through Gemma ===")
    for i in range(min(3, len(texts))):
        preview = texts[i][:100].replace('\n', ' ')
        print(f"Text {i}: {preview}...")
    print("...")
    
    # Load model and tokenizer
    # https://huggingface.co/google/gemma-2-2b
    # https://ai.google.dev/gemma/terms
    print("Loading Gemma-2-2B model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModel.from_pretrained("google/gemma-2-2b", output_hidden_states=True)
    model.to(device)
    model.eval()
    
    all_activations = []
    problematic_samples = 0
    
    # Process in batches to avoid OOM
    mini_batch_size = 4
    for i in tqdm(range(0, len(texts), mini_batch_size), desc="Extracting activations"):
        batch_texts = texts[i:i+mini_batch_size]
        
        # Ensure each text has content
        for j in range(len(batch_texts)):
            if not batch_texts[j] or len(batch_texts[j].strip()) < 10:
                print(f"Warning: Empty text detected at index {i+j}")
                batch_texts[j] = "This is a replacement for an empty text sample."
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=SEQUENCE_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract activations from layer TARGET_LAYER
                if TARGET_LAYER < len(outputs.hidden_states):
                    layer_activations = outputs.hidden_states[TARGET_LAYER]
                else:
                    print(f"Warning: Target layer {TARGET_LAYER} exceeds model layers ({len(outputs.hidden_states)})")
                    layer_activations = outputs.hidden_states[-1]  # Use last layer
                
                # Get last non-padding token for each sequence
                for batch_idx in range(layer_activations.shape[0]):
                    # Get sum of attention mask (number of non-padding tokens)
                    sequence_length = torch.sum(inputs["attention_mask"][batch_idx]).item()
                    
                    # Safeguard against empty sequences
                    if sequence_length <= 0:
                        print(f"Warning: Empty sequence detected at index {i+batch_idx}")
                        continue
                    
                    # Get last token index (0-indexed, so subtract 1)
                    last_token_idx = int(sequence_length - 1)
                    
                    # Extract activation vector
                    activation = layer_activations[batch_idx, last_token_idx].cpu()
                    
                    # Check for NaN values
                    if torch.isnan(activation).any():
                        print(f"Warning: NaN detected in sample {i+batch_idx}")
                        problematic_samples += 1
                        continue
                    
                    all_activations.append(activation)
        
        except Exception as e:
            print(f"Error processing batch {i//mini_batch_size}: {e}")
            continue
        
        # Free up memory
        del outputs, layer_activations, inputs
        torch.cuda.empty_cache()
    
    print(f"Extracted {len(all_activations)} valid activations (skipped {problematic_samples} problematic samples)")
    
    # Stack all activations into a single tensor
    if all_activations:
        activations_tensor = torch.stack(all_activations)
        
        # Final safety check
        print(f"Activation stats before saving: Shape={activations_tensor.shape}, "
              f"Contains NaN: {torch.isnan(activations_tensor).any().item()}")
        
        # Save clean data
        torch.save(activations_tensor, output_file)
        print(f"Saved {len(all_activations)} activations to {output_file}")
    else:
        print("Error: No valid activations extracted!")
        activations_tensor = torch.zeros((1, GEMMA_LAYER_WIDTH))  # Create empty tensor as fallback
    
    # Clean up to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return activations_tensor

def prepare_datasets(activation_tensor, train_split=0.9):
    """Split activations into train and test sets"""
    # Get actual dimensions from data
    global GEMMA_LAYER_WIDTH
    GEMMA_LAYER_WIDTH = activation_tensor.size(1)
    print(f"Detected activation width: {GEMMA_LAYER_WIDTH}")
    
    # Compute dataset statistics
    print(f"Activation statistics: Mean={activation_tensor.mean().item():.4f}, "
          f"Std={activation_tensor.std().item():.4f}, "
          f"Min={activation_tensor.min().item():.4f}, "
          f"Max={activation_tensor.max().item():.4f}")
    
    dataset_size = activation_tensor.size(0)
    indices = torch.randperm(dataset_size)
    train_size = int(train_split * dataset_size)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_activations = activation_tensor[train_indices]
    test_activations = activation_tensor[test_indices]
    
    train_dataset = TensorDataset(train_activations)
    test_dataset = TensorDataset(test_activations)
    
    return train_dataset, test_dataset

def load_or_extract_activations():
    """Load existing activation file or extract new ones if needed"""
    activation_file = os.path.join(DATA_FOLDER, f"gemma_layer{TARGET_LAYER}_activations.pt")
    
    if os.path.exists(activation_file):
        print(f"Loading existing activations from {activation_file}")
        activations = torch.load(activation_file)
    else:
        print("Extracting activations from Gemma model")
        texts = get_text_samples(NUM_SAMPLES)
        activations = extract_gemma_activations(texts, activation_file)
    
    # Filter out any NaN activations to ensure clean data
    if torch.isnan(activations).any():
        print("Warning: NaN values found in activations. Filtering them out.")
        nan_mask = ~torch.isnan(activations).any(dim=1)
        clean_activations = activations[nan_mask]
        print(f"Filtered out {activations.size(0) - clean_activations.size(0)} samples with NaNs.")
        activations = clean_activations

    # === Normalize activations (zero mean, unit variance) ===
    mean = activations.mean(dim=0, keepdim=True)
    std = activations.std(dim=0, keepdim=True) + 1e-8
    activations = (activations - mean) / std
    print(f"[Normalization] Activation mean after normalization: {activations.mean().item():.4f}, std: {activations.std().item():.4f}")

    # Optionally, save mean and std for future use (not implemented here)

    return activations

###############################
# Model Definitions 
###############################

# Base Autoencoder with configurable sparsity
class SparseAutoencoderBase(nn.Module):
    def __init__(self, input_dim, latent_dim, activation_type=ACTIVATION_TYPE, 
                 k=K, l1_lambda=L1_LAMBDA, 
                 mu_enc=MU_ENC, nu_enc=NU_ENC):
        super(SparseAutoencoderBase, self).__init__()
        self.encoder_weights = nn.Parameter(torch.randn(input_dim, latent_dim) / np.sqrt(input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))  # Set bias to zero after normalization
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Store previous parameters for "stay-close" regularization
        self.prev_encoder_weights = None
        self.prev_encoder_bias = None
        self.prev_decoder_bias = None
        
        # Configuration parameters
        self.k = k
        self.l1_lambda = l1_lambda
        self.activation_type = activation_type
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Quadratic update cost parameters
        self.mu_enc = mu_enc
        self.nu_enc = nu_enc
    
    def store_previous_params(self):
        """Store current parameters for "stay-close" regularization"""
        self.prev_encoder_weights = self.encoder_weights.detach().clone()
        self.prev_encoder_bias = self.encoder_bias.detach().clone()
        self.prev_decoder_bias = self.decoder_bias.detach().clone()
    
    def encode(self, x):
        # Linear encoder
        out = x @ self.encoder_weights + self.encoder_bias
        # Debug: print pre-activation stats for the first batch in each epoch
        if hasattr(self, 'debug_print') and self.debug_print:
            print("[DEBUG] Encoder pre-activation stats:")
            print("  Mean:", out.mean().item())
            print("  Std:", out.std().item())
            print("  Min:", out.min().item())
            print("  Max:", out.max().item())
            print("[DEBUG] Encoder bias stats:")
            print("  Mean:", self.encoder_bias.data.mean().item())
            print("  Min:", self.encoder_bias.data.min().item())
            print("  Max:", self.encoder_bias.data.max().item())
            self.debug_print = False  # Only print once per epoch
        return out
    
    def apply_sparsity(self, z):
        """Apply sparsity based on activation_type"""
        if self.activation_type == "topk":
            # Apply k-sparsity: keep only the k largest activations per sample
            values, indices = torch.topk(torch.abs(z), k=self.k, dim=1)
            mask = torch.zeros_like(z).scatter_(1, indices, 1)
            return z * mask
        elif self.activation_type == "relu":
            # Apply ReLU activation for sparsity
            return torch.relu(z)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")
            
    def get_l1_loss(self, z_sparse):
        """Calculate L1 sparsity loss for ReLU activation"""
        if self.activation_type == "relu":
            # Use the instance's l1_lambda value which may be different for PAM-SGD vs regular SGD
            # For PAM-SGD with ReLU, we use a reduced L1_LAMBDA_PAM_RELU value to avoid excessive penalties
            return self.l1_lambda * z_sparse.abs().sum(dim=1).mean()
        return 0.0  # No L1 loss for TopK
    
    def get_stay_close_loss(self):
        """Calculate quadratic 'stay-close' loss for encoder parameters"""
        loss = 0
        if self.prev_encoder_weights is not None and self.mu_enc > 0:
            # μ_enc * ||W_enc - W_enc_prev||²_F
            loss += self.mu_enc * ((self.encoder_weights - self.prev_encoder_weights) ** 2).sum()
        
        if self.prev_encoder_bias is not None and self.nu_enc > 0:
            # ν_enc * ||b_enc - b_enc_prev||²_2
            loss += self.nu_enc * ((self.encoder_bias - self.prev_encoder_bias) ** 2).sum()
        
        return loss
        
    def decode(self, z):
        # Tied weights: use transpose of encoder weights
        return z @ self.encoder_weights.t() + self.decoder_bias
    
    def forward(self, x):
        # Encode (linear)
        z = self.encode(x)
        
        # Apply sparsity
        z_sparse = self.apply_sparsity(z)
        
        # Decode and return
        recon = self.decode(z_sparse)
               
        return recon, z_sparse

# SGD Version (standard autodiff)
class SGDAutoencoder(SparseAutoencoderBase):
    def __init__(self, input_dim, latent_dim, activation_type=ACTIVATION_TYPE, 
                 k=K, l1_lambda=L1_LAMBDA, mu_enc=MU_ENC, nu_enc=NU_ENC):
        super(SGDAutoencoder, self).__init__(input_dim, latent_dim, activation_type, 
                                           k, l1_lambda, mu_enc, nu_enc)
        # This is the baseline version using standard SGD

# EM Version (PAM-SGD)
class EMAutoencoder(SparseAutoencoderBase):
    def __init__(self, input_dim, latent_dim, activation_type=ACTIVATION_TYPE, 
                 k=K, l1_lambda=None,
                 mu_enc=MU_ENC, nu_enc=NU_ENC, mu_dec=MU_DEC, nu_dec=NU_DEC,
                 alpha_w=ALPHA_W, beta_b=BETA_B):
        # Use reduced L1 lambda specifically for PAM-SGD with ReLU activation
        if l1_lambda is None:
            if activation_type == "relu":
                l1_lambda = L1_LAMBDA_PAM_RELU  # Use lower lambda for PAM-SGD with ReLU
            else:
                l1_lambda = L1_LAMBDA  # Use default lambda for other cases
        
        super(EMAutoencoder, self).__init__(input_dim, latent_dim, activation_type, 
                                          k, l1_lambda, mu_enc, nu_enc)
        
        # For EM, handle the decoder weights separately (not as Parameter)
        self.decoder_weights = torch.randn(latent_dim, input_dim, device=device) / np.sqrt(latent_dim)
        
        # Store previous decoder parameters for "stay-close" regularization
        self.prev_decoder_weights = None
        
        # Decoder regularization parameters
        self.mu_dec = mu_dec  # Weight "stay-close" penalty
        self.nu_dec = nu_dec  # Bias "stay-close" penalty
        self.alpha_w = alpha_w  # Weight decay for weights
        self.beta_b = beta_b   # Weight decay for biases
    
    def store_previous_params(self):
        """Store current parameters for "stay-close" regularization"""
        super().store_previous_params()
        self.prev_decoder_weights = self.decoder_weights.detach().clone()
    
    def decode(self, z):
        # explicitly maintained decoder weights instead of tied weights
        return z @ self.decoder_weights + self.decoder_bias
    
    def apply_sparsity(self, z):
        """Apply sparsity based on activation_type"""
        if self.activation_type == "topk":
            # Apply k-sparsity: keep only the k largest activations per sample
            values, indices = torch.topk(torch.abs(z), k=self.k, dim=1)
            mask = torch.zeros_like(z).scatter_(1, indices, 1)
            return z * mask
        elif self.activation_type == "relu":
            # Apply ReLU activation for sparsity
            return torch.relu(z)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")    
        
    def update_decoder_analytically(self, data_loader, num_batches=None, decoder_reg=DECODER_REG, use_decoder_stay_close=False):
        """
        Solve for optimal decoder weights using the pseudo-inverse:
        W_dec = (Z^T Z)^(-1) Z^T X with regularization.

        By default, only a small ridge regularization (decoder_reg) is used for numerical stability.
        Decoder weight decay (alpha_w, beta_b) and stay-close (mu_dec, nu_dec) regularization are only applied
        if explicitly enabled (e.g., in ablation studies or if use_decoder_stay_close=True).

        Args:
            data_loader: DataLoader for the M-step.
            num_batches: Number of batches to use (optional).
            decoder_reg: Ridge regularization for decoder analytic update.
            use_decoder_stay_close: If True, apply decoder weight stay-close and weight decay (for ablation only).
        """
        self.eval()  # Set to eval mode for consistent sparsity application

        # Collect activations and targets
        all_z_sparse = []
        all_inputs = []

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if num_batches is not None and i >= num_batches:
                    break
                data = data[0].to(device)
                z = self.encode(data)
                z_sparse = self.apply_sparsity(z)
                all_z_sparse.append(z_sparse)
                all_inputs.append(data)

        Z = torch.cat(all_z_sparse, dim=0)  # [N, latent_dim]
        X = torch.cat(all_inputs, dim=0)     # [N, input_dim]

        # Default: only ridge regularization for decoder analytic update
        reg_Z = torch.cat([Z, decoder_reg * torch.eye(Z.shape[1], device=device)], dim=0)
        reg_X = torch.cat([X, torch.zeros(Z.shape[1], X.shape[1], device=device)], dim=0)

        # Optionally add decoder weight decay and stay-close regularization (for ablation only)
        if use_decoder_stay_close and (self.alpha_w > 0 or self.mu_dec > 0):
            n_latent = Z.shape[1]
            reg_eye = torch.eye(n_latent, device=Z.device)
            reg_rows = []
            reg_targets = []
            if self.alpha_w > 0:
                reg_rows.append((self.alpha_w ** 0.5) * reg_eye)
                reg_targets.append(torch.zeros((n_latent, X.shape[1]), device=X.device))
            if self.mu_dec > 0 and self.prev_decoder_weights is not None:
                reg_rows.append((self.mu_dec ** 0.5) * reg_eye)
                reg_targets.append((self.mu_dec ** 0.5) * self.prev_decoder_weights)
            if reg_rows:
                reg_Z = torch.cat([reg_Z] + reg_rows, dim=0)
                reg_X = torch.cat([reg_X] + reg_targets, dim=0)

        # Use least-squares solution for regularized system
        self.decoder_weights = torch.linalg.lstsq(reg_Z, reg_X).solution

        # Update bias term (mean residual)
        pred = Z @ self.decoder_weights
        residual = X - pred
        b = torch.mean(residual, dim=0)
        # Optionally add stay-close and weight decay for bias (for ablation only)
        if use_decoder_stay_close:
            if self.prev_decoder_bias is not None and self.nu_dec > 0:
                b = (b + self.nu_dec * self.prev_decoder_bias) / (1 + self.nu_dec)
            if self.beta_b > 0:
                b = b / (1 + self.beta_b)
        self.decoder_bias.data = b

###############################
# Training Functions
###############################

# SGD training loop
def train_sgd(model, train_loader, test_loader, optimizer, criterion, epochs):
    train_losses = []
    test_losses = []
    wall_times = []
    recon_examples = []
    
    # Store some fixed test examples for visualization
    test_examples = next(iter(test_loader))[0][:10].to(device)
    
    # Store initial parameters for "stay-close" regularization
    model.store_previous_params()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)
            
            # Enable debug print for the first batch of each epoch
            if batch_idx == 0:
                model.debug_print = False

            optimizer.zero_grad()
            recon, z_sparse = model(data)
            
            # Reconstruction loss
            loss = criterion(recon, data)
              # Add L1 sparsity loss for ReLU activation
            l1_loss = model.get_l1_loss(z_sparse)
            if l1_loss > 0:
                loss = loss + l1_loss
                if batch_idx == 0 and epoch % 5 == 0:
                    print(f"SGD - L1 loss component: {l1_loss.item():.6f}, L1 lambda: {model.l1_lambda}")
            
            # Add "stay-close" loss
            stay_close_loss = model.get_stay_close_loss()
            if stay_close_loss > 0:
                loss = loss + stay_close_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'SGD - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        # Store parameters for next epoch's "stay-close" regularization
        model.store_previous_params()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)        
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data[0].to(device)
                recon, z_sparse = model(data)
                loss = criterion(recon, data)
                  # Also include L1 sparsity in evaluation for consistency with training
                l1_loss = model.get_l1_loss(z_sparse)
                if l1_loss > 0:
                    loss = loss + l1_loss
                    if batch_idx == 0 and epoch % 5 == 0:
                        active_percent = 100 * (z_sparse != 0).float().mean().item()
                        print(f"SGD Test - Basic loss: {loss.item() - l1_loss.item():.6f}, " + 
                              f"L1 loss: {l1_loss.item():.6f}, L1 lambda: {model.l1_lambda}, " +
                              f"Active neurons: {active_percent:.2f}%")
                        # Debug: print encoder bias during test
                        #print("[DEBUG] Encoder bias stats (test):")
                        #print("  Mean:", model.encoder_bias.data.mean().item())
                        #print("  Min:", model.encoder_bias.data.min().item())
                        #print("  Max:", model.encoder_bias.data.max().item())
                
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Record wall clock time for epoch
        time_elapsed = time.time() - start_time
        wall_times.append(time_elapsed)
        
        print(f'SGD - Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, Time: {time_elapsed:.2f}s')
        
        # Save reconstruction examples at intervals
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                recon, _ = model(test_examples)
                recon_examples.append((epoch, deepcopy(recon.detach())))
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "wall_times": wall_times,
        "total_time": sum(wall_times),
        "recon_examples": recon_examples
    }

# EM (PAM-SGD) training loop
def train_em(model, train_loader, em_train_loader, test_loader, encoder_optimizer, criterion, 
            epochs, sgd_updates_per_batch=1):
    """
    Train using Plug-and-Play Alternating Minimization SGD (PAM-SGD):
    E-step: Compute latent codes with fixed parameters
    M-step: 1) Analytically update decoder weights
            2) Update encoder weights with SGD
    """
    train_losses = []
    test_losses = []
    wall_times = []
    recon_examples = []
    
    # Store some fixed test examples for visualization
    test_examples = next(iter(test_loader))[0][:10].to(device)
    
    # Store initial parameters for "stay-close" regularization
    model.store_previous_params()
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # ==================== E-STEP ====================
        # Compute latent codes (no parameter updates)
        print(f'EM (E-step) - Epoch {epoch+1}/{epochs}, Computing latent codes...')
          # M-STEP PART 1: UPDATE DECODER ANALYTICALLY
        print(f'EM (M-step Part 1) - Epoch {epoch+1}/{epochs}, Computing analytic decoder solution...')
        
        # Print decoder norm before update for diagnostics
        if epoch % 5 == 0:
            decoder_norm_before = torch.norm(model.decoder_weights).item()
            print(f"Decoder weights norm before update: {decoder_norm_before:.6f}")
        
        # Use the simplified decoder update with reduced regularization
        model.update_decoder_analytically(em_train_loader, decoder_reg=DECODER_REG)
        
        # Print decoder norm after update for diagnostics
        if epoch % 5 == 0:
            decoder_norm_after = torch.norm(model.decoder_weights).item()
            print(f"Decoder weights norm after update: {decoder_norm_after:.6f}")
            print(f"Decoder weights change: {decoder_norm_after - decoder_norm_before:.6f}")
        
        # M-STEP PART 2: UPDATE ENCODER WITH SGD
        print(f'EM (M-step Part 2) - Epoch {epoch+1}/{epochs}, Updating encoder with SGD...')
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)
            batch_count += 1
            batch_loss = 0
            
            # Perform SGD updates on the encoder
            for update_idx in range(sgd_updates_per_batch):
                encoder_optimizer.zero_grad()
                recon, z_sparse = model(data)
                
                # Reconstruction loss
                loss = criterion(recon, data)
                basic_loss = loss.item()  # Store for diagnostics
                
                # Add L1 sparsity loss for ReLU activation
                l1_loss = model.get_l1_loss(z_sparse)
                if l1_loss > 0:
                    loss = loss + l1_loss
                    if batch_idx == 0 and update_idx == 0 and epoch % 5 == 0:
                        print(f"PAM-SGD - L1 loss component: {l1_loss.item():.6f}, L1 lambda: {model.l1_lambda}")
                
                # Add minimal "stay-close" loss for encoder weights
                stay_close_loss = model.get_stay_close_loss()
                if stay_close_loss > 0:
                    loss = loss + stay_close_loss
                    if batch_idx == 0 and update_idx == 0 and epoch % 5 == 0:
                        print(f"PAM-SGD - Stay-close loss: {stay_close_loss.item():.6f}")
                
                loss.backward()
                
                # Add gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                encoder_optimizer.step()
                batch_loss += loss.item()
                
                # Diagnostic printing for first update of first batch periodically
                if batch_idx == 0 and update_idx == 0 and epoch % 5 == 0:
                    print(f"PAM-SGD Losses - Basic: {basic_loss:.6f}, " +
                          f"With L1: {basic_loss + l1_loss.item() if l1_loss > 0 else basic_loss:.6f}, " +
                          f"Total: {loss.item():.6f}")
            
            # Average batch loss over all updates
            batch_loss = batch_loss / sgd_updates_per_batch
            epoch_loss += batch_loss
            
            if (batch_idx + 1) % 10 == 0:
                print(f'EM (M-step Part 2) - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {batch_loss:.6f}')
        
        # Store parameters for next epoch's "stay-close" regularization
        model.store_previous_params()
        
        avg_train_loss = epoch_loss / len(train_loader)

        # Add train loss tracking for PAM-SGD/EM implementation
        train_losses.append(avg_train_loss)        
        
        
        # EVALUATION
        model.eval()
        test_loss = 0
        test_base_loss = 0  # Track reconstruction loss without regularization
        test_l1_loss = 0    # Track L1 regularization component
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                batch_count += 1
                data = data[0].to(device)
                recon, z_sparse = model(data)
                
                # Calculate reconstruction loss (MSE)
                base_loss = criterion(recon, data)
                test_base_loss += base_loss.item()
                
                # Total loss starts with reconstruction loss
                loss = base_loss                
                # Add L1 sparsity in evaluation - with reduced impact 
                l1_loss = model.get_l1_loss(z_sparse)
                if l1_loss > 0:
                    test_l1_loss += l1_loss.item()
                    loss = loss + l1_loss
                    
                    # Print detailed diagnostics
                    if batch_idx == 0 and epoch % 5 == 0:
                        active_percent = 100 * (z_sparse != 0).float().mean().item()
                        print(f"PAM-SGD Test - Basic loss: {base_loss.item():.6f}, " + 
                              f"L1 loss: {l1_loss.item():.6f}, L1 lambda: {model.l1_lambda}, " +
                              f"Active neurons: {active_percent:.2f}%")
                
                test_loss += loss.item()
        
        # Compute averages
        avg_test_base_loss = test_base_loss / batch_count
        avg_test_l1_loss = test_l1_loss / batch_count if test_l1_loss > 0 else 0
        avg_test_loss = test_loss / batch_count
        
        # Detailed loss reporting
        if epoch % 5 == 0:
            print(f"PAM-SGD Test Losses - Reconstruction: {avg_test_base_loss:.6f}, " +
                  f"L1 Component: {avg_test_l1_loss:.6f}, Total: {avg_test_loss:.6f}")
        
        test_losses.append(avg_test_loss)
        
        # Record wall clock time
        time_elapsed = time.time() - start_time
        wall_times.append(time_elapsed)
        
        print(f'EM - Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, Time: {time_elapsed:.2f}s')
              
        # Save reconstruction examples at intervals
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                recon, _ = model(test_examples)
                recon_examples.append((epoch, deepcopy(recon.detach())))
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "wall_times": wall_times,
        "total_time": sum(wall_times),
        "recon_examples": recon_examples
    }

###############################
# Visualization Functions
###############################

# Plot loss curves
def plot_losses(sgd_results, em_results, output_folder):
    plt.figure(figsize=(12, 6))

    plt.plot(sgd_results["train_losses"], 'b-', label='SGD Train')
    plt.plot(sgd_results["test_losses"], 'b--', label='SGD Test')
    plt.plot(em_results["train_losses"], 'r-', label='PAM-SGD Train')
    plt.plot(em_results["test_losses"], 'r--', label='PAM-SGD Test')

    # Ignore the first epoch (or more) for y-limit calculation
    skip = 10  # Number of initial epochs to skip (can increase if needed)
    all_loss_values = (
        sgd_results["train_losses"][skip:] +
        sgd_results["test_losses"][skip:] +
        em_results["train_losses"][skip:] +
        em_results["test_losses"][skip:]
    )


    if all_loss_values:
        # Set y-limit to show both curves, but cap at a reasonable value if extremely high
        max_loss = max(all_loss_values)
        min_loss = min(all_loss_values)
        # If the max loss is much higher than the 99th percentile, show the full range but annotate
        y_limit = max_loss * 1.05  # 5% margin above max
        # If max_loss is extremely high, cap at 10x the 99th percentile and annotate
        perc99 = np.percentile(all_loss_values, 99)
        if max_loss > 10 * perc99:
            y_limit = 10 * perc99
            plt.ylim(0, y_limit)
            plt.figtext(0.01, 0.01, f"Note: Max loss {max_loss:.2e} (Y-axis capped at {y_limit:.1f})",
                        fontsize=8, ha='left')
        else:
            plt.ylim(0, y_limit)
        # Optionally, annotate if any values are out of range
        if max_loss > y_limit:
            plt.figtext(0.01, 0.05, f"Some values exceed y-limit and are not shown.", fontsize=8, ha='left')

    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss Comparison')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_folder, 'loss_comparison.png'))

    # --- Fix for log scale plot ---
    def safe_log(vals):
        # Replace zero or negative values with a small positive value for log plotting
        return [max(v, 1e-8) for v in vals]

    plt.clf()  # Clear the figure before re-plotting
    plt.plot(safe_log(sgd_results["train_losses"]), 'b-', label='SGD Train')
    plt.plot(safe_log(sgd_results["test_losses"]), 'b--', label='SGD Test')
    plt.plot(safe_log(em_results["train_losses"]), 'r-', label='PAM-SGD Train')
    plt.plot(safe_log(em_results["test_losses"]), 'r--', label='PAM-SGD Test')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE, log scale)')
    plt.title('Training and Test Loss Comparison (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'loss_comparison_log.png'))

    plt.close()
    print(f"Saved loss comparison plots to {output_folder}")

# Compare activations between models
def compare_activations(sgd_model, em_model, data_loader, output_folder):
    """Compare the activations between SGD and EM models"""
    sgd_model.eval()
    em_model.eval()
    
    # Get a single batch
    data = next(iter(data_loader))[0].to(device)
    data = data[:5]  # Use only 5 samples for visualization
    
    # Get latent activations and pre-activations for SGD
    with torch.no_grad():
        # Pre-activation (before sparsity)
        sgd_pre_z = sgd_model.encode(data)
        _, sgd_z = sgd_model(data)
        _, em_z = em_model(data)

    # Log debug info for SGD model
    print("\n[DEBUG] SGD Model Activation Type:", getattr(sgd_model, 'activation_type', 'UNKNOWN'))
    print("[DEBUG] SGD Pre-activation stats (before sparsity):")
    print("  Mean: ", sgd_pre_z.mean().item())
    print("  Std:  ", sgd_pre_z.std().item())
    print("  Min:  ", sgd_pre_z.min().item())
    print("  Max:  ", sgd_pre_z.max().item())
    print("[DEBUG] SGD Activation stats (after sparsity):")
    print("  Mean: ", sgd_z.mean().item())
    print("  Std:  ", sgd_z.std().item())
    print("  Min:  ", sgd_z.min().item())
    print("  Max:  ", sgd_z.max().item())
    print("  Nonzero count: ", (sgd_z != 0).sum().item(), "/", sgd_z.numel())

    # Plot
    fig, axes = plt.subplots(5, 2, figsize=(16, 10))

    for i in range(5):
        # SGD activations
        #axes[i, 0].set_ylim(0, 10)
        #axes[i, 1].set_ylim(0, 10)

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
    # Ensure output directory exists before saving
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'activation_comparison.png'))
    plt.close()
    print(f"Saved activation comparison plot to {output_folder}")

# Plot reconstructions over time for both models
def plot_reconstruction_progress(sgd_results, em_results, test_examples, output_folder):
    # Get reconstruction examples from results
    sgd_recons = sgd_results["recon_examples"]
    em_recons = em_results["recon_examples"]
    
    # Check if we have reconstruction examples saved
    if not sgd_recons or not em_recons:
        print("Warning: No reconstruction examples available to plot.")
        return
    
    # For LLM activations, we can't directly visualize the reconstructions as images
    # Instead, plot the first 20 dimensions for a single example over time
    
    # Select a single example and take the first 20 dimensions
    example_idx = 0
    ndims = 20
    
    # Get original example data
    original = test_examples[example_idx].cpu().numpy()[:ndims]
    
    # Select equally spaced epochs to display
    n_steps = 5
    sgd_idxs = np.linspace(0, len(sgd_recons)-1, n_steps).astype(int)
    em_idxs = np.linspace(0, len(em_recons)-1, n_steps).astype(int)
    
    sgd_epochs = [sgd_recons[i][0] for i in sgd_idxs]
    
    # Create a figure with 2 rows
    fig, axes = plt.subplots(2, n_steps, figsize=(15, 6))
    
    # Original data reference (horizontal line)
    for row in range(2):
        for col in range(n_steps):
            axes[row, col].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
              # Collect all errors to determine appropriate axis limits
    all_errors = []
    
    # Collect SGD errors
    sgd_all_errors = []
    for idx, i in enumerate(sgd_idxs):
        epoch, recon = sgd_recons[i]
        recon_data = recon[example_idx].cpu().numpy()[:ndims]
        error = recon_data - original
        sgd_all_errors.append(error)
        
        # Skip the first epoch (which often has extreme error values) when calculating y-axis limits
        # We'll still display it, just not use it for determining the scale
        if idx > 0:
            all_errors.extend(error)
    
    # Collect EM errors
    em_all_errors = []
    for idx, i in enumerate(em_idxs):
        epoch, recon = em_recons[i]
        recon_data = recon[example_idx].cpu().numpy()[:ndims]
        error = recon_data - original
        em_all_errors.append(error)
        
        # Skip the first epoch when calculating y-axis limits
        if idx > 0:
            all_errors.extend(error)
    
    # Calculate dynamic limits with 10% padding, but set a reasonable cap
    # If there are no valid errors (e.g., only epoch 0 available), use a default value
    if all_errors:
        # Use the second epoch's errors as reference (as requested)
        if len(sgd_idxs) > 1:  # Ensure there's a second epoch to use as reference
            second_epoch_sgd_error = sgd_all_errors[1]
            second_epoch_em_error = em_all_errors[1]
            reference_errors = list(second_epoch_sgd_error) + list(second_epoch_em_error)
            max_abs_error = max(abs(min(reference_errors)), abs(max(reference_errors))) * 1.2  # 20% padding
            y_limit = max_abs_error
        else:
            # Fall back to using all errors except the first epoch
            max_abs_error = max(abs(min(all_errors)), abs(max(all_errors))) * 1.1
            y_limit = max_abs_error
    else:
        # Default fallback if no valid errors
        y_limit = 5.0  # Reasonable default based on MSE values you mentioned (~1.9)
      # SGD reconstructions over time
    for col, i in enumerate(sgd_idxs):
        idx = sgd_idxs[col]
        epoch, recon = sgd_recons[idx]
        recon_data = recon[example_idx].cpu().numpy()[:ndims]
        error = recon_data - original
        
        # Calculate the MSE for this epoch
        mse = np.mean(error**2)
        
        # Set special handling for first epoch if it has extreme values
        first_epoch = col == 0
        extreme_values = max(abs(error)) > y_limit * 5  # Arbitrary threshold to identify extreme outliers
        
        # Plot the bars with clipping for extreme values
        if first_epoch and extreme_values:
            # Use a different color for the first epoch if it has extreme values
            bars = axes[0, col].bar(range(ndims), np.clip(error, -y_limit, y_limit), color='orange', alpha=0.7)
            # Add a note about clipping
            axes[0, col].text(ndims/2, 0, "Values clipped", ha='center', color='red', fontsize=8)
        else:
            # Regular plotting for non-extreme epochs
            bars = axes[0, col].bar(range(ndims), error)
            
        axes[0, col].set_ylim(-y_limit, y_limit)  # Use consistent limits for better comparison
        axes[0, col].set_title(f"Epoch {epoch+1}")
        
        # Add MSE info to subtitle
        axes[0, col].set_xlabel(f"MSE: {mse:.4f}")
        
        if col == 0:
            axes[0, col].set_ylabel("SGD Error\n(recon - original)")
    
    # EM reconstructions over time
    for col, i in enumerate(em_idxs):
        idx = em_idxs[col]
        epoch, recon = em_recons[idx]
        recon_data = recon[example_idx].cpu().numpy()[:ndims]
        error = recon_data - original
        
        # Calculate the MSE for this epoch
        mse = np.mean(error**2)
        
        # Set special handling for first epoch if it has extreme values
        first_epoch = col == 0
        extreme_values = max(abs(error)) > y_limit * 5  # Arbitrary threshold to identify extreme outliers
        
        # Plot the bars with clipping for extreme values
        if first_epoch and extreme_values:
            # Use a different color for the first epoch if it has extreme values
            bars = axes[1, col].bar(range(ndims), np.clip(error, -y_limit, y_limit), color='orange', alpha=0.7)
            # Add a note about clipping
            axes[1, col].text(ndims/2, 0, "Values clipped", ha='center', color='red', fontsize=8)
        else:
            # Regular plotting for non-extreme epochs
            bars = axes[1, col].bar(range(ndims), error)
        
        axes[1, col].set_ylim(-y_limit, y_limit)  # Use consistent limits for better comparison
        
        # Add MSE info to subtitle
        axes[1, col].set_xlabel(f"MSE: {mse:.4f}")
        
        if col == 0:
            axes[1, col].set_ylabel("PAM-SGD Error\n(recon - original)")
      # Calculate overall reconstruction progress
    sgd_mses = [np.mean((sgd_all_errors[i])**2) for i in range(len(sgd_idxs))]
    em_mses = [np.mean((em_all_errors[i])**2) for i in range(len(em_idxs))]
    
    sgd_progress = 100 * (sgd_mses[0] - sgd_mses[-1]) / sgd_mses[0] if sgd_mses[0] > 0 else 0
    em_progress = 100 * (em_mses[0] - em_mses[-1]) / em_mses[0] if em_mses[0] > 0 else 0
    
    title = (f"Reconstruction Error Over Time (First {ndims} dimensions)\n"
             f"SGD improvement: {sgd_progress:.1f}%, PAM-SGD improvement: {em_progress:.1f}%")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Add a note about extreme values in the first epoch if applicable
    max_first_sgd = max(abs(sgd_all_errors[0])) if sgd_all_errors else 0
    max_first_em = max(abs(em_all_errors[0])) if em_all_errors else 0
    
    if max_first_sgd > y_limit * 5 or max_first_em > y_limit * 5:
        note = (f"Note: First epoch has extreme error values (SGD max: {max_first_sgd:.2f}, PAM-SGD max: {max_first_em:.2f}).\n"
                f"Y-axis range set to ±{y_limit:.2f} based on later epochs for better visualization.")
        plt.figtext(0.5, 0.01, note, ha='center', fontsize=9, 
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        plt.subplots_adjust(bottom=0.1)
    
    plt.savefig(os.path.join(output_folder, 'reconstruction_progress.png'))
    plt.close()  # Close instead of show
    print(f"Saved reconstruction progress plot to {output_folder} with improved y-limits")

# Plot reconstruction quality comparison
def plot_reconstruction_comparison(sgd_results, em_results, test_examples, output_folder):
    # Get the latest reconstructions
    _, latest_sgd_recons = sgd_results["recon_examples"][-1]
    _, latest_em_recons = em_results["recon_examples"][-1]
    
    # For LLM activations, let's compare the reconstruction error for each example
    n_examples = min(5, test_examples.size(0))
    
    # Prepare data
    originals = test_examples[:n_examples]
    sgd_recons = latest_sgd_recons[:n_examples]
    em_recons = latest_em_recons[:n_examples]
    
    # Calculate errors
    sgd_errors = torch.mean((sgd_recons - originals)**2, dim=1).cpu().numpy()
    em_errors = torch.mean((em_recons - originals)**2, dim=1).cpu().numpy()
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot errors for each example
    x = np.arange(n_examples)
    width = 0.35
    
    ax.bar(x - width/2, sgd_errors, width, label='SGD Error')
    ax.bar(x + width/2, em_errors, width, label='PAM-SGD Error')
    
    ax.set_xlabel('Example')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction MSE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Example {i+1}' for i in range(n_examples)])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'reconstruction_comparison.png'))
    plt.close()
    
    print(f"Saved reconstruction comparison plot to {output_folder}")

# Visualize filter weights
def visualize_filters(sgd_model, em_model, output_folder, n_filters=10):
    # For LLM activations, we can't directly visualize filters as 2D patterns
    # Instead, let's plot the distribution of weights for a few selected filters
    
    # Print model regularization parameters for diagnostics
    print(f"\nModel regularization parameters:")
    print(f"SGD model - L1 lambda: {sgd_model.l1_lambda}, activation: {sgd_model.activation_type}")
    print(f"PAM-SGD model - L1 lambda: {em_model.l1_lambda}, activation: {em_model.activation_type}")
    
    # Get encoder weights
    sgd_enc_weights = sgd_model.encoder_weights.data.cpu().numpy()
    em_enc_weights = em_model.encoder_weights.data.cpu().numpy()
    
    # Get decoder weights
    sgd_dec_weights = sgd_model.encoder_weights.t().data.cpu().numpy()  # Tied weights
    em_dec_weights = em_model.decoder_weights.data.cpu().numpy()
    
    # Select a subset of filters to visualize
    filter_indices = np.random.choice(sgd_model.latent_dim, size=n_filters, replace=False)
    
    fig, axes = plt.subplots(4, n_filters, figsize=(16, 12))
    
    # Plot histograms for each filter
    for i, idx in enumerate(filter_indices):
        # SGD encoder filter weights distribution
        axes[0, i].hist(sgd_enc_weights[:, idx], bins=30, alpha=0.7)
        axes[0, i].set_title(f"SGD Enc Filter {idx}")
        axes[0, i].set_yticks([])
        
        # EM encoder filter weights distribution
        axes[1, i].hist(em_enc_weights[:, idx], bins=30, alpha=0.7)
        axes[1, i].set_title(f"PAM-SGD Enc Filter {idx}")
        axes[1, i].set_yticks([])
        
        # SGD decoder filter weights distribution
        axes[2, i].hist(sgd_dec_weights[idx, :], bins=30, alpha=0.7)
        axes[2, i].set_title(f"SGD Dec Filter {idx}")
        axes[2, i].set_yticks([])
        
        # EM decoder filter weights distribution
        axes[3, i].hist(em_dec_weights[idx, :], bins=30, alpha=0.7)
        axes[3, i].set_title(f"PAM-SGD Dec Filter {idx}")
        axes[3, i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'filter_weight_distributions.png'))
    plt.close()
    
    # Also visualize weight statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot weight magnitude distributions
    sgd_enc_magnitude = np.linalg.norm(sgd_enc_weights, axis=0)
    em_enc_magnitude = np.linalg.norm(em_enc_weights, axis=0)
    
    axes[0, 0].hist(sgd_enc_magnitude, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title("SGD Encoder Filter Magnitudes")
    axes[0, 0].set_xlabel("L2 Norm")
    axes[0, 0].set_ylabel("Count")
    
    axes[0, 1].hist(em_enc_magnitude, bins=30, alpha=0.7, color='red')
    axes[0, 1].set_title("PAM-SGD Encoder Filter Magnitudes")
    axes[0, 1].set_xlabel("L2 Norm")
    axes[0, 1].set_ylabel("Count")
    
    # Plot decoder weight statistics
    sgd_dec_magnitude = np.linalg.norm(sgd_dec_weights, axis=1)
    em_dec_magnitude = np.linalg.norm(em_dec_weights, axis=1)
    
    axes[1, 0].hist(sgd_dec_magnitude, bins=30, alpha=0.7, color='blue')
    axes[1, 0].set_title("SGD Decoder Filter Magnitudes")
    axes[1, 0].set_xlabel("L2 Norm")
    axes[1, 0].set_ylabel("Count")
    
    axes[1, 1].hist(em_dec_magnitude, bins=30, alpha=0.7, color='red')
    axes[1, 1].set_title("PAM-SGD Decoder Filter Magnitudes")
    axes[1, 1].set_xlabel("L2 Norm")
    axes[1, 1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'filter_magnitude_distributions.png'))
    plt.close()
    
    # Create an additional comparison plot showing both models' filter properties side by side
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart to compare average filter magnitudes
    labels = ['Encoder', 'Decoder']
    sgd_means = [np.mean(sgd_enc_magnitude), np.mean(sgd_dec_magnitude)]
    em_means = [np.mean(em_enc_magnitude), np.mean(em_dec_magnitude)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, sgd_means, width, label='SGD', color='blue', alpha=0.7)
    ax.bar(x + width/2, em_means, width, label='PAM-SGD', color='red', alpha=0.7)
    
    ax.set_ylabel('Average Filter Magnitude')
    ax.set_title('Comparison of Filter Magnitudes Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add detailed stats as text
    stats_text = (
        f"SGD Encoder: mean={np.mean(sgd_enc_magnitude):.4f}, std={np.std(sgd_enc_magnitude):.4f}\n"
        f"PAM-SGD Encoder: mean={np.mean(em_enc_magnitude):.4f}, std={np.std(em_enc_magnitude):.4f}\n"
        f"SGD Decoder: mean={np.mean(sgd_dec_magnitude):.4f}, std={np.std(sgd_dec_magnitude):.4f}\n"
        f"PAM-SGD Decoder: mean={np.mean(em_dec_magnitude):.4f}, std={np.std(em_dec_magnitude):.4f}"
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(output_folder, 'filter_comparison.png'))
    plt.close()
    
    print(f"Saved filter visualizations to {output_folder}")
    print(f"Saved filter comparison plot to {output_folder}")

###############################
# Ablation Studies
###############################

# Run experiments with different training set sizes
def run_training_size_ablation(full_train_dataset, test_dataset, output_folder):
    # Training set sizes to try (fraction of full dataset)
    sizes = [0.01, 0.05, 0.1, 0.25, 0.40, 0.45, 0.5, 0.65, 0.75, 0.85, 1.0]
    sgd_results = []
    em_results = []
    
    import sys
    log_path = os.path.join(output_folder, "training_size_ablation_debug.txt")
    log_file = open(log_path, "w")
    orig_stdout = sys.stdout
    sys.stdout = log_file
    try:
        for size in sizes:
            size_fraction = size  # Store actual fraction for reporting
            # Create dataset subset
            subset_size = max(int(len(full_train_dataset) * size), 10)  # Ensure at least 10 samples
            print(f"\n===== Training with {subset_size} samples ({size_fraction*100:.1f}% of data) =====")
            # Create subset of the training data
            indices = torch.randperm(len(full_train_dataset))[:subset_size]
            print(f"[DEBUG] Indices for this subset (first 10): {indices[:10].tolist()}")
            print(f"[DEBUG] Number of unique indices: {len(set(indices.tolist()))} / {len(indices)}")
            train_subset = Subset(full_train_dataset, indices)
            # Print mean/std of activations in this subset
            subset_acts = full_train_dataset.tensors[0][indices]
            print(f"[DEBUG] Subset activations mean: {subset_acts.mean().item():.4f}, std: {subset_acts.std().item():.4f}, min: {subset_acts.min().item():.4f}, max: {subset_acts.max().item():.4f}")
            # Print mean/std of test set activations
            test_acts = test_dataset.tensors[0]
            print(f"[DEBUG] Test set activations mean: {test_acts.mean().item():.4f}, std: {test_acts.std().item():.4f}, min: {test_acts.min().item():.4f}, max: {test_acts.max().item():.4f}")
            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            em_train_loader = DataLoader(train_subset, batch_size=EM_BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            # Initialize models
            print("Initializing models...")
            sgd_model = SGDAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=ACTIVATION_TYPE, k=K).to(device)
            em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=ACTIVATION_TYPE, k=K).to(device)
            # Initialize optimizers and loss
            criterion = nn.MSELoss()
            sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=LEARNING_RATE)
            em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=LEARNING_RATE)
            # Train SGD model
            print(f"\n===== Training SGD Model with {size_fraction*100:.1f}% of data =====")
            sgd_size_result = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, NUM_EPOCHS // 2)
            print(f"[DEBUG] SGD test losses (first 5): {sgd_size_result['test_losses'][:5]}")
            print(f"[DEBUG] SGD test losses (last 5): {sgd_size_result['test_losses'][-5:]}")
            # Train EM model
            print(f"\n===== Training PAM-SGD Model with {size_fraction*100:.1f}% of data =====")
            em_size_result = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, NUM_EPOCHS // 2)
            print(f"[DEBUG] PAM-SGD test losses (first 5): {em_size_result['test_losses'][:5]}")
            print(f"[DEBUG] PAM-SGD test losses (last 5): {em_size_result['test_losses'][-5:]}")
            sgd_results.append((size_fraction, sgd_size_result))
            em_results.append((size_fraction, em_size_result))
    finally:
        sys.stdout = orig_stdout
        log_file.close()
    print(f"Debug output for training size ablation written to {log_path}")
      # Plot final test loss vs dataset size
    plt.figure(figsize=(10, 6))
    sizes_str = [f"{s*100:.1f}%" for s, _ in sgd_results]
    sgd_losses = [r["test_losses"][-1] for _, r in sgd_results]
    em_losses = [r["test_losses"][-1] for _, r in em_results]
    
    x = np.arange(len(sizes))
    width = 0.35
    
    plt.bar(x - width/2, sgd_losses, width, label='SGD')
    plt.bar(x + width/2, em_losses, width, label='PAM-SGD')
    plt.xlabel('Training Set Size', fontsize=14)
    plt.ylabel('Final Test Loss', fontsize=14)
    plt.title('Effect of Training Set Size on Model Performance', fontsize=16)
    plt.xticks(x, sizes_str)
    plt.legend()
    plt.ylim(0, 15)
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_folder, 'training_size_comparison.png'))
    plt.close()  # Close instead of show
    
    # Plot loss curves for each size (multi-panel grid, dynamic sizing)
    import math
    n_sizes = len(sgd_results)
    ncols = 4  # You can change this to 3 if you prefer
    nrows = math.ceil(n_sizes / ncols)
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for i, (size, res) in enumerate(sgd_results):
        plt.subplot(nrows, ncols, i+1)
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
    
    print(f"Saved training size ablation plots to {output_folder}")
    return sgd_results, em_results

# Run experiments with different numbers of SGD updates per batch
def run_sgd_updates_ablation(train_dataset, test_dataset, output_folder):
    # Number of SGD updates per batch to try
    sgd_updates_list = [1, 3, 5, 10]
    em_results = []
    
    for sgd_updates in sgd_updates_list:
        print(f"\n===== Training PAM-SGD with {sgd_updates} SGD updates per batch =====")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        em_train_loader = DataLoader(train_dataset, batch_size=EM_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
          # Initialize model
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, ACTIVATION_TYPE).to(device)
        
        # Initialize optimizer and loss
        criterion = nn.MSELoss()
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=LEARNING_RATE)
        
        # Train model with specified number of SGD updates
        updates_result = train_em(em_model, train_loader, em_train_loader, test_loader, 
                                em_optimizer, criterion, NUM_EPOCHS // 2, sgd_updates_per_batch=sgd_updates)
        
        em_results.append((sgd_updates, updates_result))
    
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
    colors = plt.cm.viridis(np.linspace(0, 1, len(sgd_updates_list)))
    
    for i, ((updates, res), color) in enumerate(zip(em_results, colors)):
        plt.plot(res["test_losses"], color=color, label=f'{updates} Updates')
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs. Epochs for Different Numbers of SGD Updates per Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sgd_updates_loss_curves.png'))
    plt.close()
    
    print(f"Saved SGD updates ablation plots to {output_folder}")
    return em_results

# Run experiments comparing TopK vs ReLU activation
def run_activation_type_ablation(train_dataset, test_dataset, output_folder):
    """Compare TopK vs ReLU activation on both SGD and PAM-SGD"""
    activation_types = ["topk", "relu"]
    sgd_results = []
    em_results = []

    for act_type in activation_types:
        print(f"\n===== Training models with {act_type.upper()} activation =====")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        em_train_loader = DataLoader(train_dataset, batch_size=EM_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
      # Initialize models with appropriate regularization parameters based on activation type
        if act_type == "relu":
            # For ReLU, use the reduced L1 regularization for both models
            sgd_model = SGDAutoencoder(
                GEMMA_LAYER_WIDTH, 
                LATENT_DIM, 
                activation_type=act_type,
                l1_lambda=L1_LAMBDA_PAM_RELU  # Use same reduced L1 parameter for fair comparison
            ).to(device)
            
            # For ReLU, use the very reduced L1 regularization
            em_model = EMAutoencoder(
                GEMMA_LAYER_WIDTH, 
                LATENT_DIM, 
                activation_type=act_type,
                l1_lambda=L1_LAMBDA_PAM_RELU,  # Use greatly reduced L1 parameter
                mu_enc=0.0005,                 # Further reduce stay-close penalties for ReLU
                nu_enc=0.0005
            ).to(device)
            
            print(f"Using reduced L1_LAMBDA_PAM_RELU={L1_LAMBDA_PAM_RELU} for both SGD-RELU and PAM-SGD-RELU")
            print("This ensures fair comparison by preventing excessive L1 regularization penalties with ReLU activation")
        else:
            # For TopK, use standard parameters
            sgd_model = SGDAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=act_type).to(device)
            em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=act_type).to(device)
            print(f"Using standard parameters for TopK activation models")
        
        # Initialize optimizers and loss
        criterion = nn.MSELoss()
        sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=LEARNING_RATE)
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=LEARNING_RATE)
        
        # Train SGD model
        print(f"\n===== Training SGD Model with {act_type.upper()} activation =====")
        sgd_act_result = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, NUM_EPOCHS // 2)
        
        # Train EM model
        print(f"\n===== Training PAM-SGD Model with {act_type.upper()} activation =====")
        em_act_result = train_em(em_model, train_loader, em_train_loader, test_loader, 
                               em_optimizer, criterion, NUM_EPOCHS // 2)
        
        sgd_results.append((act_type, sgd_act_result))
        em_results.append((act_type, em_act_result))
          # Store model references for later use
        sgd_act_result["model"] = sgd_model
        em_act_result["model"] = em_model
          # Also compare activations right after training
        activation_output_dir = os.path.join(output_folder, f'{act_type}_activation_comparison')
        os.makedirs(activation_output_dir, exist_ok=True)
        compare_activations(sgd_model, em_model, test_loader, activation_output_dir)
      # Plot loss curves for different activation types
    plt.figure(figsize=(12, 6))
    
    # Plot SGD results
    plt.plot(sgd_results[0][1]["test_losses"], 'b-', label=f'SGD-{activation_types[0].upper()}', linewidth=2)
    plt.plot(sgd_results[1][1]["test_losses"], 'b--', label=f'SGD-{activation_types[1].upper()}', linewidth=2)
    
    # Plot EM results
    plt.plot(em_results[0][1]["test_losses"], 'r-', label=f'PAM-SGD-{activation_types[0].upper()}', linewidth=2)
    plt.plot(em_results[1][1]["test_losses"], 'r--', label=f'PAM-SGD-{activation_types[1].upper()}', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Test Loss', fontsize=14)
    plt.title('Test Loss vs. Epochs for Different Activation Types', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'activation_type_comparison.png'))
    plt.close()    # Compute sparsity levels for different activation types
    avg_topk_sparsity = compute_activation_sparsity(sgd_results[0][1]["model"], em_results[0][1]["model"], test_loader)
    avg_relu_sparsity = compute_activation_sparsity(sgd_results[1][1]["model"], em_results[1][1]["model"], test_loader)
      # Plot sparsity comparison
    plt.figure(figsize=(8, 6))
    labels = ['SGD', 'PAM-SGD']
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, [avg_topk_sparsity["sgd"], avg_topk_sparsity["em"]], width, label='TopK')
    plt.bar(x + width/2, [avg_relu_sparsity["sgd"], avg_relu_sparsity["em"]], width, label='ReLU')
    
    plt.xlabel('Optimization Method', fontsize=14)
    plt.ylabel('Average Sparsity (% non-zero)', fontsize=14)
    plt.title('Activation Sparsity Comparison', fontsize=16)
    
    # Add value annotations on top of each bar
    for i, v in enumerate([avg_topk_sparsity["sgd"], avg_topk_sparsity["em"]]):
        plt.text(i - width/2, v + 0.5, f"{v:.1f}%", ha='center', fontsize=11)
    
    for i, v in enumerate([avg_relu_sparsity["sgd"], avg_relu_sparsity["em"]]):
        plt.text(i + width/2, v + 0.5, f"{v:.1f}%", ha='center', fontsize=11)
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sparsity_comparison.png'))
    plt.close()
    
    print(f"Saved activation type ablation plots to {output_folder}")
    return sgd_results, em_results

def compute_activation_sparsity(sgd_model, em_model, test_loader):
    """Compute average sparsity (% of non-zero activations) for models"""
    # Handle both model objects and model paths
    if isinstance(sgd_model, str):
        sgd_model = SGDAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM)
        sgd_model.load_state_dict(torch.load(sgd_model))
        sgd_model = sgd_model.to(device)
        
    if isinstance(em_model, str):
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM)
        em_model.load_state_dict(torch.load(em_model))
        em_model = em_model.to(device)
    
    sgd_model.eval()
    em_model.eval()
    
    # Get a batch of test data
    data = next(iter(test_loader))[0].to(device)
    
    # Get activations
    with torch.no_grad():
        _, sgd_z = sgd_model(data)
        _, em_z = em_model(data)
    
    # Calculate sparsity (percentage of non-zero activations)
    sgd_sparsity = 100 * (sgd_z != 0).float().mean().item()
    em_sparsity = 100 * (em_z != 0).float().mean().item()
    
    return {"sgd": sgd_sparsity, "em": em_sparsity}

# Run experiments with different weight decay values
def run_weight_decay_ablation(train_dataset, test_dataset, output_folder):
    """Compare different weight decay values for both SGD and PAM-SGD"""
    # Weight decay values to try (applied to different components)
    weight_decay_values = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    sgd_results = []
    em_results = []
    
    for weight_decay in weight_decay_values:
        print(f"\n===== Training with weight decay = {weight_decay} =====")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        em_train_loader = DataLoader(train_dataset, batch_size=EM_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
          # Initialize SGD model
        sgd_model = SGDAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, 
                                 activation_type=ACTIVATION_TYPE, k=K).to(device)
        
        # Initialize EM model with weight decay for decoder
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, 
                               activation_type=ACTIVATION_TYPE, k=K,
                               alpha_w=weight_decay, beta_b=weight_decay).to(device)
        
        # Initialize optimizers with weight decay for SGD
        criterion = nn.MSELoss()
        sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], 
                                lr=LEARNING_RATE, weight_decay=weight_decay)
        
        # Train SGD model
        print(f"\n===== Training SGD Model with weight decay = {weight_decay} =====")
        sgd_wd_result = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, NUM_EPOCHS // 2)
        
        # Train EM model
        print(f"\n===== Training PAM-SGD Model with weight decay = {weight_decay} =====")
        em_wd_result = train_em(em_model, train_loader, em_train_loader, test_loader, 
                             em_optimizer, criterion, NUM_EPOCHS // 2)
        
        sgd_results.append((weight_decay, sgd_wd_result))
        em_results.append((weight_decay, em_wd_result))
      # Plot final test loss vs weight decay
    plt.figure(figsize=(12, 6))
    x = [str(wd) if wd > 0 else "0" for wd in weight_decay_values]
    sgd_losses = [r["test_losses"][-1] for _, r in sgd_results]
    em_losses = [r["test_losses"][-1] for _, r in em_results]
    
    plt.plot(x, sgd_losses, 'b-o', label='SGD', linewidth=2, markersize=8)
    plt.plot(x, em_losses, 'r-o', label='PAM-SGD', linewidth=2, markersize=8)
    plt.xlabel('Weight Decay', fontsize=14)
    plt.ylabel('Final Test Loss', fontsize=14)
    plt.title('Effect of Weight Decay on Model Performance', fontsize=16)
    
    # Increase x-tick label visibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, 'weight_decay_comparison.png'))
    plt.close()
    
    # Plot loss curves for different weight decays
    plt.figure(figsize=(15, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(weight_decay_values)))
    
    # Plot SGD results
    for i, ((wd, res), color) in enumerate(zip(sgd_results, colors)):
        wd_str = str(wd) if wd > 0 else "0"
        plt.plot(res["test_losses"], linestyle='-', color=color, label=f'SGD-WD={wd_str}')
    
    # Plot EM results
    for i, ((wd, res), color) in enumerate(zip(em_results, colors)):
        wd_str = str(wd) if wd > 0 else "0"
        plt.plot(res["test_losses"], linestyle='--', color=color, label=f'PAM-SGD-WD={wd_str}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs. Epochs for Different Weight Decay Values')
    plt.ylim(0, 20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'weight_decay_loss_curves.png'))
    plt.close()
    
    print(f"Saved weight decay ablation plots to {output_folder}")
    return sgd_results, em_results

# Run experiments with different values of K for TopK activation
# Run experiments with different values of K for TopK activation
def run_k_value_ablation(train_dataset, test_dataset, output_folder):
    """Compare different K values (10, 15, 20, 30, 40, 80, 160) for both SGD and PAM-SGD with TopK activation"""
    
    # K values to test
    k_values = [15, 20, 40, 80, 160, 320, 640, 1280]
    em_results = []

    for k_val in k_values:
        print(f"\n===== Training PAM-SGD Model with K = {k_val} =====")
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        em_train_loader = DataLoader(train_dataset, batch_size=EM_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize PAM-SGD (EM) model with the specific K value
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type="topk", k=k_val).to(device)
        criterion = nn.MSELoss()
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=LEARNING_RATE)

        # Train EM model
        em_k_result = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, NUM_EPOCHS // 2)
        em_results.append((k_val, em_k_result))

    # Plot PAM-SGD test loss curves for different K values (single chart)
    plt.figure(figsize=(12, 8))
    #colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(k_values)))
    for (k_val, res), color in zip(em_results, colors):
        plt.plot(res["test_losses"], color=color, label=f'K={k_val}')

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Test Loss', fontsize=14)
    plt.title('PAM-SGD: Test Loss vs. Epochs for Different K Values', fontsize=16)
    plt.legend()
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'k_value_loss_curves.png'))
    plt.close()

    print(f"Saved K value ablation plot to {output_folder}")
    return em_results


# Run experiments with different mu/nu values
def run_mu_nu_ablation(train_dataset, test_dataset, output_folder):
    """Compare different mu/nu values for PAM-SGD"""
    # mu/nu values to try (on logarithmic scale)
    mu_nu_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    results = []
    
    for mu_nu in mu_nu_values:
        print(f"\n===== Training PAM-SGD with mu=nu={mu_nu} =====")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        em_train_loader = DataLoader(train_dataset, batch_size=EM_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
          # Initialize EM model with mu/nu parameters
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, 
                               activation_type=ACTIVATION_TYPE, k=K,
                               mu_enc=mu_nu, nu_enc=mu_nu, 
                               mu_dec=mu_nu, nu_dec=mu_nu).to(device)
        
        # Initialize optimizer and loss
        criterion = nn.MSELoss()
        em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=LEARNING_RATE)
        
        # Train model
        mu_nu_result = train_em(em_model, train_loader, em_train_loader, test_loader, 
                             em_optimizer, criterion, NUM_EPOCHS // 2)
        
        results.append((mu_nu, mu_nu_result))
      # Plot final test loss vs mu/nu values
    plt.figure(figsize=(10, 6))
    x = [str(v) for v in mu_nu_values]
    losses = [r["test_losses"][-1] for _, r in results]
    
    plt.plot(x, losses, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('μ = ν', fontsize=14)
    plt.ylabel('Final Test Loss', fontsize=14)
    plt.title('Effect of μ/ν Parameters on PAM-SGD Performance', fontsize=16)
    
    # Add value annotations on top of each point
    for i, v in enumerate(losses):
        plt.text(i, v + 0.0005, f"{v:.4f}", ha='center', fontsize=11)
    
    # Increase x-tick label visibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'mu_nu_comparison.png'))
    plt.close()
    
    # Plot loss curves for different mu/nu values
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mu_nu_values)))
    
    for i, ((mu_nu, res), color) in enumerate(zip(results, colors)):
        plt.plot(res["test_losses"], color=color, label=f'μ=ν={mu_nu}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs. Epochs for Different μ/ν Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mu_nu_loss_curves.png'))
    plt.close()
    
    print(f"Saved mu/nu ablation plots to {output_folder}")
    return results

###############################
# Main Functions
###############################

def plot_combined_results(results_dict, output_folder):
    """
    Plot combined results from all ablation studies for easy comparison
    
    Args:
        results_dict: Dictionary with keys for different ablation types and values containing results
        output_folder: Folder to save the combined plots
    """
    # Create a comprehensive plot showing the best performers from each ablation
    plt.figure(figsize=(15, 8))
    
    # Extract best performers based on final test loss
    best_results = {}
    
    # For activation type ablation
    if "activation" in results_dict:
        sgd_results, em_results = results_dict["activation"]
        
        # Find best activation for SGD and EM
        best_sgd_act = min(sgd_results, key=lambda x: x[1]["test_losses"][-1])
        best_em_act = min(em_results, key=lambda x: x[1]["test_losses"][-1])
        
        # Add to best results
        best_results["SGD-" + best_sgd_act[0]] = best_sgd_act[1]["test_losses"]
        best_results["PAM-SGD-" + best_em_act[0]] = best_em_act[1]["test_losses"]
    
    # For weight decay ablation
    if "weight_decay" in results_dict:
        sgd_results, em_results = results_dict["weight_decay"]
        
        # Find best weight decay for SGD and EM
        best_sgd_wd = min(sgd_results, key=lambda x: x[1]["test_losses"][-1])
        best_em_wd = min(em_results, key=lambda x: x[1]["test_losses"][-1])
        
        # Add to best results
        wd_sgd_str = str(best_sgd_wd[0]) if best_sgd_wd[0] > 0 else "0"
        wd_em_str = str(best_em_wd[0]) if best_em_wd[0] > 0 else "0"
        best_results[f"SGD-WD={wd_sgd_str}"] = best_sgd_wd[1]["test_losses"]
        best_results[f"PAM-SGD-WD={wd_em_str}"] = best_em_wd[1]["test_losses"]
    
    # For mu/nu ablation
    if "mu_nu" in results_dict:
        mu_nu_results = results_dict["mu_nu"]
        
        # Find best mu/nu setting
        best_mu_nu = min(mu_nu_results, key=lambda x: x[1]["test_losses"][-1])
        
        # Add to best results
        best_results[f"PAM-SGD-μ=ν={best_mu_nu[0]}"] = best_mu_nu[1]["test_losses"]
      # Plot all best performers
    colors = plt.cm.tab10(np.linspace(0, 1, len(best_results)))
    for (label, losses), color in zip(best_results.items(), colors):
        plt.plot(losses, label=label, color=color, linewidth=2)
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Test Loss', fontsize=14)
    plt.title('Best Performers from Each Ablation Study', fontsize=16)
    
    # Increase tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save combined results plot
    plt.savefig(os.path.join(output_folder, 'combined_best_results.png'))
    plt.close()
    
    # Create a summary bar chart for final performance
    plt.figure(figsize=(12, 6))
    labels = list(best_results.keys())
    final_losses = [losses[-1] for losses in best_results.values()]
    
    plt.bar(range(len(labels)), final_losses, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Model Configuration')
    plt.ylabel('Final Test Loss')
    plt.title('Final Test Loss Comparison Across Best Configurations')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save summary bar chart
    plt.savefig(os.path.join(output_folder, 'combined_final_losses.png'))
    plt.close()
      # Optionally create combined training time comparison
    if ("activation" in results_dict and
        all("total_time" in result[1] for result in results_dict["activation"][0]) and 
        all("total_time" in result[1] for result in results_dict["activation"][1])):
        plt.figure(figsize=(10, 6))
        
        # Calculate average times
        avg_sgd_time = np.mean([result[1]["total_time"] for result in sgd_results])
        avg_em_time = np.mean([result[1]["total_time"] for result in em_results])
        
        plt.bar(["SGD", "PAM-SGD"], [avg_sgd_time, avg_em_time])
        plt.xlabel('Optimization Method')
        plt.ylabel('Average Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save training time comparison
        plt.savefig(os.path.join(output_folder, 'combined_training_time.png'))
        plt.close()

def run_comparison():
    ensure_folder(DATA_FOLDER)
    ensure_folder(RESULTS_FOLDER)
    
    print("\n===== Loading LLM Activations =====")
    activations = load_or_extract_activations()
    train_dataset, test_dataset = prepare_datasets(activations)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    em_train_loader = DataLoader(train_dataset, batch_size=EM_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
      # Initialize models
    print("\n===== Initializing Models =====")
    # Use correct L1 lambda for each activation type
    if ACTIVATION_TYPE == "relu":
        sgd_model = SGDAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=ACTIVATION_TYPE, k=K, l1_lambda=L1_LAMBDA_PAM_RELU).to(device)
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=ACTIVATION_TYPE, k=K, l1_lambda=L1_LAMBDA_PAM_RELU).to(device)
    else:
        sgd_model = SGDAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=ACTIVATION_TYPE, k=K, l1_lambda=L1_LAMBDA).to(device)
        em_model = EMAutoencoder(GEMMA_LAYER_WIDTH, LATENT_DIM, activation_type=ACTIVATION_TYPE, k=K, l1_lambda=L1_LAMBDA).to(device)
    
    # Initialize optimizers and loss
    criterion = nn.MSELoss()
    sgd_optimizer = optim.Adam(sgd_model.parameters(), lr=LEARNING_RATE)
    em_optimizer = optim.Adam([em_model.encoder_weights, em_model.encoder_bias], lr=LEARNING_RATE)
    
    # Train SGD model
    print("\n===== Training SGD Model =====")
    sgd_results = train_sgd(sgd_model, train_loader, test_loader, sgd_optimizer, criterion, NUM_EPOCHS)
    
    # Train EM model
    print("\n===== Training PAM-SGD (EM) Model =====")
    em_results = train_em(em_model, train_loader, em_train_loader, test_loader, em_optimizer, criterion, NUM_EPOCHS, 1)
    
    # Get test examples for visualization
    test_examples = next(iter(test_loader))[0][:10].to(device)
    
    # Create visualizations
    print("\n===== Creating Visualizations =====")
    plot_losses(sgd_results, em_results, RESULTS_FOLDER)
    compare_activations(sgd_model, em_model, test_loader, RESULTS_FOLDER)
    plot_reconstruction_comparison(sgd_results, em_results, test_examples, RESULTS_FOLDER)
    plot_reconstruction_progress(sgd_results, em_results, test_examples, RESULTS_FOLDER)
    visualize_filters(sgd_model, em_model, RESULTS_FOLDER)
    
    return sgd_results, em_results, sgd_model, em_model, train_dataset, test_dataset

def run_ablation_studies(train_dataset, test_dataset):
    print("\n===== Running Ablation Studies =====")
    results_dict = {}
    
    
    # Run sgd updates ablation
    print("\n===== Running SGD Updates Ablation =====")
    sgd_updates_results = run_sgd_updates_ablation(train_dataset, test_dataset, RESULTS_FOLDER)
    results_dict["sgd_updates"] = sgd_updates_results
    
    # Run training size ablation
    print("\n===== Running Training Size Ablation =====")
    size_results = run_training_size_ablation(train_dataset, test_dataset, RESULTS_FOLDER)
    results_dict["training_size"] = size_results
    
    # Run activation type ablation
    print("\n===== Running Activation Type Ablation =====")
    print(f"Using default L1_LAMBDA={L1_LAMBDA} for SGD-RELU but reduced L1_LAMBDA_PAM_RELU={L1_LAMBDA_PAM_RELU} for PAM-SGD-RELU")
    print("This ensures fair comparison by preventing excessive L1 regularization penalties in the PAM-SGD-RELU model")
    act_results = run_activation_type_ablation(train_dataset, test_dataset, RESULTS_FOLDER)
    results_dict["activation"] = act_results
    
    # Run weight decay ablation
    print("\n===== Running Weight Decay Ablation =====")
    wd_results = run_weight_decay_ablation(train_dataset, test_dataset, RESULTS_FOLDER)
    results_dict["weight_decay"] = wd_results
    
    # Run mu/nu ablation
    print("\n===== Running μ/ν Ablation =====")
    mu_nu_results = run_mu_nu_ablation(train_dataset, test_dataset, RESULTS_FOLDER)
    results_dict["mu_nu"] = mu_nu_results
    
    # Run K value ablation
    print("\n===== Running K Value Ablation =====")
    k_results = run_k_value_ablation(train_dataset, test_dataset, RESULTS_FOLDER)
    results_dict["k_value"] = k_results
    
    #Plot combined results
    print("\n===== Creating Combined Results Visualization =====")
    plot_combined_results(results_dict, RESULTS_FOLDER)
    
    
    return results_dict




def main():
    """
    Main entry point for running LLM activation experiments with SGD and PAM-SGD (EM),
    including ablation studies for weight decay and quadratic update costs (μ, ν).
    All key hyperparameters can be set via command-line arguments for reproducibility.
    """
    parser = argparse.ArgumentParser(description="LLM Activations: SGD vs PAM-SGD Comparison and Ablations")
    parser.add_argument('--latent-dim', type=int, default=4096, help='Latent dimension size')
    parser.add_argument('--activation-type', type=str, default='topk', choices=['topk', 'relu'], help='Activation function type')
    parser.add_argument('--k', type=int, default=320, help='TopK value (if using topk activation)')
    parser.add_argument('--l1-lambda', type=float, default=0.01, help='L1 regularization strength')
    parser.add_argument('--l1-lambda-pam-relu', type=float, default=0.00001, help='L1 regularization for PAM-SGD-RELU')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizers (lowered for stability)')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for SGD')
    parser.add_argument('--em-batch-size', type=int, default=2048, help='Batch size for EM (PAM-SGD) decoder update (increased for stability)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for optimizers (SGD and EM)')
    parser.add_argument('--mu-enc', type=float, default=0.001, help='Quadratic update cost μ for encoder weights (PAM-SGD)')
    parser.add_argument('--nu-enc', type=float, default=0.001, help='Quadratic update cost ν for encoder bias (PAM-SGD)')
    parser.add_argument('--mu-dec', type=float, default=0.001, help='Quadratic update cost μ for decoder weights (PAM-SGD, ablation only, increased for stability)')
    parser.add_argument('--nu-dec', type=float, default=0.001, help='Quadratic update cost ν for decoder bias (PAM-SGD, ablation only, increased for stability)')
    parser.add_argument('--decoder-reg', type=float, default=1e-5, help='L2 regularization for decoder weights (PAM-SGD, increased for stability)')
    parser.add_argument('--alpha-w', type=float, default=1e-6, help='Decoder weight decay alpha_w (PAM-SGD, increased for stability)')
    parser.add_argument('--beta-b', type=float, default=1e-6, help='Decoder bias decay beta_b (PAM-SGD, increased for stability)')
    parser.add_argument('--decoder-stay-close', action='store_true', help='Enable decoder weight stay-close and weight decay regularization in analytic update (for ablation only)')
    parser.add_argument('--run-ablation', action='store_true', default=True, help='Run ablation studies (weight decay, mu/nu). Default: True')
    parser.add_argument('--results-folder', type=str, default='results', help='Folder to save results and plots')
    # Default: run all ablations unless --no-ablation is specified
    parser.add_argument('--no-ablation', action='store_true', help='Disable ablation studies (weight decay, mu/nu)')
    args = parser.parse_args()

    # If --no-ablation is not set, always run ablations
    if not hasattr(args, 'run_ablation') or args.run_ablation:
        args.run_ablation = not getattr(args, 'no_ablation', False)

    # Set global variables from argparse (for legacy code compatibility)
    global LATENT_DIM, ACTIVATION_TYPE, K, L1_LAMBDA, L1_LAMBDA_PAM_RELU, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, EM_BATCH_SIZE, RESULTS_FOLDER, MU_DEC, NU_DEC, DECODER_REG, ALPHA_W, BETA_B, DECODER_STAY_CLOSE
    LATENT_DIM = args.latent_dim
    ACTIVATION_TYPE = args.activation_type
    K = args.k
    L1_LAMBDA = args.l1_lambda
    L1_LAMBDA_PAM_RELU = args.l1_lambda_pam_relu
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    EM_BATCH_SIZE = args.em_batch_size
    #RESULTS_FOLDER = args.results_folder
    MU_DEC = args.mu_dec
    NU_DEC = args.nu_dec
    DECODER_REG = args.decoder_reg
    ALPHA_W = args.alpha_w
    BETA_B = args.beta_b
    DECODER_STAY_CLOSE = args.decoder_stay_close

    print("\n===== LLM Activations: SGD vs PAM-SGD Comparison =====")
    ensure_folder(RESULTS_FOLDER)
    print(f"Results will be saved to: {RESULTS_FOLDER}")

    # Run the main comparison and get results/models
    sgd_results, em_results, sgd_model, em_model, train_dataset, test_dataset = run_comparison()

    ablation_results = None
    # By default, run ablation studies unless --no-ablation is specified
    if args.run_ablation:
        ablation_results = run_ablation_studies(train_dataset, test_dataset)

    # Save the final models, results, and ablation results
    torch.save({
        'sgd_model': sgd_model.state_dict(),
        'em_model': em_model.state_dict(),
        'sgd_results': sgd_results,
        'em_results': em_results,
        'ablation_results': ablation_results
    }, os.path.join(RESULTS_FOLDER, 'llm_comparison_results.pt'))

    print(f"\nAll experiments completed successfully. Results saved to: {RESULTS_FOLDER}")
    # Print final metrics
    sgd_final_loss = sgd_results['test_losses'][-1]
    em_final_loss = em_results['test_losses'][-1]
    improvement = (1 - em_final_loss/sgd_final_loss) * 100

    print("\n===== Final Results =====")
    print(f"SGD Final Test Loss: {sgd_final_loss:.6f}")
    print(f"PAM-SGD Final Test Loss: {em_final_loss:.6f}")
    print(f"Improvement: {improvement:.2f}%")

    print(f"Note: PAM-SGD-RELU performance is improved by using a reduced L1_LAMBDA_PAM_RELU={L1_LAMBDA_PAM_RELU}")
    print(f"compared to the standard L1_LAMBDA={L1_LAMBDA} to prevent excessive regularization penalties")

    sgd_total_time = sgd_results['total_time']
    em_total_time = em_results['total_time']
    time_diff = (1 - em_total_time/sgd_total_time) * 100 if em_total_time < sgd_total_time else -(sgd_total_time/em_total_time - 1) * 100

    print(f"\nSGD Total Training Time: {sgd_total_time:.2f}s")
    print(f"PAM-SGD Total Training Time: {em_total_time:.2f}s")
    print(f"Time Difference: {time_diff:.2f}%")

    if ablation_results is not None:
        print("\n===== Ablation Study Highlights =====")
        # Report best configuration from each ablation study
        if "activation" in ablation_results:
            sgd_act, em_act = ablation_results["activation"]
            best_sgd_act = min(sgd_act, key=lambda x: x[1]["test_losses"][-1])
            best_em_act = min(em_act, key=lambda x: x[1]["test_losses"][-1])
            print(f"Best activation type - SGD: {best_sgd_act[0].upper()}, PAM-SGD: {best_em_act[0].upper()}")
        if "weight_decay" in ablation_results:
            sgd_wd, em_wd = ablation_results["weight_decay"]
            best_sgd_wd = min(sgd_wd, key=lambda x: x[1]["test_losses"][-1])
            best_em_wd = min(em_wd, key=lambda x: x[1]["test_losses"][-1])
            print(f"Best weight decay - SGD: {best_sgd_wd[0]}, PAM-SGD: {best_em_wd[0]}")
        if "mu_nu" in ablation_results:
            mu_nu_results = ablation_results["mu_nu"]
            best_mu_nu = min(mu_nu_results, key=lambda x: x[1]["test_losses"][-1])
            print(f"Best μ/ν parameters for PAM-SGD: {best_mu_nu[0]}")

if __name__ == "__main__":
    main()
