# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import openpnm as op
from data_gen import build_network_matrix
import matplotlib.pyplot as plt

# preprocessing first (std mean norm)
data = np.load('training_data.npz')
X = data['X']
Y = data['Y']

# Separate X into pore and throat channels
n_samples = len(X)
size = X.shape[1]  # 5 for 3x3 network

X_pores = np.zeros_like(X)
X_throats = np.zeros_like(X)

# Extract pore values (even-even positions)
for i in range(0, size, 2):
    for j in range(0, size, 2):
        X_pores[:, i, j] = X[:, i, j]

# Extract throat values (odd-even and even-odd positions)
for i in range(size):
    for j in range(size):
        if (i % 2) != (j % 2):  # One even, one odd
            X_throats[:, i, j] = X[:, i, j]

# Stack into 2-channel output
X = np.stack([X_pores, X_throats], axis=1)  # Shape: (n, 2, 5, 5)

# Normalize Y (input to decoder)
Y_mean, Y_std = Y.mean(), Y.std()
Y = (Y - Y_mean) / (Y_std + 1e-8)

# Normalize X (output of decoder)
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / (X_std + 1e-8)

n_ipc_points = len(Y[0])*2
Y_flat = Y.reshape(n_samples, n_ipc_points)

Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)

indices  = torch.randperm(len(Y_tensor))
Y_tensor = Y_tensor[indices]
X_tensor = X_tensor[indices]

split = int(0.8 * len(Y_tensor))

Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]
X_train, X_test = X_tensor[:split], X_tensor[split:]

train_dataset = torch.utils.data.TensorDataset(Y_train, X_train)
test_dataset  = torch.utils.data.TensorDataset(Y_test,  X_test)

train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=16, shuffle=False)

class ConvDecoder(nn.Module):
    def __init__(self, n_ipc_points, output_size):
        super().__init__()

        # Transposed convolutions to upsample to (2, output_size, output_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=3, padding=2),
        )

        # Calculate bottleneck size by summing shrinkage from each ConvTranspose layer
        # For ConvTranspose2d with stride=1: H_out = H_in + kernel_size - 2*padding - 1
        # Working backwards: H_bottleneck = H_output - sum(kernel_size - 2*padding - 1)
        total_shrinkage = sum(
            layer.kernel_size[0] - 2 * layer.padding[0] - 1
            for layer in self.deconv 
            if isinstance(layer, nn.ConvTranspose2d)
        )
        bottleneck_spatial = output_size - total_shrinkage
        bottleneck_flat_size = 128 * bottleneck_spatial * bottleneck_spatial

        # Expand IPC curve to spatial feature maps
        self.fc = nn.Sequential(
            nn.Linear(n_ipc_points, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_flat_size),
            nn.ReLU(),
        )

        self.bottleneck_spatial = bottleneck_spatial
        self.output_size = output_size

    def forward(self, y):
        x = self.fc(y)
        x = x.view(x.size(0), 128, self.bottleneck_spatial, self.bottleneck_spatial)
        x = self.deconv(x)
        # Crop to ensure (2, output_size, output_size)
        return x[:, :, :self.output_size, :self.output_size]

model     = ConvDecoder(n_ipc_points=n_ipc_points, output_size=size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── Train ────────────────────────────────────────────────────────────────────────
for epoch in range(25):

    # — Training —
    model.train()
    train_loss = 0
    for y_batch, x_batch in train_loader:
        optimizer.zero_grad()
        pred = model(y_batch)
        loss = criterion(pred, x_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # — Testing —
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for y_batch, x_batch in test_loader:
            pred = model(y_batch)
            test_loss += criterion(pred, x_batch).item()

    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {train_loss/len(train_loader):.6f} | "
          f"Test Loss:  {test_loss/len(test_loader):.6f}")

# ── Visual Evaluation: Predict networks from IPC curves ────────────────────────────────────────
model.eval()
Np = (size + 1) // 2

# Create results directory
import os
os.makedirs('results/decoder_predictions', exist_ok=True)

# Collect all predictions
all_predictions = []

for i in range(10):
    pn_test = op.network.Demo(shape=[Np, Np, 1], spacing=1e-4)
    air_test = op.phase.Air(network=pn_test)
    air_test['pore.contact_angle'] = 120
    air_test['pore.surface_tension'] = 0.072
    air_test.add_model(propname='throat.entry_pressure', model=op.models.physics.capillary_pressure.washburn,
                       surface_tension='throat.surface_tension',
                       contact_angle='throat.contact_angle',
                       diameter='throat.diameter')

    ip_test = op.algorithms.InvasionPercolation(network=pn_test, phase=air_test)
    ip_test.set_inlet_BC(pores=pn_test.pores('left'))
    ip_test.run()
    data_test = ip_test.pc_curve()

    # True network
    X_true = build_network_matrix(pn_test, Np=Np)
    X_true_pores = np.zeros_like(X_true)
    X_true_throats = np.zeros_like(X_true)
    
    for ii in range(0, size, 2):
        for jj in range(0, size, 2):
            X_true_pores[ii, jj] = X_true[ii, jj]
    
    for ii in range(size):
        for jj in range(size):
            if (ii % 2) != (jj % 2):
                X_true_throats[ii, jj] = X_true[ii, jj]
    
    X_true_dual = np.stack([X_true_pores, X_true_throats], axis=0)

    # Predicted network from IPC curve
    Y_test_norm = (np.column_stack([data_test.pc, data_test.snwp]) - Y_mean) / (Y_std + 1e-8)
    Y_test_flat = Y_test_norm.reshape(1, n_ipc_points)
    Y_test_tensor = torch.tensor(Y_test_flat, dtype=torch.float32)

    with torch.no_grad():
        X_pred_norm = model(Y_test_tensor).squeeze().numpy()

    X_pred = X_pred_norm * X_std + X_mean

    # Store prediction data
    all_predictions.append({
        'X_true': X_true_dual,
        'X_pred': X_pred,
        'pc': data_test.pc,
        'snwp': data_test.snwp
    })

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Decoder Prediction {i+1}', fontsize=13)

    # True pores
    axes[0, 0].imshow(X_true_dual[0], cmap='viridis')
    axes[0, 0].set_title('True Pores')
    axes[0, 0].axis('off')

    # Predicted pores
    axes[0, 1].imshow(X_pred[0], cmap='viridis')
    axes[0, 1].set_title('Predicted Pores')
    axes[0, 1].axis('off')

    # True throats
    axes[1, 0].imshow(X_true_dual[1], cmap='viridis')
    axes[1, 0].set_title('True Throats')
    axes[1, 0].axis('off')

    # Predicted throats
    axes[1, 1].imshow(X_pred[1], cmap='viridis')
    axes[1, 1].set_title('Predicted Throats')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'results/decoder_predictions/prediction_{i:03d}.png', dpi=150)
    plt.show()

# Save all predictions once
np.savez('results/decoder_predictions/all_predictions.npz', 
         **{f'pred_{i}': p for i, p in enumerate(all_predictions)})

print("All predictions saved to results/decoder_predictions/")
