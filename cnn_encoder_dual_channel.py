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

# Stack into 2-channel input
X = np.stack([X_pores, X_throats], axis=1)  # Shape: (n, 2, 5, 5)

# Normalize
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / (X_std + 1e-8)

# Normalize Y separately
Y_mean = Y.mean(axis=(0, 1))
Y_std  = Y.std(axis=(0, 1))    
Y = (Y - Y_mean) / (Y_std + 1e-8)

n_ipc_points = len(Y[0])*2
X_tensor = torch.tensor(X, dtype=torch.float32)    
Y_tensor = torch.tensor(Y, dtype=torch.float32).reshape(n_samples, n_ipc_points) 

indices  = torch.randperm(len(X_tensor))
X_tensor = X_tensor[indices]
Y_tensor = Y_tensor[indices]

split = int(0.8 * len(X_tensor))

X_train, X_test = X_tensor[:split], X_tensor[split:]
Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset  = torch.utils.data.TensorDataset(X_test,  Y_test)

train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=16, shuffle=False)

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(),
        )

        # Dummy forward pass to get actual flattened size
        dummy_input = torch.zeros(1, 2, size, size)
        dummy_output = self.conv(dummy_input)
        flat_size = dummy_output.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.head(self.conv(x))
    
model     = ConvEncoder(latent_dim=n_ipc_points)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ── Train ────────────────────────────────────────────────────────────────────────
for epoch in range(10):

    # — Training —
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # — Testing —
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch)
            test_loss += criterion(pred, y_batch).item()

    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {train_loss/len(train_loader):.6f} | "
          f"Test Loss:  {test_loss/len(test_loader):.6f}")

# ── Visual Evaluation on 5 new networks ────────────────────────────────────────
model.eval()
Np = (size + 1) // 2
for i in range(100):
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

    X_new = build_network_matrix(pn_test, Np=Np)
    
    # Separate into pore and throat channels
    X_new_pores = np.zeros_like(X_new)
    X_new_throats = np.zeros_like(X_new)
    
    for ii in range(0, size, 2):
        for jj in range(0, size, 2):
            X_new_pores[ii, jj] = X_new[ii, jj]
    
    for ii in range(size):
        for jj in range(size):
            if (ii % 2) != (jj % 2):
                X_new_throats[ii, jj] = X_new[ii, jj]
    
    X_new_dual = np.stack([X_new_pores, X_new_throats], axis=0)
    X_new_norm = (X_new_dual - X_mean) / (X_std + 1e-8)
    X_new_tensor = torch.tensor(X_new_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        Y_pred_norm = model(X_new_tensor).squeeze().numpy()

    Y_pred    = Y_pred_norm.reshape(Y.shape[1], 2) * Y_std + Y_mean
    pc_pred   = Y_pred[:, 0]
    snwp_pred = Y_pred[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Network {i+1}', fontsize=13)

    axes[0].plot(data_test.pc, data_test.snwp, c='green')
    axes[0].set_xlabel('Capillary Pressure'); axes[0].set_ylabel('Snwp')
    axes[0].set_title('True IP Curve')

    axes[1].plot(data_test.pc, data_test.snwp, c='green', label='True')
    axes[1].plot(pc_pred, snwp_pred,           c='red',   label='Predicted', linestyle='--')
    axes[1].set_xlabel('Capillary Pressure'); axes[1].set_ylabel('Snwp')
    axes[1].set_title('PC Curve'); axes[1].legend()

    plt.tight_layout()
    plt.show()
