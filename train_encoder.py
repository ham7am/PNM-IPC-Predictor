import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import ConvEncoder

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load('./data/10_10_1_training_data.npz')
X = data['X']  # (n, size, size)
Y = data['Y']  # (n, n_steps, 2)

n_samples = len(X)
size = X.shape[1]  # spatial size of the matrix (5 for a 3×3 network)

# ── Build dual-channel X: (n, 2, size, size) ─────────────────────────────────
X_pores   = np.zeros_like(X)
X_throats = np.zeros_like(X)

for i in range(0, size, 2):
    for j in range(0, size, 2):
        X_pores[:, i, j] = X[:, i, j]

for i in range(size):
    for j in range(size):
        if (i % 2) != (j % 2):
            X_throats[:, i, j] = X[:, i, j]

X_dual = np.stack([X_pores, X_throats], axis=1)  # (n, 2, size, size)

# ── Reformat Y to (n, 2, n_steps) ────────────────────────────────────────────
# channel 0: Pc, channel 1: Snwp
Y = Y.transpose(0, 2, 1)
n_steps = Y.shape[2]

# ── Normalize ─────────────────────────────────────────────────────────────────
# X: global — pore and throat diameters share the same physical scale
X_mean = X_dual.mean()
X_std  = X_dual.std()
X_dual = (X_dual - X_mean) / (X_std + 1e-8)

# Y: per-channel — Pc (Pascals, ~1k-20k) and Snwp (0-1) have very different scales
# To switch to global: Y_mean = Y.mean(); Y_std = Y.std() (scalar, no indexing needed)
Y_mean = Y.mean(axis=(0, 2))  # (2,) — one value per channel across all samples and steps
Y_std  = Y.std(axis=(0, 2))   # (2,)
Y = (Y - Y_mean[None, :, None]) / (Y_std[None, :, None] + 1e-8)

# ── Save normalization stats ──────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
np.savez('results/encoder_stats.npz',
         X_mean=X_mean, X_std=X_std,
         Y_mean=Y_mean, Y_std=Y_std,
         n_steps=n_steps, input_size=size)
print("Normalization stats saved to results/encoder_stats.npz")

# ── Tensors and train/test split ──────────────────────────────────────────────
X_tensor = torch.tensor(X_dual, dtype=torch.float32)
Y_tensor = torch.tensor(Y,      dtype=torch.float32)

indices  = torch.randperm(n_samples)
X_tensor = X_tensor[indices]
Y_tensor = Y_tensor[indices]

split = int(0.8 * n_samples)
X_train, X_test = X_tensor[:split], X_tensor[split:]
Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, Y_train),
    batch_size=16, shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, Y_test),
    batch_size=16, shuffle=False,
)

# ── Model, optimizer, loss ────────────────────────────────────────────────────
model     = ConvEncoder(n_steps=n_steps, in_channels=2, input_size=size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"Training ConvEncoder — n_steps={n_steps}, input_size={size}")
print(f"Train: {split} samples | Test: {n_samples - split} samples\n")

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            test_loss += criterion(model(x_batch), y_batch).item()

    print(f"Epoch {epoch + 1:3d} | "
          f"Train Loss: {train_loss / len(train_loader):.6f} | "
          f"Test Loss:  {test_loss / len(test_loader):.6f}")

# ── Save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), 'results/10_10_1_encoder.pth')
print("\nModel saved to results/10_10_1_encoder.pth")
