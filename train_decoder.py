import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import ConvEncoder, ConvDecoder

LAMBDA = 1.0  # weight of perceptual loss term relative to geometry loss

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Load normalization stats (do not recompute from data) ─────────────────────
stats      = np.load('results/encoder_stats.npz')
X_mean     = stats['X_mean']
X_std      = stats['X_std']
Y_mean     = stats['Y_mean']  # (2,) — per channel: [Pc_mean, Snwp_mean]
Y_std      = stats['Y_std']   # (2,) — per channel: [Pc_std,  Snwp_std]
n_steps    = int(stats['n_steps'])
input_size = int(stats['input_size'])

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load('training_data.npz')
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

# ── Normalize using encoder stats ─────────────────────────────────────────────
# X: global — same stats as encoder training, do not refit
X_norm = (X_dual - X_mean) / (X_std + 1e-8)

# Y: per-channel — same stats as encoder training, do not refit
Y = Y.transpose(0, 2, 1)  # (n, 2, n_steps) — ch0: Pc, ch1: Snwp
Y_norm = (Y - Y_mean[None, :, None]) / (Y_std[None, :, None] + 1e-8)

# ── Tensors and train/test split ──────────────────────────────────────────────
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

indices  = torch.randperm(n_samples)
X_tensor = X_tensor[indices]
Y_tensor = Y_tensor[indices]

split = int(0.8 * n_samples)
X_train, X_test = X_tensor[:split], X_tensor[split:]
Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]

# Input: Y (IPC curve), target: X (network matrix) — inverse of encoder
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Y_train, X_train),
    batch_size=16, shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Y_test, X_test),
    batch_size=16, shuffle=False,
)

# ── Frozen encoder for perceptual loss ────────────────────────────────────────
encoder = ConvEncoder(n_steps=n_steps, in_channels=2, input_size=input_size)
encoder.load_state_dict(torch.load('results/encoder.pth', weights_only=True))
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False

# ── Decoder, optimizer, loss ──────────────────────────────────────────────────
decoder   = ConvDecoder(n_steps=n_steps, output_size=input_size, out_channels=2)
optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"Training ConvDecoder — n_steps={n_steps}, output_size={input_size}, LAMBDA={LAMBDA}")
print(f"Train: {split} samples | Test: {n_samples - split} samples\n")

# ── Loss history ──────────────────────────────────────────────────────────────
n_epochs = 50
train_loss_hist = np.zeros(n_epochs)
test_loss_hist  = np.zeros(n_epochs)
train_geom_hist = np.zeros(n_epochs)
test_geom_hist  = np.zeros(n_epochs)
train_perc_hist = np.zeros(n_epochs)
test_perc_hist  = np.zeros(n_epochs)

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(n_epochs):
    decoder.train()
    train_loss = train_geom = train_perc = 0.0

    for y_batch, x_batch in train_loader:
        optimizer.zero_grad()
        pred_x    = decoder(y_batch)
        geom_loss = criterion(pred_x, x_batch)
        perc_loss = criterion(encoder(pred_x), y_batch)
        loss      = geom_loss + LAMBDA * perc_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_geom += geom_loss.item()
        train_perc += perc_loss.item()

    decoder.eval()
    test_loss = test_geom = test_perc = 0.0

    with torch.no_grad():
        for y_batch, x_batch in test_loader:
            pred_x    = decoder(y_batch)
            geom_loss = criterion(pred_x, x_batch)
            perc_loss = criterion(encoder(pred_x), y_batch)
            test_loss += (geom_loss + LAMBDA * perc_loss).item()
            test_geom += geom_loss.item()
            test_perc += perc_loss.item()

    n_tr = len(train_loader)
    n_te = len(test_loader)

    train_loss_hist[epoch] = train_loss / n_tr
    test_loss_hist[epoch]  = test_loss  / n_te
    train_geom_hist[epoch] = train_geom / n_tr
    test_geom_hist[epoch]  = test_geom  / n_te
    train_perc_hist[epoch] = train_perc / n_tr
    test_perc_hist[epoch]  = test_perc  / n_te

    print(f"Epoch {epoch + 1:3d} | "
          f"Train: {train_loss/n_tr:.6f} (geom {train_geom/n_tr:.6f}, perc {train_perc/n_tr:.6f}) | "
          f"Test:  {test_loss/n_te:.6f} (geom {test_geom/n_te:.6f}, perc {test_perc/n_te:.6f})")

# ── Save model and stats ──────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
torch.save(decoder.state_dict(), 'results/decoder.pth')
np.savez('results/decoder_stats.npz',
         train_loss=train_loss_hist, test_loss=test_loss_hist,
         train_geom=train_geom_hist, test_geom=test_geom_hist,
         train_perc=train_perc_hist, test_perc=test_perc_hist)
print("\nModel saved to results/decoder.pth")
print("Training stats saved to results/decoder_stats.npz")
