import os
import numpy as np
import torch
import openpnm as op
import matplotlib.pyplot as plt
from data_gen import build_network_matrix
from models import ConvEncoder

# ── Load stats and model ──────────────────────────────────────────────────────
stats      = np.load('results/encoder_stats.npz')
X_mean     = stats['X_mean']
X_std      = stats['X_std']
Y_mean     = stats['Y_mean']  # (2,) — per channel: [Pc_mean, Snwp_mean]
Y_std      = stats['Y_std']   # (2,) — per channel: [Pc_std,  Snwp_std]
n_steps    = int(stats['n_steps'])
input_size = int(stats['input_size'])

Np = (input_size + 1) // 2

model = ConvEncoder(n_steps=n_steps, in_channels=2, input_size=input_size)
model.load_state_dict(torch.load('results/encoder.pth', weights_only=True))
model.eval()
print(f"Loaded encoder — n_steps={n_steps}, input_size={input_size}, Np={Np}\n")

# ── Evaluate on fresh networks ────────────────────────────────────────────────
N = 10
os.makedirs('results/eval_encoder', exist_ok=True)

for i in range(N):
    # True network and IP simulation
    pn  = op.network.Demo(shape=[Np, Np, 1], spacing=1e-4)
    air = op.phase.Air(network=pn)
    air['pore.contact_angle']   = 120
    air['pore.surface_tension'] = 0.072
    air.add_model(
        propname='throat.entry_pressure',
        model=op.models.physics.capillary_pressure.washburn,
        surface_tension='throat.surface_tension',
        contact_angle='throat.contact_angle',
        diameter='throat.diameter',
    )
    ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
    ip.set_inlet_BC(pores=pn.pores('left'))
    ip.run()
    curve     = ip.pc_curve()
    pc_true   = curve.pc
    snwp_true = curve.snwp

    # Build dual-channel input matrix
    mat       = build_network_matrix(pn, Np=Np)
    X_pores   = np.zeros_like(mat)
    X_throats = np.zeros_like(mat)

    for r in range(0, input_size, 2):
        for c in range(0, input_size, 2):
            X_pores[r, c] = mat[r, c]

    for r in range(input_size):
        for c in range(input_size):
            if (r % 2) != (c % 2):
                X_throats[r, c] = mat[r, c]

    X_dual   = np.stack([X_pores, X_throats], axis=0)          # (2, size, size)
    X_norm   = (X_dual - X_mean) / (X_std + 1e-8)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)  # (1, 2, size, size)

    # Predict and denormalize
    with torch.no_grad():
        Y_pred_norm = model(X_tensor).squeeze().numpy()         # (2, n_steps)

    Y_pred    = Y_pred_norm * (Y_std[:, None] + 1e-8) + Y_mean[:, None]
    pc_pred   = Y_pred[0]  # Pc channel
    snwp_pred = Y_pred[1]  # Snwp channel

    # Plot true vs predicted
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pc_true,   snwp_true, 'g-',  linewidth=2, label='True')
    ax.plot(pc_pred,   snwp_pred, 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel('Capillary Pressure (Pa)')
    ax.set_ylabel('Non-wetting Phase Saturation')
    ax.set_title(f'Encoder Prediction — Sample {i + 1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/eval_encoder/prediction_{i:03d}.png', dpi=150)
    plt.close()

    print(f"Sample {i + 1:2d} | saved to results/eval_encoder/prediction_{i:03d}.png")

print("\nAll plots saved to results/eval_encoder/")
