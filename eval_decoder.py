import os
import numpy as np
import torch
import openpnm as op
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_gen import build_network_matrix
from ipc_compare import build_network_object
from models import ConvDecoder

def plot_network_overlay(ax, X_true, X_pred, Np, title):
    """
    Draw true (blue) and predicted (red) networks overlaid on one axes.
    Overlap renders as purple; overshoot extends red past blue; undershoot vice versa.

    X_true, X_pred : (2, size, size) arrays in physical (denormalized) space.
                     Channel 0 = pore diameters, channel 1 = throat diameters.
    Np             : number of pores along one side.
    """
    d_max = max(X_true[0].max(), X_true[1].max(),
                X_pred[0].max(), X_pred[1].max())

    def draw_throats(X, color, zorder):
        for ii in range(Np):
            for jj in range(Np):
                # Horizontal throat — between (ii,jj) and (ii,jj+1)
                if jj < Np - 1:
                    d = X[1, 2 * ii, 2 * jj + 1]
                    w = (d / d_max) * 0.7 if d_max > 0 else 0.0
                    rect = mpatches.Rectangle((jj, ii - w / 2), width=1.0, height=w,
                                              facecolor=color, edgecolor='none',
                                              alpha=0.5, zorder=zorder)
                    ax.add_patch(rect)
                # Vertical throat — between (ii,jj) and (ii+1,jj)
                if ii < Np - 1:
                    d = X[1, 2 * ii + 1, 2 * jj]
                    w = (d / d_max) * 0.7 if d_max > 0 else 0.0
                    rect = mpatches.Rectangle((jj - w / 2, ii), width=w, height=1.0,
                                              facecolor=color, edgecolor='none',
                                              alpha=0.5, zorder=zorder)
                    ax.add_patch(rect)

    def draw_pores(X, color, zorder):
        for ii in range(Np):
            for jj in range(Np):
                d = X[0, 2 * ii, 2 * jj]
                r = (d / d_max) * 0.35 if d_max > 0 else 0.1
                circle = mpatches.Circle((jj, ii), radius=r,
                                         facecolor=color, edgecolor='none',
                                         alpha=0.5, zorder=zorder)
                ax.add_patch(circle)

    draw_throats(X_true, color='blue', zorder=1)
    draw_throats(X_pred, color='red',  zorder=2)
    draw_pores(X_true,   color='blue', zorder=3)
    draw_pores(X_pred,   color='red',  zorder=4)

    ax.set_aspect('equal')
    ax.set_xlim(-0.6, Np - 1 + 0.6)
    ax.set_ylim(-0.6, Np - 1 + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    legend_handles = [
        mpatches.Patch(facecolor='blue', alpha=0.5, label='True'),
        mpatches.Patch(facecolor='red',  alpha=0.5, label='Predicted'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')


# ── Load stats and model ──────────────────────────────────────────────────────
stats      = np.load('results/encoder_stats.npz')
X_mean     = stats['X_mean']
X_std      = stats['X_std']
Y_mean     = stats['Y_mean']  # (2,) — per channel: [Pc_mean, Snwp_mean]
Y_std      = stats['Y_std']   # (2,) — per channel: [Pc_std,  Snwp_std]
n_steps    = int(stats['n_steps'])
input_size = int(stats['input_size'])

Np = (input_size + 1) // 2

model = ConvDecoder(n_steps=n_steps, output_size=input_size, out_channels=2)
model.load_state_dict(torch.load('results/decoder.pth', weights_only=True))
model.eval()
print(f"Loaded decoder — n_steps={n_steps}, output_size={input_size}, Np={Np}\n")

# ── Evaluate on fresh networks ────────────────────────────────────────────────
N = 10
os.makedirs('results/eval_decoder', exist_ok=True)

for i in range(N):
    # True network and IP simulation — this IPC is both ground truth and decoder input
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

    # Build dual-channel matrix from the true network (for the network comparison plot)
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

    X_true_dual = np.stack([X_pores, X_throats], axis=0)  # (2, size, size)

    # Build normalized IPC input for decoder — (1, 2, n_steps)
    Y_true      = np.stack([pc_true, snwp_true], axis=0)          # (2, n_steps)
    Y_norm      = (Y_true - Y_mean[:, None]) / (Y_std[:, None] + 1e-8)
    Y_tensor    = torch.tensor(Y_norm, dtype=torch.float32).unsqueeze(0)  # (1, 2, n_steps)

    # Predict network matrix and denormalize
    with torch.no_grad():
        X_pred_norm = model(Y_tensor).squeeze().numpy()            # (2, size, size)

    X_pred = X_pred_norm * (X_std + 1e-8) + X_mean
    X_pred = np.clip(X_pred, 1e-7, None)  # negative diameters crash OpenPNM

    # Reconstruct PNM from predicted matrix and re-simulate
    pn_pred = build_network_object((X_pred[0], X_pred[1]), Np)
    air_pred = op.phase.Air(network=pn_pred)
    air_pred['pore.contact_angle']   = 120
    air_pred['pore.surface_tension'] = 0.072
    air_pred.add_model(
        propname='throat.entry_pressure',
        model=op.models.physics.capillary_pressure.washburn,
        surface_tension='throat.surface_tension',
        contact_angle='throat.contact_angle',
        diameter='throat.diameter',
    )
    ip_pred = op.algorithms.InvasionPercolation(network=pn_pred, phase=air_pred)
    ip_pred.set_inlet_BC(pores=pn_pred.pores('left'))
    ip_pred.run()
    curve_pred = ip_pred.pc_curve()
    pc_pred    = curve_pred.pc
    snwp_pred  = curve_pred.snwp

    # IPC comparison plot — input IPC vs re-simulated from predicted network
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pc_true,  snwp_true,  'g-',  linewidth=2, label='Input IPC')
    ax.plot(pc_pred,  snwp_pred,  'r--', linewidth=2, label='Re-simulated from predicted network')
    ax.set_xlabel('Capillary Pressure (Pa)')
    ax.set_ylabel('Non-wetting Phase Saturation')
    ax.set_title(f'Decoder IPC Comparison — Sample {i + 1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/eval_decoder/ipc_comparison_{i:03d}.png', dpi=150)
    plt.close()

    # Network overlay plot — true (blue) and predicted (red) drawn on the same axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_network_overlay(ax, X_true_dual, X_pred, Np,
                         title=f'Network Overlay — Sample {i + 1}')
    plt.tight_layout()
    plt.savefig(f'results/eval_decoder/network_comparison_{i:03d}.png', dpi=150)
    plt.close()

    print(f"Sample {i + 1:2d} | saved to results/eval_decoder/ipc_comparison_{i:03d}.png, network_comparison_{i:03d}.png")

print("\nAll plots saved to results/eval_decoder/")
