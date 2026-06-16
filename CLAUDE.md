# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning research project learning bidirectional mappings between **Pore Network Models (PNM)** and **Invasion Percolation Curves (IPC)**:

- **Encoder (forward):** PNM geometry → IPC curve (CNN regression)
- **Decoder (inverse):** IPC curve → reconstructed PNM geometry (transposed CNN)

Physics context: a pore network is a lattice of connected pores and throats. Running an Invasion Percolation drainage simulation produces a PC curve (capillary pressure vs. non-wetting phase saturation). All networks use OpenPNM's `Demo` type with air/water physics (contact angle 120°, Washburn entry pressure model). Two network sizes are active: 3×3 (`Np=3`) and 10×10 (`Np=10`).

Real-world use case: take an experimental IPC curve as input, predict the pore network that produced it. The decoder's success criterion is **functional fidelity** — the predicted network, when re-simulated via OpenPNM, reproduces the input IPC. Geometric fidelity (predicted matrix ≈ true matrix) is a secondary metric used during training diagnostics only, since the PNM→IPC mapping is many-to-one and no unique ground truth network exists for a given IPC.

## File Structure

Active scripts:
```
models.py          # ConvEncoder and ConvDecoder classes — import from here
data_gen.py        # generates training_data.npz via OpenPNM simulation
train_encoder.py   # trains encoder → results/encoder.pth + results/encoder_stats.npz
eval_encoder.py    # evaluates encoder → results/eval_encoder/
train_decoder.py   # trains decoder (Option B loss) → results/decoder.pth
eval_decoder.py    # evaluates decoder → results/eval_decoder/
ipc_compare.py     # utility: build_network_object (matrix → OpenPNM), build_network_matrix
```

Legacy scripts (`cnn_encoder.py`, `cnn_decoder_dual_channel.py`, etc.) are kept for reference but superseded by the above. Do not import from them.

## Run Order

```
python data_gen.py        # generate training_data.npz (slow, parallelized)
python train_encoder.py   # train encoder, saves .pth and stats
python eval_encoder.py    # evaluate encoder on fresh networks, saves plots
python train_decoder.py   # train decoder using frozen encoder as perceptual loss
python eval_decoder.py    # evaluate decoder, saves IPC comparison + network overlay plots
```

Each size (3×3, 10×10) uses separate filenames for data, model weights, and stats — see I/O section in each training script.

## Architecture — `models.py`

### Network size config (`_ARCH_CONFIGS`)

All spatial architecture decisions live in a single dict keyed by matrix spatial size (`2*Np - 1`). To support a new network size add one entry — nothing else changes:

```python
_ARCH_CONFIGS = {
    5: {  # Np=3, 3×3 network, 5×5 matrix
        'channels': [16, 32, 64, 128],
        'encoder' : [(3,1,1), (3,1,2), (3,1,2), (2,0,1)],  # 5→5→3→2→1
        'decoder' : [(2,0,1), (3,1,2), (3,1,2), (3,1,1)],  # 1→2→3→5→5
    },
    9: {  # Np=5, 5×5 network, 9×9 matrix
        'channels': [16, 32, 64, 128],
        'encoder' : [(3,1,1), (3,1,2), (3,1,2), (3,0,1)],  # 9→9→5→3→1
        'decoder' : [(3,0,1), (3,1,2), (3,1,2), (3,1,1)],  # 1→3→5→9→9
    },
    19: {  # Np=10, 10×10 network, 19×19 matrix
        'channels': [32, 64, 128, 256, 256],
        'encoder' : [(3,1,1), (5,2,2), (4,1,2), (3,1,2), (3,0,1)],  # 19→19→10→5→3→1
        'decoder' : [(3,0,1), (3,1,2), (4,1,2), (5,2,2), (3,1,1)],  # 1→3→5→10→19→19
    },
}
```

**Critical rule:** `len(channels)` must equal `len(encoder)` (same as `len(decoder)`). In the constructor, `in_channels` is prepended to channels to form pairs — if lengths mismatch, an IndexError will occur at model construction time. This will surface immediately on the first forward pass.

Encoder conv layers compress spatial to 1×1 while growing channels. Decoder conv layers expand from 1×1 back to output size while shrinking channels. FC layers bridge encoder↔IPC and IPC↔decoder bottleneck.

### IPC representation

IPC curves are always `(2, n_steps)` — **channel 0 is Pc, channel 1 is Snwp**. Never flattened to a 1D vector. `n_steps = Np_side² + 2*Np_side*(Np_side-1)` (pores + throats): 21 for 3×3, 280 for 10×10.

### Network matrix encoding

A `Np×Np` network → `(2*Np-1) × (2*Np-1)` matrix, split into two channels:
- **Channel 0 (pores):** pore diameters at even-row, even-col positions
- **Channel 1 (throats):** throat diameters at even-odd / odd-even positions
- Odd-odd positions are always zero (structurally meaningless; `build_network_object` skips them)

Input to encoder is always `(2, size, size)`.

## Normalization

Stats are saved to `results/encoder_stats.npz` at training time and loaded at eval time — eval scripts never need `training_data.npz`.

- **X (network matrix):** global mean/std across the full dual-channel array
- **Y (IPC curve):** per-channel — Pc and Snwp normalized independently using `axis=(0,2)` on `(n, 2, n_steps)`. Pc (~1k–20k Pa) and Snwp (0–1) have incompatible scales.

`encoder_stats.npz` keys: `X_mean`, `X_std`, `Y_mean` (shape `(2,)`), `Y_std` (shape `(2,)`), `n_steps`, `input_size`.

## Decoder Loss Function — Option B (implemented)

`train_decoder.py` uses:
```
loss = MSE(decoder(ipc), true_network_matrix)
     + LAMBDA * MSE(encoder(decoder(ipc)), ipc)
```
`LAMBDA = 1.0` at the top of the file. Encoder is loaded from `results/encoder.pth`, frozen (no grad). Both loss terms operate in normalized space. The perceptual term uses the encoder as a differentiable physics proxy — it penalizes predicted networks that would produce the wrong IPC without requiring OpenPNM during training.

Option A (plain geometry MSE only) was rejected: the many-to-one nature of PNM→IPC means MSE on geometry alone encourages the decoder to output an averaged network that satisfies no individual case well.

## eval_decoder.py — Network Overlay Plot

The network comparison plot draws true (blue) and predicted (red) networks on the same axes with `alpha=0.5`. Overlap appears purple — a perfectly predicted element is fully purple; overshoot extends red past blue; undershoot vice versa. Pores are `matplotlib.patches.Circle` in data units; throats are `matplotlib.patches.Rectangle` in data units. A single shared `d_max` normalizes both pore and throat sizes so physical diameter ratios are preserved on screen.

## Known Issues and Open Design Questions

### FC bottleneck in encoder for large networks (active problem)

The encoder FC head is:
```python
nn.Linear(flat_size, flat_size // 2)
nn.Linear(flat_size // 2, 2 * n_steps)
```
For 3×3: `flat_size=128`, hidden=64, output=42 — **no bottleneck** (64 > 42). For 10×10 with current config: `flat_size=256`, hidden=128, output=560 — **hidden layer is a bottleneck** (128 < 560). This limits how well the encoder can represent the full IPC curve regardless of channel depth.

**Observed effect:** 10×10 encoder plateaus at ~0.023 test loss (vs ~0.003 for 3×3). More data (50k → 500k samples) gave less than 5% improvement, confirming the bottleneck is architectural not data-limited.

**Two proposed fixes (not yet implemented):**

1. **Massive bottleneck channels** — `channels[-1] >= 560` so `flat_size//2 > 2*n_steps`. Requires wide channels like `[64, 128, 256, 512, 1024]`. Expensive to train.

2. **Stop spatial compression at 3×3 instead of 1×1** — remove the final `(3,0,1)` encoder layer so `flat_size = 3×3×256 = 2304`, giving FC `2304 → 1152 → 560` with no bottleneck. More principled. Requires changing `ConvDecoder` which currently hardcodes starting from `1×1` in its forward pass: `x = x.view(x.size(0), self.bottleneck_ch, 1, 1)`.

### data_gen.py memory usage for large sample counts

`joblib.Parallel` returns a list by default — all results sit in RAM simultaneously alongside the pre-allocated arrays. For 500k samples at 10×10, peak RAM is ~7 GB. Fix: add `return_as='generator'` to the `Parallel` call (joblib ≥ 1.2). Not yet applied to the script.

### Odd-odd positions (deferred)

4 of 25 positions in the 5×5 matrix (and 81 of 361 in the 19×19 matrix) are structurally zero. The decoder wastes output capacity learning to predict these. Fixing requires changing the representation (flat vector of active values only), which loses the spatial structure CNNs exploit. Deferred.

### VAE variant (deferred)

A `VariationalConvDecoder` exists in legacy scripts. When ported to `models.py` it should be a subclass of `ConvDecoder`, not a separate class. The only training difference is one extra KL loss term.

### Conv1d for larger networks (future work)

Since IPC is `(2, n_steps)`, Conv1d could process along `n_steps` in the decoder input, exploiting the sequential/monotonic structure of the curve. Marginal benefit at n_steps=21; meaningful at n_steps=280.

## Dependencies

```
pip install -r requirements.txt
```
Core: `torch`, `openpnm`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`.
