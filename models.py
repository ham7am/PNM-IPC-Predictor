import torch
import torch.nn as nn


# Keyed by matrix spatial size (2*Np - 1).
#
# 'channels' : intermediate channel progression, shared by encoder and decoder.
# 'encoder'  : (kernel, padding, stride) per Conv2d layer.
#              Spatial must compress from input_size down to 1×1.
# 'decoder'  : (kernel, padding, stride) per ConvTranspose2d layer.
#              Spatial must expand from 1×1 back up to output_size.
#              Must be the spatial mirror of 'encoder'.
#
# To support a new network size, add one entry here — nothing else changes.
_ARCH_CONFIGS = {
    5: {  # Np=3,  3×3 network,  5×5 matrix
        'channels': [16, 32, 64, 128],
        'encoder' : [(3, 1, 1), (3, 1, 2), (3, 1, 2), (2, 0, 1)],  # 5→5→3→2→1
        'decoder' : [(2, 0, 1), (3, 1, 2), (3, 1, 2), (3, 1, 1)],  # 1→2→3→5→5
    },
    9: {  # Np=5,  5×5 network,  9×9 matrix
        'channels': [16, 32, 64, 128],
        'encoder' : [(3, 1, 1), (3, 1, 2), (3, 1, 2), (3, 0, 1)],  # 9→9→5→3→1
        'decoder' : [(3, 0, 1), (3, 1, 2), (3, 1, 2), (3, 1, 1)],  # 1→3→5→9→9
    },
    19: {  # Np=10, 10×10 network, 19×19 matrix
        'channels': [32, 64, 128, 256, 256],  # 5 entries for 5 layers
        'encoder' : [(3,1,1), (5,2,2), (4,1,2), (3,1,2), (3,0,1)],  # 19→19→10→5→3→1
        'decoder' : [(3,0,1), (3,1,2), (4,1,2), (5,2,2), (3,1,1)],  # 1→3→5→10→19→19
    },
}


class ConvEncoder(nn.Module):
    """
    CNN encoder: pore network matrix → 2-channel IPC curve.

    Conv layers compress spatial to 1×1 while growing channels (abstract features).
    FC projects the bottleneck to the IPC latent representation.

    Input shape:  (batch, in_channels, input_size, input_size)
    Output shape: (batch, 2, n_steps)  — channel 0: Pc, channel 1: Snwp

    Parameters
    ----------
    n_steps    : number of points on the IPC curve
    in_channels: 2 for dual-channel (pore + throat), 1 for merged single-channel
    input_size : spatial size of the square input matrix (5 for a 3×3 network)
    """
    def __init__(self, n_steps: int, in_channels: int = 2, input_size: int = 5):
        super().__init__()

        if input_size not in _ARCH_CONFIGS:
            raise ValueError(
                f"input_size={input_size} not in _ARCH_CONFIGS. "
                f"Supported sizes: {list(_ARCH_CONFIGS)}. "
                "Add an entry to _ARCH_CONFIGS to support this size."
            )

        cfg = _ARCH_CONFIGS[input_size]
        channels = [in_channels] + cfg['channels']

        conv_layers = []
        for i, (k, p, s) in enumerate(cfg['encoder']):
            conv_layers += [
                nn.Conv2d(channels[i], channels[i + 1], k, padding=p, stride=s),
                nn.ReLU(),
            ]
        self.conv = nn.Sequential(*conv_layers)

        dummy = torch.zeros(1, in_channels, input_size, input_size)
        flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, flat_size // 2),
            nn.ReLU(),
            nn.Linear(flat_size // 2, 2 * n_steps),
        )

        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.conv(x))
        return out.view(out.size(0), 2, self.n_steps)


class ConvDecoder(nn.Module):
    """
    Transposed CNN decoder: 2-channel IPC curve → pore network matrix.

    FC expands the IPC latent to a (bottleneck_ch, 1, 1) feature map.
    Deconv layers grow spatial while shrinking channels back to concrete features.

    Input shape:  (batch, 2, n_steps)  — channel 0: Pc, channel 1: Snwp
    Output shape: (batch, out_channels, output_size, output_size)

    Parameters
    ----------
    n_steps     : number of points on the IPC curve
    output_size : spatial size of the square output matrix (5 for a 3×3 network)
    out_channels: 2 for dual-channel output (pore + throat)
    """
    def __init__(self, n_steps: int, output_size: int = 5, out_channels: int = 2):
        super().__init__()

        if output_size not in _ARCH_CONFIGS:
            raise ValueError(
                f"output_size={output_size} not in _ARCH_CONFIGS. "
                f"Supported sizes: {list(_ARCH_CONFIGS)}. "
                "Add an entry to _ARCH_CONFIGS to support this size."
            )

        cfg = _ARCH_CONFIGS[output_size]
        bottleneck_ch = cfg['channels'][-1]

        self.fc = nn.Sequential(
            nn.Linear(2 * n_steps, bottleneck_ch // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_ch // 2, bottleneck_ch),
            nn.ReLU(),
        )

        channels = list(reversed(cfg['channels'])) + [out_channels]

        deconv_layers = []
        for i, (k, p, s) in enumerate(cfg['decoder']):
            deconv_layers.append(
                nn.ConvTranspose2d(channels[i], channels[i + 1], k, padding=p, stride=s)
            )
            if i < len(cfg['decoder']) - 1:
                deconv_layers.append(nn.ReLU())
        self.deconv = nn.Sequential(*deconv_layers)

        self.bottleneck_ch = bottleneck_ch
        self.output_size = output_size

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_flat = y.view(y.size(0), -1)
        x = self.fc(y_flat)
        x = x.view(x.size(0), self.bottleneck_ch, 1, 1)
        x = self.deconv(x)
        return x[:, :, :self.output_size, :self.output_size]
