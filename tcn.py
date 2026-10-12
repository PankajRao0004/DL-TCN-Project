
"""
A small Temporal Convolutional Network (TCN) implementation.

Input shape:  (batch_size, input_channels, sequence_length)
Output shape: (batch_size, output_channels, sequence_length)

The model uses causal dilated convolutions and residual blocks.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """A 1D convolution that cannot look at future time steps."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, use_weight_norm=True):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

        # Weight normalization is used in the reference paper.
        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x):
        # Pad only on the left, so no future input can affect the output.
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Two causal convolutions with a residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, dropout=0.1, use_weight_norm=True):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size,
            dilation, use_weight_norm
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size,
            dilation, use_weight_norm
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Match channel counts before adding the residual.
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.residual_projection = nn.Identity()

        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual_projection(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Both tensors now have the same shape.
        return self.final_relu(out + residual)


class TemporalConvNet(nn.Module):
    """Stack temporal blocks with exponentially increasing dilation."""

    def __init__(self, input_channels, channels,
                 kernel_size=3, dropout=0.1,
                 use_weight_norm=True):
        super().__init__()

        if not channels or any(c <= 0 for c in channels):
            raise ValueError("channels must contain positive integers")
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.channels = list(channels)
        self.kernel_size = kernel_size

        blocks = []
        in_channels = input_channels

        for level, out_channels in enumerate(self.channels):
            dilation = 2 ** level

            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )

            in_channels = out_channels

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        return self.network(x)

    @property
    def receptive_field(self):
        # Two convolutions per block; dilation doubles at each level.
        return 1 + 2 * (self.kernel_size - 1) * (
            2 ** len(self.channels) - 1
        )
