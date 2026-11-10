"""1D CNN classifier and a CNN-LSTM hybrid over one-hot encoded DNA."""

from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


class CNNClassifier(nn.Module):
    """One-hot DNA (batch, length, 4) -> stacked conv blocks -> global mean+max pool -> logit."""

    def __init__(
        self, channels: int = 64, kernel_size: int = 9, num_layers: int = 3, dropout: float = 0.2
    ):
        super().__init__()
        layers = [conv_block(4, channels, kernel_size)]
        for _ in range(num_layers - 1):
            layers += [nn.MaxPool1d(2), conv_block(channels, channels, kernel_size)]
        self.convs = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x.transpose(1, 2))  # Conv1d wants (batch, channels, length)
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)


class CNNLSTMClassifier(nn.Module):
    """A conv layer finds local motifs, then a BiLSTM models their order."""

    def __init__(
        self, channels: int = 64, kernel_size: int = 9, hidden_size: int = 64, dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Sequential(conv_block(4, channels, kernel_size), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(channels, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4 * hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # back to (batch, length, channels)
        states, _ = self.lstm(h)
        pooled = torch.cat([states.mean(dim=1), states.max(dim=1).values], dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)
