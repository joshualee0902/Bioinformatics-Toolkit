"""Bidirectional LSTM classifier over one-hot encoded DNA."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    """One-hot DNA (batch, length, 4) -> BiLSTM -> mean+max pooling -> logit.

    Pooling over all positions (instead of using only the last hidden state)
    lets every position contribute evenly, which suits short fixed-length windows.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4 * hidden_size, 1)  # (mean, max) x (forward, backward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(x)  # (batch, length, 2 * hidden)
        pooled = torch.cat([states.mean(dim=1), states.max(dim=1).values], dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)
