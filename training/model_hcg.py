"""Hierarchical CNN+GRU model for VIX mean-reversion prediction.

Secondary model (Phase 2d) - only trained if XGBoost leaves clear room
for improvement (>2% AUC gap on walk-forward validation).

Architecture: Three parallel streams processing different timescales,
merged before output heads.

Stream 1 - INTRADAY: Last 2 days, 5-min bars (~156 bars)
Stream 2 - SHORT-TERM: Last 2 weeks, hourly bars (~130 bars)
Stream 3 - REGIME: Last 2-3 months, daily bars (~60 bars)
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalStream(nn.Module):
    """A single temporal stream: Conv1D layers → GRU → output hidden state.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    conv_kernels : list[int]
        Kernel sizes for sequential Conv1D layers.
    conv_channels : int
        Number of output channels for each Conv1D layer.
    gru_hidden : int
        GRU hidden dimension.
    dropout : float
        Dropout rate applied after each Conv1D.
    """

    def __init__(
        self,
        n_features: int,
        conv_kernels: list[int],
        conv_channels: int = 32,
        gru_hidden: int = 48,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = n_features

        for kernel_size in conv_kernels:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels, conv_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(conv_channels))
            in_channels = conv_channels

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.gru_hidden = gru_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        Tensor of shape (batch, gru_hidden)
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Back to (batch, seq_len, channels) for GRU
        x = x.transpose(1, 2)

        _, h_n = self.gru(x)  # h_n: (1, batch, hidden)
        return h_n.squeeze(0)  # (batch, hidden)


class HierarchicalCNNGRU(nn.Module):
    """Hierarchical CNN+GRU model with three temporal streams.

    Parameters
    ----------
    intraday_features : int
        Number of features in the intraday (5-min) stream.
    shortterm_features : int
        Number of features in the short-term (hourly) stream.
    regime_features : int
        Number of features in the regime (daily) stream.
    """

    def __init__(
        self,
        intraday_features: int = 5,
        shortterm_features: int = 8,
        regime_features: int = 18,
        intraday_hidden: int = 48,
        shortterm_hidden: int = 48,
        regime_hidden: int = 32,
        merge_dropout: float = 0.3,
    ):
        super().__init__()

        # Stream 1: Intraday (5-min bars, ~156 steps)
        self.intraday_stream = TemporalStream(
            n_features=intraday_features,
            conv_kernels=[6, 24],  # ~30min, ~2hr patterns
            conv_channels=32,
            gru_hidden=intraday_hidden,
        )

        # Stream 2: Short-term (hourly bars, ~130 steps)
        self.shortterm_stream = TemporalStream(
            n_features=shortterm_features,
            conv_kernels=[4, 24],  # ~4hr, ~1day patterns
            conv_channels=32,
            gru_hidden=shortterm_hidden,
        )

        # Stream 3: Regime (daily bars, ~60 steps)
        self.regime_stream = TemporalStream(
            n_features=regime_features,
            conv_kernels=[5, 15],  # ~1week, ~3week patterns
            conv_channels=32,
            gru_hidden=regime_hidden,
        )

        merge_dim = intraday_hidden + shortterm_hidden + regime_hidden  # 128

        # Merge layers
        self.merge = nn.Sequential(
            nn.Linear(merge_dim, 64),
            nn.ReLU(),
            nn.Dropout(merge_dropout),
        )

        # Output heads
        self.head_revert = nn.Linear(64, 1)      # p_revert
        self.head_spike = nn.Linear(64, 1)        # p_spike_first
        self.head_magnitude = nn.Linear(64, 1)    # expected_magnitude

    def forward(
        self,
        x_intraday: torch.Tensor,
        x_shortterm: torch.Tensor,
        x_regime: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three streams.

        Parameters
        ----------
        x_intraday : (batch, 156, intraday_features)
        x_shortterm : (batch, 130, shortterm_features)
        x_regime : (batch, 60, regime_features)

        Returns
        -------
        p_revert : (batch, 1) - sigmoid probability
        p_spike_first : (batch, 1) - sigmoid probability
        expected_magnitude : (batch, 1) - continuous value
        """
        h_intraday = self.intraday_stream(x_intraday)
        h_shortterm = self.shortterm_stream(x_shortterm)
        h_regime = self.regime_stream(x_regime)

        merged = torch.cat([h_intraday, h_shortterm, h_regime], dim=1)
        features = self.merge(merged)

        p_revert = torch.sigmoid(self.head_revert(features))
        p_spike = torch.sigmoid(self.head_spike(features))
        magnitude = self.head_magnitude(features)

        return p_revert, p_spike, magnitude

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskLoss(nn.Module):
    """Multi-task loss for the three output heads.

    Combines:
      - BCE for p_revert
      - BCE for p_spike_first
      - MSE for expected_magnitude

    With tunable weights and optional focal loss for classification heads.
    """

    def __init__(
        self,
        w_revert: float = 1.0,
        w_spike: float = 1.0,
        w_magnitude: float = 0.5,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
    ):
        super().__init__()
        self.w_revert = w_revert
        self.w_spike = w_spike
        self.w_magnitude = w_magnitude
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

    def focal_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss variant of BCE for handling class imbalance."""
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def forward(
        self,
        p_revert: torch.Tensor,
        p_spike: torch.Tensor,
        magnitude: torch.Tensor,
        y_revert: torch.Tensor,
        y_spike: torch.Tensor,
        y_magnitude: torch.Tensor,
        mask_revert: torch.Tensor | None = None,
        mask_spike: torch.Tensor | None = None,
        mask_magnitude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute multi-task loss.

        Masks allow training on subsets where labels are available.
        """
        losses = {}

        # p_revert loss
        if mask_revert is not None:
            p_r = p_revert[mask_revert]
            y_r = y_revert[mask_revert]
        else:
            p_r, y_r = p_revert, y_revert

        if len(p_r) > 0:
            if self.use_focal:
                losses["revert"] = self.focal_bce(p_r, y_r.float())
            else:
                losses["revert"] = F.binary_cross_entropy(p_r, y_r.float())
        else:
            losses["revert"] = torch.tensor(0.0)

        # p_spike_first loss
        if mask_spike is not None:
            p_s = p_spike[mask_spike]
            y_s = y_spike[mask_spike]
        else:
            p_s, y_s = p_spike, y_spike

        if len(p_s) > 0:
            if self.use_focal:
                losses["spike"] = self.focal_bce(p_s, y_s.float())
            else:
                losses["spike"] = F.binary_cross_entropy(p_s, y_s.float())
        else:
            losses["spike"] = torch.tensor(0.0)

        # Magnitude loss
        if mask_magnitude is not None:
            m_p = magnitude[mask_magnitude]
            m_y = y_magnitude[mask_magnitude]
        else:
            m_p, m_y = magnitude, y_magnitude

        if len(m_p) > 0:
            losses["magnitude"] = F.mse_loss(m_p, m_y.float())
        else:
            losses["magnitude"] = torch.tensor(0.0)

        total = (
            self.w_revert * losses["revert"]
            + self.w_spike * losses["spike"]
            + self.w_magnitude * losses["magnitude"]
        )

        return total, {k: v.item() for k, v in losses.items()}


def create_model(
    intraday_features: int = 5,
    shortterm_features: int = 8,
    regime_features: int = 18,
) -> HierarchicalCNNGRU:
    """Create a Hierarchical CNN+GRU model with default architecture."""
    model = HierarchicalCNNGRU(
        intraday_features=intraday_features,
        shortterm_features=shortterm_features,
        regime_features=regime_features,
    )
    n_params = model.count_parameters()
    logger.info(f"Created HierarchicalCNNGRU with {n_params:,} parameters")
    return model
