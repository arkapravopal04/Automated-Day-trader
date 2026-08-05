"""
model/cnn_encoder.py

Per-asset local pattern extractor over short windows (candlestick
microstructure). Ported from the Berserker PPO stack; input dims are no
longer hardcoded to the NSE/US-equity Berserker feature set -- pass in
whatever `n_features` preprocess.py's MultiTickerRolloutDataset currently
produces via the constructor rather than assuming a fixed count here.

Runs identically and independently per asset: each stream (n_envs ==
n_tickers in vec_trading_env.py) is treated as an independent batch row,
consistent with how the rest of this project vectorizes across tickers.

Design:
    - Input:  [batch, window, n_features]  (batch_first, feature-last --
      the same layout MultiTickerRolloutDataset already produces)
    - Internally transposed to [batch, n_features, window] for Conv1d
      (channels = features, length = time)
    - A stack of CAUSAL 1D convolutions (left-padded, right-trimmed) with
      increasing dilation, so no bar ever sees a future bar -- this matters
      both for the RL rollout and for any offline supervised pretraining of
      this encoder against next-bar targets.
    - Output: [batch, window, cnn_channels] -- same time length as the
      input, NOT pooled down to a single vector, so lstm_encoder.py gets a
      full sequence to run its recurrence over rather than a single
      microstructure summary.
"""

from typing import List, Sequence

import torch
import torch.nn as nn


class CausalConv1dBlock(nn.Module):
    """
    One causal (left-padded, right-trimmed) conv1d + per-timestep channel
    LayerNorm + GELU + dropout, with a residual connection (via a 1x1
    projection when in_channels != out_channels, e.g. on the first block).

    NOTE on normalization: this deliberately does NOT use GroupNorm/BatchNorm
    over the (channel, time) volume. Those normalize using statistics
    aggregated across ALL timesteps in a sample, which means a FUTURE bar's
    value shifts the mean/std applied to a PAST bar's output -- silently
    breaking causality even though the convolution itself is causal. Verified
    empirically: with GroupNorm, perturbing only the last 5 timesteps of a
    20-step window changed every earlier timestep's output (max diff ~0.27);
    with the per-timestep LayerNorm below, that diff is exactly 0.0. LayerNorm
    here normalizes each timestep independently over the channel dimension
    only, so no cross-timestep leakage is possible regardless of window
    content.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.trim = (kernel_size - 1) * dilation  # amount of right-side padding to trim off to stay causal
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.trim)
        self.norm = nn.LayerNorm(out_channels)  # applied per-timestep over channels only, see note above
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        residual = self.residual_proj(x)
        out = self.conv(x)
        if self.trim > 0:
            out = out[..., : -self.trim]  # drop the right-side padding conv1d added -> causal
        # LayerNorm normalizes over the last dim, so temporarily move time to
        # the middle: [batch, channels, time] -> [batch, time, channels] ->
        # normalize (each timestep independent) -> back to [batch, channels, time]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class CNNEncoder(nn.Module):
    """
    Per-asset local pattern extractor. Stateless across steps (unlike
    lstm_encoder.py) -- every forward() call only looks at the window it's
    given, no hidden state carried between calls.
    """

    def __init__(
        self,
        n_features: int,
        channels: Sequence[int] = (64, 64, 128),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(dilations), (
            "channels, kernel_sizes, and dilations must all be the same length -- one entry per conv block"
        )
        blocks: List[nn.Module] = []
        in_ch = n_features
        for out_ch, k, d in zip(channels, kernel_sizes, dilations):
            blocks.append(CausalConv1dBlock(in_ch, out_ch, k, d, dropout))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = in_ch  # feed this into LSTMEncoder(input_dim=cnn.output_dim, ...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, window, n_features] -> returns [batch, window, output_dim]
        """
        h = x.transpose(1, 2)  # [batch, n_features, window]
        for block in self.blocks:
            h = block(h)
        return h.transpose(1, 2)  # [batch, window, output_dim]