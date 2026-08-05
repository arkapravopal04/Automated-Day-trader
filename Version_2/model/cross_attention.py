"""
model/cross_attention.py
Multi-head attention block applied ACROSS tickers, after per-asset
lstm_encoder.py encoding. One shared MHA block (not one per ticker-pair) --
every asset's encoded state attends to every other asset's encoded state
through the same learned projections, giving the network portfolio-
correlation awareness (sector co-movement, risk-on/off regime signal)
without an O(n_tickers^2) parameter blowup.
Shape convention: in this project n_envs == n_tickers (vec_trading_env.py),
so the natural "sequence" axis for this block is the ticker axis at a single
point in time, and "batch" is whatever else you batch over (e.g. multiple
parallel time-offsets during training). Concretely:
    input:  [batch, n_tickers, embed_dim]   (embed_dim == lstm_encoder.py's
             output_dim -- typically each stream's last LSTM timestep, or a
             pooled summary of its window)
    output: [batch, n_tickers, embed_dim]   (same shape -- a standard
             transformer-style block: MHA + residual + LayerNorm, then a
             small feedforward + residual + LayerNorm)
A 2D input [n_tickers, embed_dim] (batch implicitly 1 -- e.g. a single live
decision step) is also accepted and unsqueezed/squeezed automatically.
"""
from typing import Optional
import torch
import torch.nn as nn
class CrossAssetAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        ffn_dim = ffn_dim if ffn_dim is not None else embed_dim * 4
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    def forward(
        self,
        x: torch.Tensor,
        ticker_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [batch, n_tickers, embed_dim] or [n_tickers, embed_dim]
        ticker_mask: optional bool [batch, n_tickers] (or [n_tickers] for the
            2D input case), True where that ticker slot should be EXCLUDED
            from attention this step -- e.g. a halted/delisted ticker that
            shouldn't leak correlation signal into the others that day.
            Passed straight through to nn.MultiheadAttention's
            key_padding_mask.
        Returns the same shape as the input (post squeeze/unsqueeze).
        """
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(0)  # [1, n_tickers, embed_dim]
            if ticker_mask is not None and ticker_mask.dim() == 1:
                ticker_mask = ticker_mask.unsqueeze(0)
        attn_out, _ = self.mha(x, x, x, key_padding_mask=ticker_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x.squeeze(0) if squeeze_output else x