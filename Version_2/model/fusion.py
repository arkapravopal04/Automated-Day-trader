"""
model/fusion.py
Combines cnn_encoder.py + lstm_encoder.py + cross_attention.py outputs into
one shared trunk representation, feeding both hybrid_policy.py and
dual_critic.py.
Pipeline this sits at the end of:
    CNNEncoder(x)                  -> cnn_seq   [batch, window, cnn_dim]
    cnn_last = cnn_seq[:, -1, :]                 [batch, cnn_dim]
    LSTMEncoder(cnn_seq, hidden)   -> lstm_seq   [batch, window, lstm_dim], new_hidden
    lstm_last = lstm_seq[:, -1, :]                [batch, lstm_dim]
    CrossAssetAttention(lstm_last) -> attn_out    [batch, lstm_dim]  (already
                                                     residual-connected to
                                                     lstm_last internally)
    FusionTrunk(cnn_last, attn_out) -> trunk      [batch, trunk_dim]
Two signals get fused:
    1. `attn_out` -- the cross-sectional, portfolio-correlation-aware state.
       This already carries the per-asset temporal state via its own
       internal residual connection, so it's the primary signal.
    2. `cnn_last` -- the CNN's own last-timestep local-pattern output, fed in
       as a skip path around the LSTM+attention stack, so fine-grained
       microstructure detail (e.g. a sharp intrabar reversal) isn't diluted
       by two rounds of temporal/cross-sectional averaging before it reaches
       the policy and critic heads.
"""
import torch
import torch.nn as nn
class FusionTrunk(nn.Module):
    def __init__(
        self,
        cnn_dim: int,
        lstm_dim: int,     # == cross_attention.py's embed_dim, since it operates on lstm_encoder.py's output
        trunk_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = cnn_dim + lstm_dim
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
        )
        self.norm = nn.LayerNorm(trunk_dim)
        self.output_dim = trunk_dim
    def forward(self, cnn_last: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """
        cnn_last: [batch, cnn_dim]  -- typically cnn_seq[:, -1, :]
        attn_out: [batch, lstm_dim] -- CrossAssetAttention's output for this step
        Returns: [batch, trunk_dim]
        """
        fused = torch.cat([cnn_last, attn_out], dim=-1)
        trunk = self.proj(fused)
        return self.norm(trunk)