"""
model/lstm_encoder.py
Per-asset temporal encoder. Ported from Berserker with input dims wired to
cnn_encoder.py's output_dim rather than a hardcoded feature count.
Unlike cnn_encoder.py, this is explicitly STATEFUL across steps: the whole
point (per project notes) is that hidden state is carried across the
trading session rather than re-initialized from zero on every env.step().
The caller (policy / rollout loop) owns the hidden state tensors and is
responsible for:
    1. Calling init_hidden() once at the start of a session/episode.
    2. Passing the returned (h, c) back in on every subsequent forward().
    3. Calling reset_hidden(hidden, env_mask) at session boundaries for
       whichever streams just started a new session, WITHOUT disturbing the
       hidden state of streams that are still mid-session -- mirrors the
       env_mask-based partial-reset convention already used by
       PortfolioState.reset() in portfolio_state.py.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
Hidden = Tuple[torch.Tensor, torch.Tensor]  # (h_n, c_n), each [num_layers, batch, hidden_dim]
class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,          # must match cnn_encoder.CNNEncoder.output_dim
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim  # feed this into CrossAssetAttention(embed_dim=lstm.output_dim, ...)
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Hidden:
        device = device if device is not None else next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0
    def reset_hidden(self, hidden: Hidden, env_mask: torch.Tensor) -> Hidden:
        """
        Zeros the hidden/cell state for the batch rows in `env_mask` (bool
        [batch]) -- e.g. streams whose trading session just rolled over --
        leaving every other row's carried-over state untouched. Returns a
        new (h, c) tuple rather than mutating in place, so a caller holding
        a reference to the old tuple isn't silently surprised.
        """
        h, c = hidden
        env_mask = env_mask.to(h.device)
        h = h.clone()
        c = c.clone()
        h[:, env_mask, :] = 0.0
        c[:, env_mask, :] = 0.0
        return h, c
    def forward(self, x: torch.Tensor, hidden: Optional[Hidden] = None) -> Tuple[torch.Tensor, Hidden]:
        """
        x: [batch, window, input_dim] (cnn_encoder.py's output for this
           step's window)
        hidden: (h, c) carried from the previous call, or None to zero-init
           (e.g. the very first call of a session).
        Returns:
            output: [batch, window, hidden_dim] -- full per-timestep
                sequence, so cross_attention.py (or the policy head) can use
                either the last timestep or pool over the window.
            new_hidden: (h, c) to carry into the next call.
        """
        if hidden is None:
            hidden = self.init_hidden(batch_size=x.shape[0], device=x.device)
        output, new_hidden = self.lstm(x, hidden)
        return output, new_hidden