"""
model/dual_critic.py
Value heads for long/short/flat, ported from Berserker's dual-critic setup
(see directional_bias_fix.pdf, item 6: a single shared critic tends to learn
a blended baseline dominated by whichever direction is more common in the
training data, which biases GAE advantages against the less-represented
direction and is one of the mechanisms behind directional collapse).
Three independent value heads instead of Berserker's original two (long,
short) -- FLAT gets its own head too, since hybrid_policy.py's discrete
stage is a genuine 3-way choice {SHORT, FLAT, LONG}, not just a long/short
toggle, and holding-flat has a materially different value profile
(no market exposure, no unrealized PnL risk) than either directional side.
"""
from typing import Dict
import torch
import torch.nn as nn
class DualCriticHead(nn.Module):
    def __init__(self, trunk_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        def make_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(trunk_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.value_long = make_head()
        self.value_short = make_head()
        self.value_flat = make_head()
    def forward(self, trunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        trunk: [batch, trunk_dim] (fusion.py's FusionTrunk output)
        Returns all three value estimates -- [batch] each -- since the PPO
        update needs all three (whichever one was actually used to bootstrap
        this transition's GAE, per select() below, still needs its
        counterparts' gradients to flow for the heads that WEREN'T selected
        this step but will be on some other step).
        """
        return {
            "long": self.value_long(trunk).squeeze(-1),
            "short": self.value_short(trunk).squeeze(-1),
            "flat": self.value_flat(trunk).squeeze(-1),
        }
    @staticmethod
    def select(values: Dict[str, torch.Tensor], position_direction: torch.Tensor) -> torch.Tensor:
        """
        Picks whichever critic head matches the regime the agent is
        ACTUALLY exposed to right now (not which action it just took --
        the position direction it's carrying INTO this valuation), for use
        as the bootstrap value in GAE.
        position_direction: [batch], sign of the currently held position
            (e.g. torch.sign(portfolio.positions[:, 0])), in {-1, 0, 1}.
        """
        is_long = position_direction > 0
        is_short = position_direction < 0
        return torch.where(is_long, values["long"], torch.where(is_short, values["short"], values["flat"]))