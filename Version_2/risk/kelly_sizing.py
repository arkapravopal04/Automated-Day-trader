"""
risk/kelly_sizing.py

Fractional-Kelly position-size governor. NOT a training target -- this sits
between the policy's continuous size output and risk_manager.py / the
execution sim, and deterministically shrinks (never grows) the requested
order size so the implied max position notional never exceeds a fraction of
a rolling, per-stream Kelly-optimal fraction of equity.

Kelly fraction (single-outcome approximation):
    f* = W - (1 - W) / R
where:
    W = rolling win rate  (fraction of realized/closed trades with pnl > 0)
    R = payoff ratio      (avg winning-trade pnl / avg |losing-trade pnl|)

f* is clamped to [0, kelly_cap] and then scaled by `kelly_multiplier`
(e.g. 0.25 for quarter-Kelly) to get the max fraction of current equity any
single stream's position notional is allowed to reach.

This module has no learned parameters and does not touch the environment's
reward. Feed it realized trade PnLs -- the `realized_delta` return value of
PortfolioState.step_apply() -- as they occur; it maintains its own rolling
per-env trade-outcome history and ignores steps where no trade closed
(realized_delta == 0), since it estimates edge from closed-trade outcomes,
not tick-by-tick mark-to-market.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class KellySizingResult:
    size: torch.Tensor            # [n_envs], adjusted (never increased) requested size
    kelly_fraction: torch.Tensor  # [n_envs], the fractional-Kelly cap actually applied (post-multiplier)
    win_rate: torch.Tensor        # [n_envs], rolling win rate estimate
    payoff_ratio: torch.Tensor    # [n_envs], rolling payoff ratio estimate
    is_warm: torch.Tensor         # [n_envs] bool, whether enough trade history exists to trust the estimate


class KellySizer:
    def __init__(
        self,
        n_envs: int,
        lookback_trades: int = 30,
        min_trades_for_estimate: int = 10,
        kelly_multiplier: float = 0.25,     # quarter-Kelly
        kelly_cap: float = 1.0,             # never let raw f* exceed this before the multiplier
        default_fraction: float = 0.02,     # conservative fallback while warming up / no history
        device: Optional[str] = None,
    ):
        self.n_envs = n_envs
        self.lookback_trades = lookback_trades
        self.min_trades_for_estimate = min_trades_for_estimate
        self.kelly_multiplier = kelly_multiplier
        self.kelly_cap = kelly_cap
        self.default_fraction = default_fraction
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Rolling per-env closed-trade PnL history, circular buffer. NaN
        # marks an empty slot (not 0.0 -- a real closed trade can legitimately
        # net to ~0 and shouldn't masquerade as "no data yet").
        self._pnl_hist = torch.full(
            (self.n_envs, self.lookback_trades), float("nan"), device=self.device
        )
        self._ptr = [0] * self.n_envs
        self._count = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        if env_mask is None:
            self._pnl_hist.fill_(float("nan"))
            self._ptr = [0] * self.n_envs
            self._count.zero_()
        else:
            idx = torch.nonzero(env_mask.to(self.device), as_tuple=True)[0].tolist()
            for i in idx:
                self._pnl_hist[i] = float("nan")
                self._ptr[i] = 0
            self._count[env_mask.to(self.device)] = 0

    def record_realized_pnl(self, realized_delta: torch.Tensor) -> None:
        """
        Call once per env-step with PortfolioState.step_apply()'s return
        value. Rows where realized_delta == 0 (nothing closed this step) are
        ignored.
        """
        realized_delta = realized_delta.to(self.device)
        closed = realized_delta != 0
        if not closed.any():
            return
        for i in torch.nonzero(closed, as_tuple=True)[0].tolist():
            slot = self._ptr[i] % self.lookback_trades
            self._pnl_hist[i, slot] = realized_delta[i]
            self._ptr[i] += 1
            self._count[i] = min(self._count[i].item() + 1, self.lookback_trades)

    def _edge_estimate(self):
        valid = ~torch.isnan(self._pnl_hist)                       # [n_envs, lookback]
        n_valid = valid.sum(dim=1).float().clamp(min=1.0)

        win_mask = valid & (self._pnl_hist > 0)
        loss_mask = valid & (self._pnl_hist < 0)
        n_wins = win_mask.sum(dim=1).float()
        n_losses = loss_mask.sum(dim=1).float()

        win_rate = n_wins / n_valid
        avg_win = torch.where(win_mask, self._pnl_hist, torch.zeros_like(self._pnl_hist)).sum(dim=1) / n_wins.clamp(min=1.0)
        avg_loss = torch.where(loss_mask, self._pnl_hist, torch.zeros_like(self._pnl_hist)).abs().sum(dim=1) / n_losses.clamp(min=1.0)
        payoff_ratio = avg_win / avg_loss.clamp(min=1e-8)

        is_warm = self._count >= self.min_trades_for_estimate
        have_losses = n_losses > 0

        raw_kelly = win_rate - (1.0 - win_rate) / payoff_ratio.clamp(min=1e-8)
        # No losses observed yet -> R is undefined; don't let a lucky streak
        # imply an unbounded Kelly fraction, just cap it at kelly_cap.
        raw_kelly = torch.where(have_losses, raw_kelly, torch.full_like(raw_kelly, self.kelly_cap))
        raw_kelly = raw_kelly.clamp(min=0.0, max=self.kelly_cap)

        return win_rate, payoff_ratio, raw_kelly, is_warm

    def apply(
        self,
        size: torch.Tensor,                          # [n_envs], requested shares (unsigned)
        direction: torch.Tensor,                      # [n_envs]
        mid_price: torch.Tensor,                      # [n_envs]
        equity: torch.Tensor,                         # [n_envs]
        current_position_notional: torch.Tensor,      # [n_envs], signed, this ticker's current notional
    ) -> KellySizingResult:
        size = size.to(self.device).float().clamp(min=0.0)
        direction = direction.to(self.device).float()
        mid_price = mid_price.to(self.device).float()
        equity = equity.to(self.device).float()
        current_position_notional = current_position_notional.to(self.device).float()

        win_rate, payoff_ratio, raw_kelly, is_warm = self._edge_estimate()

        fractional_kelly = torch.where(
            is_warm,
            raw_kelly * self.kelly_multiplier,
            torch.full_like(raw_kelly, self.default_fraction),
        )

        max_notional = fractional_kelly * equity.clamp(min=1e-6)
        increasing = (torch.sign(direction) == torch.sign(current_position_notional)) | (current_position_notional == 0)

        headroom = (max_notional - current_position_notional.abs()).clamp(min=0.0)
        max_qty = headroom / mid_price.clamp(min=1e-6)

        adjusted_size = torch.where(increasing, torch.minimum(size, max_qty), size)

        return KellySizingResult(
            size=adjusted_size,
            kelly_fraction=fractional_kelly,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            is_warm=is_warm,
        )