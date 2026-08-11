"""
env/portfolio_state.py

Pure bookkeeping for a vectorized multi-asset portfolio. Tracks cash,
per-ticker positions, realized/unrealized PnL, and exposure across a batch
of parallel rollout streams (envs).

This module makes NO decisions — it just records the consequences of fills
that execution_sim.py produces. vec_trading_env.py is the orchestrator that
calls into both.

Shapes:
    n_envs    -> number of parallel rollout streams (batch dimension)
    n_tickers -> number of tradable assets per stream (14 in this project)

All state tensors are [n_envs, n_tickers] unless noted otherwise, and cash /
equity are [n_envs]. Everything lives on `device` (cpu or cuda) as float32
tensors so this composes cleanly with the PPO training loop without host<->
device syncs on every step.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Fill:
    """
    A single executed (possibly partial) fill for one ticker in one env.
    Produced by execution_sim.py, consumed by PortfolioState.step_apply().

    All fields are tensors of shape [n_envs] (one row per parallel stream).
    A qty of 0 means "no fill happened this step for this ticker in this env"
    and should still be passed through (it's a no-op for that row).
    """
    ticker_idx: int              # which of the n_tickers columns this fill batch applies to
    qty: torch.Tensor            # signed shares/contracts filled this step, [n_envs]. +buy / -sell
    price: torch.Tensor          # fill price actually executed at, [n_envs]
    commission: torch.Tensor     # $ cost charged for this fill, [n_envs], >= 0


class PortfolioState:
    """
    Vectorized portfolio ledger. One instance manages n_envs parallel
    portfolios simultaneously (this is what makes the trading env
    "vectorized" — a batch of independent backtests stepped in lockstep).
    """

    def __init__(
        self,
        n_envs: int,
        n_tickers: int,
        initial_cash: float = 100_000.0,
        device: Optional[str] = None,
    ):
        self.n_envs = n_envs
        self.n_tickers = n_tickers
        self.initial_cash = float(initial_cash)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reset()

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """
        Reset portfolio state. If env_mask (bool tensor [n_envs]) is given,
        only those rows are reset (useful for auto-resetting individual
        rollout streams that hit `done` without stalling the whole batch).
        A full reset happens on first construction or when env_mask is None.
        """
        if env_mask is None:
            self.cash = torch.full((self.n_envs,), self.initial_cash, device=self.device, dtype=torch.float32)
            self.positions = torch.zeros((self.n_envs, self.n_tickers), device=self.device, dtype=torch.float32)
            self.avg_entry_price = torch.zeros((self.n_envs, self.n_tickers), device=self.device, dtype=torch.float32)
            self.realized_pnl = torch.zeros((self.n_envs,), device=self.device, dtype=torch.float32)
            self.realized_pnl_ticker = torch.zeros((self.n_envs, self.n_tickers), device=self.device, dtype=torch.float32)
            self.total_commission_paid = torch.zeros((self.n_envs,), device=self.device, dtype=torch.float32)
            self.peak_equity = torch.full((self.n_envs,), self.initial_cash, device=self.device, dtype=torch.float32)
        else:
            env_mask = env_mask.to(self.device)
            self.cash[env_mask] = self.initial_cash
            self.positions[env_mask] = 0.0
            self.avg_entry_price[env_mask] = 0.0
            self.realized_pnl[env_mask] = 0.0
            self.realized_pnl_ticker[env_mask] = 0.0
            self.total_commission_paid[env_mask] = 0.0
            self.peak_equity[env_mask] = self.initial_cash

    def step_apply(self, fill: Fill) -> torch.Tensor:
        """
        Apply a batch of fills (one ticker, all envs) to the ledger.
        Handles same-direction adds (updates weighted avg_entry_price, no
        realized PnL) and opposite-direction reduces/closes/flips (realizes
        PnL on the closed portion, and reopens at the fill price if the fill
        overshoots the existing position size).

        Returns the per-env realized PnL delta produced by this fill
        (0 for envs that only added to a position), shape [n_envs].
        """
        i = fill.ticker_idx
        qty = fill.qty.to(self.device)
        price = fill.price.to(self.device)
        commission = fill.commission.to(self.device)

        pos = self.positions[:, i]
        entry = self.avg_entry_price[:, i]

        # Cash always moves by -(qty * price) - commission, regardless of direction
        self.cash -= qty * price + commission
        self.total_commission_paid += commission

        same_direction = (torch.sign(pos) == torch.sign(qty)) | (pos == 0)
        opposite_direction = ~same_direction & (qty != 0)

        # Amount of the existing position this fill closes (0 if same-direction/no fill)
        closing_qty = torch.where(
            opposite_direction,
            torch.minimum(qty.abs(), pos.abs()),
            torch.zeros_like(qty),
        )
        realized_delta = closing_qty * (price - entry) * torch.sign(pos)
        # only counts where we actually closed something opposite to an existing position
        realized_delta = torch.where(opposite_direction, realized_delta, torch.zeros_like(realized_delta))

        self.realized_pnl += realized_delta
        self.realized_pnl_ticker[:, i] += realized_delta

        # New position: same-direction adds simply accumulate; opposite-direction
        # fills subtract the closed amount and, if the fill overshoots the
        # existing position, the remainder opens a fresh position at `price`.
        overshoot = opposite_direction & (qty.abs() > pos.abs())
        new_pos = torch.where(
            same_direction,
            pos + qty,
            torch.where(
                overshoot,
                qty + pos,          # remainder after fully closing pos (signs cancel correctly)
                pos + qty,          # partial close: pos and qty have opposite signs, this shrinks |pos|
            ),
        )

        # New avg entry price:
        #   same-direction add -> weighted average of old and new
        #   opposite, partial/full close (no overshoot) -> unchanged (remaining shares keep original entry)
        #   opposite, overshoot (flip) -> reset to fill price for the new (reversed) position
        safe_denom = torch.where(new_pos == 0, torch.ones_like(new_pos), new_pos)
        weighted_entry = (pos * entry + qty * price) / safe_denom
        new_entry = torch.where(
            same_direction,
            weighted_entry,
            torch.where(overshoot, price, entry),
        )
        new_entry = torch.where(new_pos == 0, torch.zeros_like(new_entry), new_entry)

        self.positions[:, i] = new_pos
        self.avg_entry_price[:, i] = new_entry

        return realized_delta

    def unrealized_pnl(self, current_prices: torch.Tensor) -> torch.Tensor:
        """
        current_prices: [n_envs, n_tickers] mark prices.
        Returns per-env total unrealized PnL, shape [n_envs].
        """
        current_prices = current_prices.to(self.device)
        per_ticker = self.positions * (current_prices - self.avg_entry_price)
        return per_ticker.sum(dim=1)

    def equity(self, current_prices: torch.Tensor) -> torch.Tensor:
        """Total account value = cash + market value of open positions. [n_envs]."""
        current_prices = current_prices.to(self.device)
        market_value = (self.positions * current_prices).sum(dim=1)
        return self.cash + market_value

    def gross_exposure(self, current_prices: torch.Tensor) -> torch.Tensor:
        """Sum of |position value| across tickers, per env. [n_envs]."""
        current_prices = current_prices.to(self.device)
        return (self.positions.abs() * current_prices).sum(dim=1)

    def net_exposure(self, current_prices: torch.Tensor) -> torch.Tensor:
        """Sum of signed position value across tickers, per env. [n_envs]."""
        current_prices = current_prices.to(self.device)
        return (self.positions * current_prices).sum(dim=1)

    def update_drawdown_tracking(self, current_prices: torch.Tensor) -> torch.Tensor:
        """
        Call once per step. Updates the running peak equity and returns the
        current drawdown fraction per env (0 = at peak, 0.2 = 20% underwater).
        """
        eq = self.equity(current_prices)
        self.peak_equity = torch.maximum(self.peak_equity, eq)
        drawdown = (self.peak_equity - eq) / self.peak_equity.clamp(min=1e-8)
        return drawdown

    def reset_peak_equity(self, current_prices: torch.Tensor, env_mask: Optional[torch.Tensor] = None) -> None:
        """
        Re-baselines peak_equity to CURRENT equity, WITHOUT touching cash,
        positions, realized PnL, or commission history -- unlike reset(),
        this doesn't liquidate or forget anything real, it only changes what
        "underwater" is measured against going forward.

        Exists for the same reason kill_switch.py needed a training-only
        periodic reset: risk_manager.py's RiskLimits.drawdown_halt_frac
        compares current equity against this peak, and the peak here is
        otherwise a true all-time high that (correctly, for live trading)
        never resets on its own. During TRAINING, that peak gets set once
        early and then a struggling policy can sit in "underwater" territory
        for the rest of a very long episode (see paths.py/dataset.py -- an
        episode is one full pass through the training split, potentially
        tens of thousands of bars) -- RiskManager quietly forces reduce_only
        mode for that entire stretch, with no error and no KillSwitch flag,
        which looks exactly like "the policy stopped trading for no reason."
        Calling this periodically during training (see train.py's matching
        comment) prevents that without ever touching real portfolio state.

        NEVER call this in eval/backtest_report.py or live/live_loop.py --
        cheating the drawdown reference is a real training-only easement,
        not something a go/no-go backtest or live risk pipeline should do.
        """
        current_prices = current_prices.to(self.device)
        eq = self.equity(current_prices)
        if env_mask is None:
            self.peak_equity = eq.clone()
        else:
            env_mask = env_mask.to(self.device)
            self.peak_equity[env_mask] = eq[env_mask]