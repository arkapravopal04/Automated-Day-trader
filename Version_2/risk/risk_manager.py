"""
risk/risk_manager.py

Hard, deterministic post-processor sitting between the policy's raw action
and env.step(). Unlike kelly_sizing.py (a soft, edge-aware sizing governor),
this module enforces fixed position/exposure/drawdown limits that never
depend on any rolling statistical estimate -- just current portfolio state
and current prices. Nothing here is a training signal, and nothing here is
learned or adapts over time.

Recommended pipeline order:

    policy action -> KellySizer.apply()   (soft, edge-based size cap)
                  -> RiskManager.apply()  (hard position/exposure/drawdown caps)
                  -> KillSwitch.apply()   (hard halt -- see kill_switch.py)
                  -> env.step()

Scope note: vec_trading_env.py currently gives each stream its own
single-asset PortfolioState (n_tickers == 1 per stream). Gross-exposure and
per-ticker-concentration checks below only do meaningful work if you later
share one PortfolioState across multiple tickers within a stream (n_tickers
> 1) -- pass `all_prices` in that case. With the current one-ticker-per-
stream wiring they're harmless no-ops (gross exposure degenerates to the
single position cap).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from portfolio_state import PortfolioState


@dataclass
class RiskLimits:
    max_position_frac: float = 0.25              # max |position notional| / equity, per ticker
    max_gross_exposure_frac: float = 1.0          # max sum(|position notional|) / equity, across all tickers sharing a portfolio
    max_ticker_concentration_frac: float = 0.35   # max any single ticker's share of *gross exposure* (not equity)
    max_order_notional_frac: float = 0.10         # max single order's notional / equity, regardless of resulting position
    drawdown_halt_frac: float = 0.20              # drawdown from peak equity at which new-exposure orders are blocked (reduce-only)


@dataclass
class RiskAdjustedAction:
    direction: torch.Tensor     # [n_envs], possibly zeroed where size was clipped to 0
    size: torch.Tensor          # [n_envs], clipped requested size (never increased)
    limit_offset: torch.Tensor  # [n_envs], passed through unchanged
    reduce_only: torch.Tensor   # [n_envs] bool, True where the drawdown halt is active this step
    breached: torch.Tensor      # [n_envs] bool, True where any hard limit actually clipped this step's order
    drawdown: torch.Tensor      # [n_envs], current drawdown fraction from peak equity


class RiskManager:
    """
    Every cap here behaves the same way: it only ever restricts orders that
    would INCREASE exposure. Reducing or closing an existing position is
    always allowed (including during a drawdown halt) -- a risk governor
    that could block you from de-risking would be actively dangerous.
    """

    def __init__(self, limits: RiskLimits, device: Optional[str] = None):
        self.limits = limits
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def apply(
        self,
        direction: torch.Tensor,
        size: torch.Tensor,
        limit_offset: torch.Tensor,
        mid_price: torch.Tensor,                     # [n_envs], this ticker's price
        portfolio: PortfolioState,
        ticker_idx: int = 0,
        all_prices: Optional[torch.Tensor] = None,    # [n_envs, n_tickers], needed for gross-exposure/concentration when n_tickers > 1
    ) -> RiskAdjustedAction:
        direction = direction.to(self.device).float()
        size = size.to(self.device).float().clamp(min=0.0)
        limit_offset = limit_offset.to(self.device).float()
        mid_price = mid_price.to(self.device).float()

        equity = portfolio.equity(
            all_prices.to(self.device) if all_prices is not None else mid_price.unsqueeze(1)
        ).clamp(min=1e-6)

        breached = torch.zeros_like(size, dtype=torch.bool)

        # --- drawdown halt: read-only w.r.t. portfolio.peak_equity. The
        # env's own update_drawdown_tracking() call remains the single
        # source of truth for peak tracking; we only read it here.
        peak = portfolio.peak_equity.clamp(min=1e-6)
        drawdown = (peak - equity).clamp(min=0.0) / peak
        reduce_only = drawdown >= self.limits.drawdown_halt_frac

        current_notional_ticker = portfolio.positions[:, ticker_idx] * mid_price
        increasing = (torch.sign(direction) == torch.sign(current_notional_ticker)) | (current_notional_ticker == 0)

        # --- per-order notional cap (independent of resulting position)
        max_order_notional = self.limits.max_order_notional_frac * equity
        max_order_qty = max_order_notional / mid_price.clamp(min=1e-6)
        breached = breached | (size > max_order_qty)
        size = torch.minimum(size, max_order_qty)

        # --- per-ticker position cap
        max_position_notional = self.limits.max_position_frac * equity
        headroom = (max_position_notional - current_notional_ticker.abs()).clamp(min=0.0)
        max_qty_position_cap = headroom / mid_price.clamp(min=1e-6)
        would_exceed_position_cap = increasing & (size > max_qty_position_cap)
        size = torch.where(increasing, torch.minimum(size, max_qty_position_cap), size)
        breached = breached | would_exceed_position_cap

        # --- portfolio-level gross exposure + per-ticker concentration
        # (only bites when the portfolio actually spans multiple tickers
        # sharing one capital pool -- see module docstring)
        if all_prices is not None and portfolio.n_tickers > 1:
            all_prices = all_prices.to(self.device)
            gross_exposure = portfolio.gross_exposure(all_prices)

            max_gross = self.limits.max_gross_exposure_frac * equity
            gross_headroom = (max_gross - gross_exposure).clamp(min=0.0)
            max_qty_gross_cap = gross_headroom / mid_price.clamp(min=1e-6)
            would_exceed_gross = increasing & (size > max_qty_gross_cap)
            size = torch.where(increasing, torch.minimum(size, max_qty_gross_cap), size)
            breached = breached | would_exceed_gross

            target_ticker_notional = self.limits.max_ticker_concentration_frac * gross_exposure.clamp(min=1e-6)
            conc_headroom = (target_ticker_notional - current_notional_ticker.abs()).clamp(min=0.0)
            max_qty_conc_cap = conc_headroom / mid_price.clamp(min=1e-6)
            projected_ticker_notional = current_notional_ticker.abs() + size * mid_price
            over_concentrated = increasing & (gross_exposure > 0) & (
                projected_ticker_notional > target_ticker_notional
            )
            size = torch.where(over_concentrated, torch.minimum(size, max_qty_conc_cap), size)
            breached = breached | over_concentrated

        # --- drawdown halt: block any exposure-increasing order entirely.
        # Tracked as a breach BEFORE zeroing so callers can distinguish "the
        # policy asked for something we trimmed" from "the policy asked for
        # something we trimmed AND we're in a drawdown halt" via reduce_only.
        would_be_blocked_by_halt = reduce_only & increasing & (size > 0)
        size = torch.where(reduce_only & increasing, torch.zeros_like(size), size)
        breached = breached | would_be_blocked_by_halt

        direction = torch.where(size == 0, torch.zeros_like(direction), direction)

        return RiskAdjustedAction(
            direction=direction,
            size=size,
            limit_offset=limit_offset,
            reduce_only=reduce_only,
            breached=breached,
            drawdown=drawdown,
        )