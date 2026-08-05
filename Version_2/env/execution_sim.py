"""
env/execution_sim.py

Simulates order fills for backtest rollout. Given a hybrid action
(direction, size, limit_offset) and the current market state, produces a
realistic fill: price impacted by spread + order-size impact, snapped to
the instrument's tick size, with partial fills when the requested size
exceeds what the simulated available liquidity for that bar can absorb.

This module is stateless per-call (no bookkeeping) — portfolio_state.py
owns the ledger, vec_trading_env.py wires the two together.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimulatedFill:
    """Output of ExecutionSimulator.simulate_fill(). All fields [n_envs]."""
    filled_qty: torch.Tensor      # signed shares actually filled (0 if no fill / rejected)
    fill_price: torch.Tensor      # tick-snapped executed price
    is_partial: torch.Tensor      # bool: True where requested size > filled size
    slippage_bps: torch.Tensor    # signed slippage vs. mid, in bps, for logging/diagnostics


class ExecutionSimulator:
    """
    Vectorized fill simulator across n_envs parallel streams for a single
    ticker at a time (vec_trading_env.py calls this once per ticker per step,
    consistent with how PortfolioState.step_apply is also per-ticker).

    Slippage model:
        effective_price = mid
                           + side * half_spread
                           + side * impact_coef * sqrt(participation_rate) * mid

    where participation_rate = requested_size / bar_liquidity_proxy (typically
    bar volume), clipped to [0, 1]. sqrt-impact is a standard square-root
    market-impact approximation (impact grows sublinearly with size).

    Partial fills:
        If requested size implies a participation_rate above
        `max_participation`, only max_participation * bar_liquidity_proxy
        shares are filled this step; the rest is simply dropped (the agent
        doesn't get a resting order — this is a single-step simulator, so
        it will see the unfilled remainder as still-flat and can re-submit
        next step if it wants to keep chasing size).
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        spread_bps: float = 1.0,
        impact_coef: float = 0.1,
        max_participation: float = 0.1,
        commission_per_share: float = 0.0,
        commission_bps: float = 0.0,
        min_commission: float = 0.0,
        platform_fee_per_trade: float = 1.0,
        overtrade_surcharge_bps: float = 3.0,
        device: Optional[str] = None,
    ):
        """
        Args:
            tick_size: minimum price increment for this instrument (e.g. 0.01 for US equities)
            spread_bps: assumed half-spread in basis points of mid price, applied against the taker
            impact_coef: coefficient on the sqrt-participation market impact term
            max_participation: cap on requested_size / bar_volume before the fill is partial
            commission_per_share: flat $ per share commission
            commission_bps: additional commission as bps of notional
            min_commission: minimum $ commission charged on any non-zero fill
            platform_fee_per_trade: flat $ ticket fee charged by the broker/platform on any
                non-zero fill, independent of size (separate from commission_* above, which
                model exchange/broker commission proper). Set to 0.0 to disable.
            overtrade_surcharge_bps: extra adverse slippage (bps of mid) layered on top of the
                normal spread+impact cost, scaled by an externally-supplied per-env
                `overtrading_factor` in [0, 1] passed into simulate_fill(). This models the
                real cost of churning a position on fast (5-minute) bars: the more frequently
                a stream has been trading in its recent lookback window, the worse its
                effective execution gets. Set to 0.0 to disable.
        """
        self.tick_size = tick_size
        self.spread_bps = spread_bps
        self.impact_coef = impact_coef
        self.max_participation = max_participation
        self.commission_per_share = commission_per_share
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        self.platform_fee_per_trade = platform_fee_per_trade
        self.overtrade_surcharge_bps = overtrade_surcharge_bps
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def snap_to_tick(self, price: torch.Tensor) -> torch.Tensor:
        """Round price to the nearest valid tick increment."""
        return torch.round(price / self.tick_size) * self.tick_size

    def simulate_fill(
        self,
        direction: torch.Tensor,      # [n_envs], values in {-1, 0, 1}
        size: torch.Tensor,           # [n_envs], requested shares (>= 0, unsigned magnitude)
        limit_offset: torch.Tensor,   # [n_envs], offset from mid in ticks; agent's continuous limit control
        mid_price: torch.Tensor,      # [n_envs], current bar mid/reference price
        bar_liquidity_proxy: torch.Tensor,  # [n_envs], e.g. bar volume, used to bound fill size
        overtrading_factor: Optional[torch.Tensor] = None,  # [n_envs], in [0, 1], see __init__ docstring
    ) -> SimulatedFill:
        """
        Simulates one step's fill for one ticker across all envs.

        `limit_offset` lets the agent request a price better than mid (less
        aggressive, may reduce realized slippage but this simple model does
        not currently reject fills for being outside the limit — see note
        below) — it's treated as a further price adjustment layered on top
        of the spread+impact slippage, in the agent's favor.

        `overtrading_factor` (optional) scales an additional adverse slippage
        term (`overtrade_surcharge_bps`) meant to represent the real cost of
        trading too frequently on fast bars — the caller (vec_trading_env.py)
        is responsible for computing this from its own rolling per-env trade
        history and passing it in. If omitted, no surcharge is applied.

        Direction == 0 means "no order this step" -> filled_qty is 0.
        """
        direction = direction.to(self.device).float()
        size = size.to(self.device).float().clamp(min=0.0)
        limit_offset = limit_offset.to(self.device).float()
        mid_price = mid_price.to(self.device).float()
        bar_liquidity_proxy = bar_liquidity_proxy.to(self.device).float().clamp(min=1e-6)

        if overtrading_factor is None:
            overtrading_factor = torch.zeros_like(mid_price)
        else:
            overtrading_factor = overtrading_factor.to(self.device).float().clamp(0.0, 1.0)

        no_order = (direction == 0) | (size == 0)

        # --- Partial fill sizing based on participation cap
        max_fillable = self.max_participation * bar_liquidity_proxy
        filled_size = torch.minimum(size, max_fillable)
        is_partial = (filled_size < size) & ~no_order

        # --- Slippage: half-spread + sqrt-impact, both working against the taker
        participation = (filled_size / bar_liquidity_proxy).clamp(0.0, 1.0)
        half_spread_cost = (self.spread_bps / 1e4) * mid_price
        impact_cost = self.impact_coef * torch.sqrt(participation) * mid_price
        overtrade_cost = (self.overtrade_surcharge_bps / 1e4) * overtrading_factor * mid_price

        # Effective price = mid + side*(spread + impact + overtrade_surcharge) - side*limit_offset_in_price
        # (a favorable limit_offset, e.g. requesting a better price, reduces
        # the effective adverse move; ticks -> price via tick_size)
        limit_offset_price = limit_offset * self.tick_size
        raw_price = (
            mid_price
            + direction * (half_spread_cost + impact_cost + overtrade_cost)
            - direction * limit_offset_price
        )

        fill_price = self.snap_to_tick(raw_price)
        fill_price = torch.clamp(fill_price, min=self.tick_size)  # never let price go non-positive

        filled_qty = torch.where(no_order, torch.zeros_like(filled_size), direction * filled_size)
        fill_price = torch.where(no_order, mid_price, fill_price)

        slippage_bps = torch.where(
            no_order,
            torch.zeros_like(fill_price),
            (fill_price - mid_price) / mid_price.clamp(min=1e-8) * direction * 1e4,
        )

        return SimulatedFill(
            filled_qty=filled_qty,
            fill_price=fill_price,
            is_partial=is_partial,
            slippage_bps=slippage_bps,
        )

    def compute_commission(self, filled_qty: torch.Tensor, fill_price: torch.Tensor) -> torch.Tensor:
        """Commission for a fill batch: flat per-share + bps of notional, with a floor. [n_envs]."""
        shares = filled_qty.abs()
        notional = shares * fill_price
        commission = shares * self.commission_per_share + notional * (self.commission_bps / 1e4)
        commission = torch.where(shares > 0, torch.clamp(commission, min=self.min_commission), torch.zeros_like(commission))
        return commission

    def compute_platform_fee(self, filled_qty: torch.Tensor) -> torch.Tensor:
        """
        Flat per-trade platform/broker ticket fee, charged whenever a non-zero
        fill occurred this step, independent of size or notional. [n_envs].
        """
        shares = filled_qty.abs()
        fee = torch.where(
            shares > 0,
            torch.full_like(shares, self.platform_fee_per_trade),
            torch.zeros_like(shares),
        )
        return fee