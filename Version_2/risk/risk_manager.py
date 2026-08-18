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
from typing import Optional, Tuple

import torch

from env.portfolio_state import PortfolioState

Tensor = torch.Tensor


@dataclass
class RiskLimits:
    max_position_frac: float = 0.25              # max |position notional| / equity, per ticker
    max_gross_exposure_frac: float = 1.0          # max sum(|position notional|) / equity, across all tickers sharing a portfolio
    max_ticker_concentration_frac: float = 0.35   # max any single ticker's share of *gross exposure* (not equity)
    max_order_notional_frac: float = 0.10         # max single order's notional / equity, regardless of resulting position
    drawdown_halt_frac: float = 0.20              # drawdown from peak equity at which new-exposure orders are blocked (reduce-only)
    min_order_notional: float = 2_000.0           # ABSOLUTE $; drop exposure-increasing orders smaller than this. Kept in
                                                   # sync with training/config.py's RiskConfig.min_order_notional (which is
                                                   # what train.py actually passes) -- see that field for the sizing math. See
                                                   # RiskManager._drop_dust_orders() for why this exists and why it's
                                                   # absolute rather than a fraction of equity. 0.0 disables.


@dataclass
class RiskAdjustedAction:
    direction: Tensor     # [n_envs], possibly zeroed where size was clipped to 0
    size: Tensor          # [n_envs], clipped requested size (never increased)
    limit_offset: Tensor  # [n_envs], passed through unchanged
    reduce_only: Tensor   # [n_envs] bool, True where the drawdown halt is active this step
    breached: Tensor      # [n_envs] bool, True where any hard limit actually clipped this step's order
    drawdown: Tensor      # [n_envs], current drawdown fraction from peak equity


class RiskManager:
    """
    Every cap here behaves the same way: it only ever restricts orders that
    would INCREASE exposure. Reducing or closing an existing position is
    always allowed (including during a drawdown halt) -- a risk governor
    that could block you from de-risking would be actively dangerous.
    """

    def __init__(self, limits: RiskLimits, device: Optional[str] = None) -> None:
        self.limits = limits
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def apply(
        self,
        direction: Tensor,
        size: Tensor,
        limit_offset: Tensor,
        mid_price: Tensor,                     # [n_envs], this ticker's price
        portfolio: PortfolioState,
        ticker_idx: int = 0,
        all_prices: Optional[Tensor] = None,    # [n_envs, n_tickers], needed for gross-exposure/concentration when n_tickers > 1
    ) -> RiskAdjustedAction:
        """
        Runs the fixed cap sequence -- per-order notional, per-ticker
        position, portfolio gross exposure, per-ticker concentration,
        drawdown halt -- clipping `size` (never increasing it) and zeroing
        `direction` wherever the clipped size lands at 0. Every cap only
        restricts exposure-increasing orders; reducing/closing an existing
        position always passes through untouched.
        """
        direction = direction.to(self.device).float()
        size = size.to(self.device).float().clamp(min=0.0)
        limit_offset = limit_offset.to(self.device).float()
        mid_price = mid_price.to(self.device).float()
        all_prices = all_prices.to(self.device) if all_prices is not None else None

        equity = portfolio.equity(
            all_prices if all_prices is not None else mid_price.unsqueeze(1)
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

        size, breached = self._cap_order_notional(size, breached, equity, mid_price)
        size, breached = self._cap_ticker_position(
            size, breached, direction, current_notional_ticker, equity, mid_price
        )

        if all_prices is not None and portfolio.n_tickers > 1:
            size, breached = self._cap_gross_exposure_and_concentration(
                size, breached, increasing, current_notional_ticker, portfolio, all_prices, equity, mid_price
            )

        size, breached = self._apply_drawdown_halt(size, breached, reduce_only, increasing)

        # LAST, deliberately: every cap above can SHRINK an order, so an order
        # that was a sane size when the policy asked for it may be dust by the
        # time it gets here. Checking earlier would let a capped-down order
        # through.
        size, breached = self._drop_dust_orders(size, breached, increasing, mid_price)

        direction = torch.where(size == 0, torch.zeros_like(direction), direction)

        return RiskAdjustedAction(
            direction=direction,
            size=size,
            limit_offset=limit_offset,
            reduce_only=reduce_only,
            breached=breached,
            drawdown=drawdown,
        )

    def _cap_order_notional(
        self, size: Tensor, breached: Tensor, equity: Tensor, mid_price: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Per-order notional cap, independent of the resulting position (applies regardless of direction).

        NOTE: this one deliberately applies to REDUCING orders too -- a
        single-order size bound is a fat-finger guard, not an exposure
        governor. A close larger than the cap is split across cycles (still
        de-risks, just in steps); it is never BLOCKED, only paced.
        """
        max_order_notional = self.limits.max_order_notional_frac * equity
        max_order_qty = max_order_notional / mid_price.clamp(min=1e-6)
        breached = breached | (size > max_order_qty)
        size = torch.minimum(size, max_order_qty)
        return size, breached

    def _cap_ticker_position(
        self,
        size: Tensor,
        breached: Tensor,
        direction: Tensor,
        current_notional_ticker: Tensor,
        equity: Tensor,
        mid_price: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Per-ticker position cap: bounds the RESULTING |position notional|
        after the fill, for adds, opens, AND flips alike (FLIP FIX -- see
        kelly_sizing.py's _cap_growing_size for the same fix and rationale).

        Pure reduces pass through: their resulting position is smaller than
        the current one, so the cap (max + |pos|) never binds below a full
        close. Opposite-direction fills are capped at |pos| + max shares --
        a full close plus at most `max_position_frac` of equity in NEW
        reversed exposure.
        """
        max_position_notional = self.limits.max_position_frac * equity
        mid_price = mid_price.clamp(min=1e-6)
        pos_shares = current_notional_ticker / mid_price

        opposite = (torch.sign(direction) != torch.sign(pos_shares)) & (pos_shares != 0) & (direction != 0)
        max_shares = max_position_notional / mid_price
        max_qty = max_shares + torch.where(opposite, pos_shares.abs(), -pos_shares.abs())
        max_qty = max_qty.clamp(min=0.0)

        would_exceed = size > max_qty
        size = torch.minimum(size, max_qty)
        breached = breached | would_exceed
        return size, breached

    def _cap_gross_exposure_and_concentration(
        self,
        size: Tensor,
        breached: Tensor,
        increasing: Tensor,
        current_notional_ticker: Tensor,
        portfolio: PortfolioState,
        all_prices: Tensor,
        equity: Tensor,
        mid_price: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Portfolio-level gross exposure cap + per-ticker concentration-of-
        gross-exposure cap. Only meaningful (and only called) when the
        portfolio actually spans multiple tickers sharing one capital pool
        -- see module docstring.
        """
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

        return size, breached

    def _drop_dust_orders(
        self, size: Tensor, breached: Tensor, increasing: Tensor, mid_price: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Zeroes exposure-INCREASING orders whose notional is below
        `min_order_notional`. Reducing/closing orders always pass, no matter
        how small -- same contract as every other cap in this class, and a
        hard requirement here: a minimum that could block a close would trap
        a position the system is trying to exit.

        WHY THIS EXISTS (measured, from a real 50-rollout run):
        73.8% of that run's fills had a notional under $1.00, with a MEDIAN
        fill of $0.20 -- fractions of a share worth pennies -- while each one
        paid the flat $1.00 platform ticket fee (see paths.py's
        PLATFORM_FEE_PER_TRADE). That is a >500% round-trip cost on the
        median trade, and it accounted for 99.3% of the run's entire
        $134k loss; gross PnL excluding the ticket fee was only -$980.
        Filtering those fills at $100 discards ~86% of the ORDERS but keeps
        ~98.8% of the traded NOTIONAL -- i.e. what gets dropped is
        economically meaningless.

        The root cause is a policy-expressiveness gap, not a broker cost:
        model/hybrid_policy.py's discrete direction head fires SHORT/LONG
        independently of the continuous Beta size head, and the Beta cannot
        emit exactly 0 (open support on (0,1)). So a policy that has
        correctly learned "trading is expensive, trade less" expresses that
        as "LONG, 0.0014 shares" rather than as FLAT -- which still fills and
        still gets billed a full ticket.

        Absolute dollars, not a fraction of equity, because the cost this
        defends against (the flat per-trade fee) is itself absolute: the
        thing that makes an order uneconomic is its size relative to a fixed
        $/ticket, not relative to the account.

        Applies in training, backtest AND live -- unlike the training-only
        Kelly floor in kelly_sizing.py. Submitting a 20-cent order to a real
        broker is not something we want the live loop doing either, so this
        one deliberately does NOT carve out an exception.
        """
        if self.limits.min_order_notional <= 0:
            return size, breached
        order_notional = size * mid_price
        too_small = increasing & (size > 0) & (order_notional < self.limits.min_order_notional)
        size = torch.where(too_small, torch.zeros_like(size), size)
        breached = breached | too_small
        return size, breached

    def _apply_drawdown_halt(
        self, size: Tensor, breached: Tensor, reduce_only: Tensor, increasing: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Blocks any exposure-increasing order entirely while reduce_only is
        active. Tracked as a breach BEFORE zeroing so callers can
        distinguish "the policy asked for something we trimmed" from "the
        policy asked for something we trimmed AND we're in a drawdown halt"
        via reduce_only.
        """
        would_be_blocked = reduce_only & increasing & (size > 0)
        size = torch.where(reduce_only & increasing, torch.zeros_like(size), size)
        breached = breached | would_be_blocked
        return size, breached