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

Tensor = torch.Tensor


@dataclass
class SimulatedFill:
    """Output of ExecutionSimulator.simulate_fill(). All fields [n_envs]."""
    filled_qty: Tensor      # signed shares actually filled (0 if no fill / rejected)
    fill_price: Tensor      # tick-snapped executed price
    is_partial: Tensor      # bool: True where requested size > filled size
    slippage_bps: Tensor    # signed slippage vs. mid, in bps, for logging/diagnostics


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
    ) -> None:
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
            device: torch device string (e.g. "cpu", "cuda"). Defaults to CUDA if available.
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

    def snap_to_tick(self, price: Tensor) -> Tensor:
        """Round price to the nearest valid tick increment."""
        return torch.round(price / self.tick_size) * self.tick_size

    def snap_to_tick_adverse(self, price: Tensor, direction: Tensor) -> Tensor:
        """
        Snap to the tick grid AGAINST the trader: buys round up, sells round
        down. Nearest-tick rounding hands back up to half a tick at random,
        which on a cheap stock is worth far more than the entire modelled
        cost (half a tick is 5.2 bps on a $9.66 name vs a ~0.55 bps true
        cost), so `torch.round` silently made execution free -- and randomly
        free, since whether it rounded for or against you depended on where
        mid happened to sit on the grid. You never get price improvement
        from the exchange's rounding; this direction is the honest one.
        """
        units = price / self.tick_size
        snapped = torch.where(direction > 0, torch.ceil(units), torch.floor(units))
        # direction == 0 never reaches a real fill (see simulate_fill's
        # no_order mask); keep nearest there so the value stays sane.
        snapped = torch.where(direction == 0, torch.round(units), snapped)
        return snapped * self.tick_size

    # ------------------------------------------------------------------
    # simulate_fill and its private helpers
    # ------------------------------------------------------------------

    def simulate_fill(
        self,
        direction: Tensor,      # [n_envs], values in {-1, 0, 1}
        size: Tensor,           # [n_envs], requested shares (>= 0, unsigned magnitude)
        limit_offset: Tensor,   # [n_envs], offset from mid in ticks; agent's continuous limit control
        mid_price: Tensor,      # [n_envs], current bar mid/reference price
        bar_liquidity_proxy: Tensor,  # [n_envs], e.g. bar volume, used to bound fill size
        overtrading_factor: Optional[Tensor] = None,  # [n_envs], in [0, 1], see __init__ docstring
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

        CONTRACT: direction must contain only exact values in {-1, 0, 1}.
        This is enforced here (raises ValueError on violation) rather than
        left as a documented-but-unchecked assumption -- env_digonastics.py
        previously only WARNed that a fractional direction (e.g. 0.7 from
        an un-rounded policy output) gets silently treated as a continuous
        scaling factor instead of raising. hybrid_policy.py's discrete head
        can only ever emit exact {-1,0,1} via IDX_TO_DIRECTION, so this
        should never trip in normal use -- it exists to catch a
        policy/action-postprocessing bug loudly instead of letting it train
        (or trade) on a silently wrong action semantics.
        """
        direction = self._validate_direction(direction)
        size = size.to(self.device).float().clamp(min=0.0)
        limit_offset = limit_offset.to(self.device).float()
        mid_price = mid_price.to(self.device).float()

        # ZERO-LIQUIDITY FIX. The clamp below exists so the participation
        # DIVISION (filled/proxy, in _compute_fill_price) can't divide by
        # zero -- but applying it before the fill-size cap silently turned
        # "this bar did not trade" into "this bar can absorb 1e-7 shares":
        #     max_fillable = max_participation * clamp(0.0, min=1e-6)
        #                  = 0.1 * 1e-6 = 1e-07 shares
        # So a halted/missing bar produced a NON-ZERO fill of exactly 1e-07
        # shares, which is a real fill as far as portfolio_state.py is
        # concerned: it books a position change and gets charged full
        # commission AND the flat platform ticket fee.
        #
        # Measured on a real 50-rollout run: every single sub-$1 fill in the
        # log was exactly 1.000e-07 shares -- notionals like $0.0000153 --
        # and they were ~54% of all fills in the sampled window. At $1.00 a
        # ticket that is pure, structural waste, and it bypassed
        # risk_manager.py's min_order_notional gate entirely because that
        # gate (correctly) checks the REQUESTED size, while this dust is
        # manufactured downstream of it by the participation cap.
        #
        # vec_trading_env.load_aligned_volumes()'s docstring already claimed
        # this behavior ("Zero volume correctly makes max_participation * 0
        # = 0 fillable shares") -- the clamp is what made that false. Zero
        # liquidity now means NO fill, and the clamp is kept purely for the
        # division safety it was actually there for.
        raw_liquidity = bar_liquidity_proxy.to(self.device).float()
        no_liquidity = raw_liquidity <= 0
        bar_liquidity_proxy = raw_liquidity.clamp(min=1e-6)
        overtrading_factor = self._prepare_overtrading_factor(overtrading_factor, mid_price)

        no_order = (direction == 0) | (size == 0) | no_liquidity

        # RAW liquidity here, not the clamped copy: the fill cap must scale
        # with the volume that actually traded. Passing the clamped value
        # imposed a hard floor of max_participation * 1e-6 = 1e-7 shares on
        # EVERY bar regardless of real volume, which is the dust bug above.
        # The clamped copy is still used for _compute_fill_price's
        # participation division, which is the only thing it was for.
        filled_size, is_partial = self._apply_participation_cap(size, raw_liquidity, no_order)

        fill_price = self._compute_fill_price(
            direction=direction,
            mid_price=mid_price,
            limit_offset=limit_offset,
            filled_size=filled_size,
            bar_liquidity_proxy=bar_liquidity_proxy,
            overtrading_factor=overtrading_factor,
        )
        fill_price = torch.where(no_order, mid_price, fill_price)

        filled_qty = torch.where(no_order, torch.zeros_like(filled_size), direction * filled_size)

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

    def _validate_direction(self, direction: Tensor) -> Tensor:
        """Cast direction to this device/dtype and enforce the {-1, 0, 1} contract."""
        direction = direction.to(self.device).float()
        valid = torch.isclose(direction, torch.round(direction)) & (direction.abs() <= 1.0 + 1e-6)
        if not bool(valid.all()):
            bad_values = direction[~valid].tolist()
            raise ValueError(
                f"ExecutionSimulator.simulate_fill(): direction must be exactly in {{-1, 0, 1}}, "
                f"got out-of-contract value(s) {bad_values}. Discretize the policy's direction head "
                f"(e.g. torch.sign() after rounding, or an argmax/Categorical index mapped through "
                f"IDX_TO_DIRECTION) before calling this."
            )
        return direction

    def _prepare_overtrading_factor(self, overtrading_factor: Optional[Tensor], mid_price: Tensor) -> Tensor:
        """Default to zero (no surcharge) when not supplied; otherwise cast and clip to [0, 1]."""
        if overtrading_factor is None:
            return torch.zeros_like(mid_price)
        return overtrading_factor.to(self.device).float().clamp(0.0, 1.0)

    def _apply_participation_cap(
        self, size: Tensor, bar_liquidity_proxy: Tensor, no_order: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Cap requested size at `max_participation` of bar liquidity; flag the remainder as partial."""
        max_fillable = self.max_participation * bar_liquidity_proxy
        filled_size = torch.minimum(size, max_fillable)
        is_partial = (filled_size < size) & ~no_order
        return filled_size, is_partial

    def _compute_fill_price(
        self,
        direction: Tensor,
        mid_price: Tensor,
        limit_offset: Tensor,
        filled_size: Tensor,
        bar_liquidity_proxy: Tensor,
        overtrading_factor: Tensor,
    ) -> Tensor:
        """
        effective_price = mid + side*(half_spread + sqrt-impact + overtrade_surcharge) - side*limit_offset_price

        A favorable limit_offset (requesting a better price) reduces the
        effective adverse move; ticks are converted to price via tick_size.
        The result is tick-snapped and floored at one tick to avoid a
        non-positive price.
        """
        participation = (filled_size / bar_liquidity_proxy).clamp(0.0, 1.0)
        # MINIMUM-TICK SPREAD. A proportional spread alone is unphysical at
        # low prices: spread_bps=1.0 implies a half-spread of $0.00048 on a
        # $9.66 stock, i.e. quoting INSIDE the $0.01 tick grid, which no
        # venue permits. The true minimum quotable spread is one tick, so the
        # half-spread floor is half a tick -- 5.2 bps on a $9.66 name, which
        # is what actually makes cheap stocks expensive to trade rather than
        # free. Without this floor the modelled cost fell below the rounding
        # granularity and vanished, reproducing the same 1/price subsidy as
        # the limit_offset bug in _compute_fill_price().
        half_spread_cost = torch.clamp(
            (self.spread_bps / 1e4) * mid_price, min=self.tick_size / 2.0
        )
        impact_cost = self.impact_coef * torch.sqrt(participation) * mid_price
        overtrade_cost = (self.overtrade_surcharge_bps / 1e4) * overtrading_factor * mid_price

        # FREE-MONEY FIX. limit_offset used to be subtracted from the adverse
        # move with NO CAP and no fill rejection, so a positive offset was an
        # unconditional price improvement: buys filled below mid and sells
        # above it, every time, for free. Because the improvement is a fixed
        # DOLLARS-PER-SHARE amount (offset_ticks * tick_size) its value in bps
        # scales as 1/price -- $0.20 is 3.6 bps on a $557 stock and 207 bps on
        # a $9.66 one.
        #
        # A 151-rollout run found and maximised exactly this: net worth
        # +486%, but with a per-bar directional hit rate of 41.6% (WORSE than
        # a coin flip, t-stat 0.17 on signed return) and 93/100 streams
        # ending BELOW their starting capital. The entire gain came from
        # seven of the cheapest tickers, with rank correlation between ticker
        # price and final equity of -0.921. It was harvesting the subsidy,
        # not trading. Mirroring made it worse by synthesising very low price
        # paths that don't correspond to any real quote.
        #
        # The cap below is the economically defensible bound: resting a
        # passive order can at best save you the adverse costs you would have
        # paid by crossing, so the best achievable fill is MID, never better.
        # Anything beyond that is inventing money that no counterparty paid.
        #
        # STILL OPTIMISTIC, and deliberately flagged rather than silently
        # accepted: a real passive order may not fill at all. Modelling that
        # properly needs the bar's high/low (both are in the parquet, only
        # `close` is loaded today) to fill only when the bar actually trades
        # through the limit price. Until then a passive order here always
        # fills at mid, which still favours the agent.
        adverse_cost = half_spread_cost + impact_cost + overtrade_cost
        limit_offset_price = (limit_offset * self.tick_size).clamp(max=adverse_cost)
        raw_price = (
            mid_price
            + direction * adverse_cost
            - direction * limit_offset_price
        )

        # DELIBERATELY NOT SNAPPED. `mid_price` here is a close price that
        # already sits ON the tick grid, so adding a half-tick spread and then
        # rounding away from the trader charged a FULL tick -- double the true
        # half-spread (20.7 bps round-trip on a $9.66 name where 10.4 is
        # correct). Snapping to nearest instead reintroduces the opposite
        # error, silently refunding up to half a tick. Since the cost model
        # above already prices the spread analytically, any snapping here is
        # double-counting in one direction or the other; leaving the price
        # unsnapped is the only unbiased choice. snap_to_tick/
        # snap_to_tick_adverse remain for callers that need a grid-valid quote.
        return torch.clamp(raw_price, min=self.tick_size)

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    def compute_commission(self, filled_qty: Tensor, fill_price: Tensor) -> Tensor:
        """Commission for a fill batch: flat per-share + bps of notional, with a floor. [n_envs]."""
        shares = filled_qty.abs()
        notional = shares * fill_price
        commission = shares * self.commission_per_share + notional * (self.commission_bps / 1e4)
        return torch.where(
            shares > 0,
            torch.clamp(commission, min=self.min_commission),
            torch.zeros_like(commission),
        )

    def compute_platform_fee(self, filled_qty: Tensor) -> Tensor:
        """
        Flat per-trade platform/broker ticket fee, charged whenever a non-zero
        fill occurred this step, independent of size or notional. [n_envs].
        """
        shares = filled_qty.abs()
        return torch.where(
            shares > 0,
            torch.full_like(shares, self.platform_fee_per_trade),
            torch.zeros_like(shares),
        )