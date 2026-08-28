"""
env/execution_sim.py

Simulates order fills for backtest rollout. Given a hybrid action
(direction, size, limit_offset) and the current market state, produces a
realistic fill: price impacted by spread + order-size impact, snapped to
the instrument's tick size, with partial fills when the requested size
exceeds what the simulated available liquidity for that bar can absorb.

This module is stateless per-call (no bookkeeping) — portfolio_state.py
owns the ledger, vec_trading_env.py wires the two together.

VENUE CALIBRATION (P1). The cost model is calibrated to the venue this
project actually trades -- Alpaca, US equities -- rather than to a generic
retail broker:

    commission_bps           0.0   Alpaca is commission-free on US equities.
    platform_fee_per_trade   0.0   no ticket fee. The $1 ticket previously
                                   modelled here was 81.5% of ALL measured
                                   losses over a 305,966-trade run -- an
                                   IBKR-shaped cost this venue does not charge.
    half-spread              half a tick, PER TICKER. The minimum quotable US
                                   equity spread is one tick, so half a tick is
                                   the floor; expressed in bps that is
                                   automatically per-ticker, because it is a
                                   fixed $0.005 against a per-ticker price.
    impact                   square-root law against DAILY ADV, not bar volume.
                                   See _compute_fill_price().

The overtrading surcharge that used to live here has been REMOVED, not
zeroed. It was never a venue cost -- no exchange charges you more per share
for having traded recently -- it was a shaping term that wanted the policy to
churn less. Priced as slippage it contaminated every cost measurement (there
was no real number cost_per_turnover could be compared against), so it now
lives in training/config.py's RewardConfig.overtrade_penalty_coef and is
applied in the reward, where a shaping term belongs.
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
                           + side * impact_coef * daily_sigma * sqrt(Q / ADV) * mid

    where Q is the filled share count and ADV is the ticker's average DAILY
    volume in shares. This is the standard square-root market-impact law, and
    the denominator is the part that matters: impact is a property of how much
    of a name's daily liquidity an order consumes, not of which five-minute
    bucket it happened to land in. Measured against BAR volume the same order
    looks ~78x more aggressive (78 RTH bars in a session), and because
    sqrt(x) >> x for small x that error is largest at exactly the order sizes
    this project trades.

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
        spread_bps: float = 0.0,
        impact_coef: float = 0.5,
        max_participation: float = 0.1,
        commission_per_share: float = 0.0,
        commission_bps: float = 0.0,
        min_commission: float = 0.0,
        platform_fee_per_trade: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            tick_size: minimum price increment (0.01 for US equities). Accepts a float or a
                per-ticker [n_envs] tensor. It sets the half-tick spread floor below, which
                is what makes the modelled spread per-ticker once expressed in bps.
            spread_bps: PROPORTIONAL half-spread in bps of mid, charged to the taker ON TOP
                OF the half-tick floor. Default 0.0: on liquid US large caps the inside
                market is one tick wide, so the tick floor alone is the honest model, and
                any positive value here is an extra assumption you should be able to defend
                per ticker.
            impact_coef: the dimensionless Y prefactor of the square-root impact law
                impact/price = Y * sigma_daily * sqrt(Q/ADV). The literature puts Y in the
                0.5-1.0 range; it is NOT a free knob to be tuned until a strategy looks
                good. sigma_daily and ADV are supplied per ticker through
                set_liquidity_calibration().
            max_participation: cap on filled_size / BAR volume before the fill goes partial.
                This one stays on bar volume deliberately -- it is a fill-FEASIBILITY
                constraint (you cannot buy shares that never traded in that bar), not a cost.
            commission_per_share: flat $ per share commission
            commission_bps: additional commission as bps of notional
            min_commission: minimum $ commission charged on any non-zero fill
            platform_fee_per_trade: flat $ ticket fee charged on any non-zero fill,
                independent of size. 0.0 for Alpaca; 1.0 models an IBKR-shaped venue.
            device: torch device string (e.g. "cpu", "cuda"). Defaults to CUDA if available.
        """
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tick_size = (
            tick_size.to(self.device).float() if isinstance(tick_size, torch.Tensor)
            else float(tick_size)
        )
        self.spread_bps = spread_bps
        self.impact_coef = impact_coef
        self.max_participation = max_participation
        self.commission_per_share = commission_per_share
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        self.platform_fee_per_trade = platform_fee_per_trade

        # Per-ticker impact calibration, installed by set_liquidity_calibration().
        # Left unset, the impact term falls back to the pre-P1 bar-volume
        # denominator -- wrong by a factor of ~sqrt(78) -- so the fallback WARNS
        # once rather than silently mispricing every fill.
        self.adv_shares: Optional[Tensor] = None
        self.daily_sigma: Optional[Tensor] = None
        self._warned_no_adv = False

    def set_liquidity_calibration(self, adv_shares: Tensor, daily_sigma: Tensor) -> None:
        """
        Installs the per-ticker constants the square-root impact law needs.

        Args:
            adv_shares: [n_envs] average DAILY volume in shares, per ticker.
            daily_sigma: [n_envs] daily return standard deviation, per ticker, as a
                fraction (e.g. 0.018 for 1.8%/day).

        Both are properties of the instrument over the split being traded rather than
        of any one episode, so vec_trading_env.py measures them once at construction
        from the same aligned arrays it marks against.
        """
        self.adv_shares = adv_shares.to(self.device).float().clamp(min=1.0)
        self.daily_sigma = daily_sigma.to(self.device).float().clamp(min=1e-6)

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
        mid_price: Tensor,      # [n_envs], execution reference price for the fill bar
        bar_liquidity_proxy: Tensor,  # [n_envs], bar volume, bounds fill SIZE only
    ) -> SimulatedFill:
        """
        Simulates one step's fill for one ticker across all envs.

        `limit_offset` lets the agent request a price better than mid (less
        aggressive, may reduce realized slippage but this simple model does
        not currently reject fills for being outside the limit — see note
        below) — it's treated as a further price adjustment layered on top
        of the spread+impact slippage, in the agent's favor.

        `mid_price` is the price the fill is referenced to. Since P1 that is
        the EXECUTION price of the bar AFTER the one the observation ends on
        (bar t+1's open, or its VWAP) rather than bar t's close -- see
        vec_trading_env.load_aligned_price_frames(). Nothing in this file
        depends on which of the two it is; it is stated here because
        `slippage_bps` is measured against it and is therefore slippage
        against the arrival price, not against a close the agent could not
        have traded on.

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

    def _apply_participation_cap(
        self, size: Tensor, bar_liquidity_proxy: Tensor, no_order: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Cap requested size at `max_participation` of bar liquidity; flag the remainder as partial."""
        max_fillable = self.max_participation * bar_liquidity_proxy
        filled_size = torch.minimum(size, max_fillable)
        is_partial = (filled_size < size) & ~no_order
        return filled_size, is_partial

    def _impact_cost(self, filled_size: Tensor, mid_price: Tensor, bar_liquidity_proxy: Tensor) -> Tensor:
        """
        Square-root market impact, priced against DAILY ADV.

            impact / price = Y * sigma_daily * sqrt(Q / ADV)

        Q/ADV is the fraction of a normal day's liquidity the order consumes,
        which is the quantity the square-root law is stated in and fitted on.
        The previous version divided by the 5-minute BAR's volume, which
        overstates participation by roughly the number of bars in a session
        (78) and therefore overstates impact by ~sqrt(78) ~ 8.8x. At the order
        sizes this project trades that was the difference between "moves the
        market not at all", which is the truth for a four-figure order in a
        large cap, and a charge several times the half-spread.

        The fill-SIZE cap in _apply_participation_cap() still runs off bar
        volume, and correctly so: how much you can buy in one bar is a
        feasibility question, how much you move the price is a cost question,
        and they do not share a denominator.

        Falls back to the bar-volume denominator (with a one-time warning)
        when set_liquidity_calibration() was never called, e.g. in the unit
        harnesses in env_digonastics.py that construct a bare simulator.
        """
        if self.adv_shares is not None and self.adv_shares.shape != filled_size.shape:
            # SILENT BROADCAST GUARD. adv_shares/daily_sigma are per-TICKER
            # vectors of width n_envs, so a call with a different batch width
            # does not fail -- it BROADCASTS, pairing every order with every
            # ticker's liquidity and returning an [n_envs] fill price for a
            # 1-element order. Downstream that surfaces as "a Tensor with N
            # elements cannot be converted to Scalar" somewhere unrelated, or
            # worse, as a silently wrong impact charge where the widths happen
            # to be compatible. Fail here, where the cause is still legible.
            raise ValueError(
                f"ExecutionSimulator: liquidity calibration is for "
                f"{self.adv_shares.numel()} ticker(s) but simulate_fill() was called with "
                f"{filled_size.numel()}. Call set_liquidity_calibration() with vectors matching "
                f"this batch, or construct a separate simulator for a different-width harness."
            )
        if self.adv_shares is None or self.daily_sigma is None:
            if not self._warned_no_adv:
                print(
                    "[execution_sim] WARNING: no ADV calibration installed -- falling back to "
                    "the bar-volume impact denominator, which overstates impact by ~sqrt(bars "
                    "per session). Call set_liquidity_calibration() for the calibrated model."
                )
                self._warned_no_adv = True
            participation = (filled_size / bar_liquidity_proxy).clamp(0.0, 1.0)
            return self.impact_coef * torch.sqrt(participation) * mid_price

        adv_participation = (filled_size / self.adv_shares).clamp(0.0, 1.0)
        return self.impact_coef * self.daily_sigma * torch.sqrt(adv_participation) * mid_price

    def _compute_fill_price(
        self,
        direction: Tensor,
        mid_price: Tensor,
        limit_offset: Tensor,
        filled_size: Tensor,
        bar_liquidity_proxy: Tensor,
    ) -> Tensor:
        """
        effective_price = mid + side*(half_spread + sqrt-impact) - side*limit_offset_price

        A favorable limit_offset (requesting a better price) reduces the
        effective adverse move; ticks are converted to price via tick_size.
        The result is tick-snapped and floored at one tick to avoid a
        non-positive price.
        """
        # PER-TICKER HALF-TICK SPREAD. A proportional spread alone is
        # unphysical at low prices: spread_bps=1.0 implies a half-spread of
        # $0.00048 on a $9.66 stock, i.e. quoting INSIDE the $0.01 tick grid,
        # which no venue permits. The minimum quotable US equity spread is one
        # tick, so the half-spread is half a tick -- and because that is a
        # fixed $0.005 against a per-ticker price, the resulting cost in bps is
        # per-ticker by construction: 5.2 bps on a $9.66 name against 0.09 bps
        # on a $557 one. That, not a flat bps number, is what actually makes
        # cheap stocks expensive to trade.
        #
        # With spread_bps at its P1 default of 0.0 this IS the spread model.
        # A positive spread_bps layers a proportional component on top, for a
        # ticker whose inside market is genuinely wider than one tick.
        # torch.as_tensor: tick_size may be a per-ticker [n_envs] tensor.
        half_tick = torch.as_tensor(
            self.tick_size, device=mid_price.device, dtype=mid_price.dtype
        ) / 2.0
        half_spread_cost = torch.maximum(
            (self.spread_bps / 1e4) * mid_price, half_tick.expand_as(mid_price)
        )
        impact_cost = self._impact_cost(filled_size, mid_price, bar_liquidity_proxy)

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
        # properly means filling only when the bar actually trades through the
        # limit price. As of P1 the inputs for that exist -- vec_trading_env's
        # load_aligned_price_frames() loads the fill bar's high and low and the
        # env carries them as `bar_high`/`bar_low` -- but the fill rule itself
        # is NOT implemented here, so a passive order still always fills at
        # mid. That remains in the agent's favour, and it is now a wiring job
        # rather than a data-availability one.
        adverse_cost = half_spread_cost + impact_cost
        limit_offset_price = torch.minimum(
            limit_offset * torch.as_tensor(
                self.tick_size, device=mid_price.device, dtype=mid_price.dtype
            ),
            adverse_cost,
        )
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
        min_price = torch.as_tensor(
            self.tick_size, device=mid_price.device, dtype=mid_price.dtype
        ).expand_as(raw_price)
        return torch.maximum(raw_price, min_price)

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