"""
training/config.py

Central config for the PPO training run: hyperparameters, ticker universe,
window sizes, PPO clip range, learning rates, entropy coefficients, and the
env/risk knobs introduced in the earlier env/model/risk modules. Grouped
into nested dataclasses so each module (env, model, risk, reward, PPO, run)
can be constructed from its own slice instead of threading forty individual
kwargs through every constructor.

This is a plain Python dataclass tree, not a YAML/CLI-loaded config -- edit
it directly, or override via `dataclasses.replace(cfg.ppo, clip_range=0.1)`
etc. Add a file-based loader later if you actually need one; not building
that speculatively.

Single-source-of-truth note: MIRROR_PROB, DIVERSITY_WINDOW, DIVERSITY_COEF,
OVERTRADE_WINDOW, OVERTRADE_FREE_TRADES, OVERTRADE_PENALTY_COEF, and
PLATFORM_FEE_PER_TRADE all previously existed as BOTH paths.py's
.env-driven constants AND separately hardcoded EnvConfig field defaults --
two places that happened to agree today but had no mechanism keeping them
in sync. EnvConfig's defaults below now import and reference paths.py's
constants directly, so a .env override (e.g. TRADING_MIRROR_PROB=0.3) takes
effect for every EnvConfig built through this file, not just for a
VecTradingEnv built directly without going through TrainingConfig. The
friction-only fields (spread_bps, impact_coef, max_participation,
commission_*) have no paths.py equivalent -- those are intentionally
Python-only, toggled via EnvConfig.for_friction() below rather than .env,
since "which of three named regimes" isn't naturally a single tunable value.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# paths.py lives at the project root; training/config.py is one level down.
# Bootstrap the root onto sys.path so this works standalone (e.g. a test
# importing training.config directly) and not just when launched via
# train.py/main.py, which already do their own root-level sys.path setup.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    MIRROR_PROB,
    DIVERSITY_WINDOW,
    DIVERSITY_COEF,
    OVERTRADE_WINDOW,
    OVERTRADE_FREE_TRADES,
    OVERTRADE_PENALTY_COEF,
    PLATFORM_FEE_PER_TRADE,
)


# --------------------------------------------------------------------------
# Ticker universe -- MUST match fetch_alpaca.py's TICKERS list. Only used by
# dataset.py/training indirectly (auto-discovered from metadata.json
# instead); but live_loop.py's LiveLoop/AlpacaBarPoller read this list
# directly (no dataset object exists in live mode) -- keep the two lists in
# sync manually whenever either changes, nothing enforces it automatically.
# --------------------------------------------------------------------------

DEFAULT_TICKERS: List[str] = [
    # Broad ETFs (10)
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    # Technology & Software (18)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD",
    "QCOM", "INTC", "MU", "TXN", "ORCL", "CRM", "ADBE", "NOW", "PANW",
    # Financials & FinTech (15)
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW", "V", "MA",
    "AXP", "PYPL", "SQ", "COIN", "BRK.B",
    # Healthcare & Biotechnology (14)
    "JNJ", "PFE", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "VRTX",
    # Consumer Discretionary & Staples (15)
    "WMT", "COST", "PG", "KO", "PEP", "NKE", "HD", "MCD", "SBUX",
    "TGT", "LOW", "PM", "MO", "CL", "MDLZ",
    # Energy & Utilities (10)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "NEE", "DUK",
    # Industrials, Aerospace & Defense (12)
    "CAT", "GE", "BA", "LMT", "RTX", "HON", "DE", "UNP", "UPS",
    "FDX", "MMM", "GD",
    # Communications & Entertainment (6)
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
]


# --------------------------------------------------------------------------
# Friction presets -- toggle how expensive/realistic execution is without
# hand-editing every cost field. Only the fields that actually represent
# "friction" (spread, impact, participation cap, commissions, platform fee)
# are touched; everything else in EnvConfig (tickers, window_size, mirroring,
# reward shaping, etc.) is untouched by this. The overtrading penalty is NOT
# a friction any more -- it is reward shaping and lives in RewardConfig -- so
# only its free-trade allowance appears here, as a churn-window definition.
#
#   low        -- near-frictionless: tight spread, minimal impact, high
#                 participation allowance, zero commissions/fees. Useful for
#                 isolating "does the policy have any edge at all" from
#                 "can it survive realistic costs" -- if it can't be
#                 profitable here, it won't be profitable anywhere.
#   realistic  -- the project's original defaults (retail/prosumer-tier
#                 costs on 5-min US equity bars). This is what go/no-go
#                 backtests and live trading should use.
#   high       -- stress test: wide spread, heavy impact, tight participation
#                 cap, real commissions + platform fee, a near-zero free-trade
#                 allowance. If the policy is still profitable
#                 here, realistic-mode profitability has real margin of
#                 safety; if it collapses, you've found how fragile the
#                 edge is to execution assumptions before deploying capital
#                 on it.
#
# Usage:
#     cfg = TrainingConfig()
#     cfg.env = EnvConfig.for_friction("high")                    # swap the whole preset
#     cfg.env = EnvConfig.for_friction("low", tickers=my_tickers)  # preset + explicit overrides
# --------------------------------------------------------------------------

FRICTION_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": dict(
        spread_bps=0.0,
        impact_coef=0.25,             # Y prefactor of the sqrt law -- see EnvConfig.impact_coef
        max_participation=0.5,
        commission_per_share=0.0,
        commission_bps=0.0,
        min_commission=0.0,
        platform_fee_per_trade=0.0,
        overtrade_free_trades=9999,   # effectively disables the overtrading penalty
    ),
    "realistic": dict(
        # 1.0 -> 0.0. The half-tick floor IS the spread model now (see
        # EnvConfig.spread_bps); a proportional term on top would be an extra
        # assumption with nothing behind it.
        spread_bps=0.0,
        # 0.015 -> 0.5. NOT a 33x increase in charged impact -- the units
        # changed. impact_coef is now the dimensionless Y of
        # impact/price = Y * sigma_daily * sqrt(Q/ADV), with sigma_daily
        # measured per ticker instead of folded into the constant, and ADV
        # replacing bar volume in the denominator. 0.015 was Y * sigma_daily
        # collapsed into one number against a per-BAR denominator; 0.5 is Y
        # alone, at the low end of the literature's 0.5-1.0. Net effect on a
        # small order is a LARGE reduction, because the denominator grew by
        # ~78x and sqrt(x) >> x for small x.
        impact_coef=0.5,
        max_participation=0.1,
        # 0.5 -> 0.0. Alpaca, the venue live/broker_client.py actually
        # targets, is commission-free on US equities. 0.5 bps was a
        # placeholder for a broker this project does not use.
        commission_per_share=0.0,
        commission_bps=0.0,
        min_commission=0.0,
        platform_fee_per_trade=PLATFORM_FEE_PER_TRADE,
        overtrade_free_trades=OVERTRADE_FREE_TRADES,
    ),
    "high": dict(
        spread_bps=5.0,
        impact_coef=1.5,              # above the literature's Y range, deliberately
        max_participation=0.03,
        commission_per_share=0.005,
        commission_bps=1.5,
        min_commission=1.0,
        platform_fee_per_trade=1.0,
        overtrade_free_trades=1,
    ),
}


@dataclass
class EnvConfig:
    """Mirrors vec_trading_env.VecTradingEnv's constructor -- keep these in sync if that file's defaults change."""

    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    window_size: int = 60
    # DELIBERATELY 10_000: this mirrors the capital the system will actually
    # be deployed with, and a sim funded 10x richer than the real account
    # teaches the policy trade economics it will never encounter.
    #
    # KNOWN CONSEQUENCE, measured -- do not "fix" this by raising the number.
    # The platform ticket fee is a FLAT $1/trade, so its cost as a fraction
    # of notional is set purely by order size. At this account size
    # max_order_notional_frac (0.10) caps an order at $1,000 and
    # max_position_frac (0.25) caps the position at $2,500, so:
    #     one-way cost ~29.8 bps   vs   measured gross edge ~7.06 bps
    #     => roughly -13.5 bps per trade, i.e. structurally unprofitable
    # (measured over rollouts 31-49 of a real 50-rollout run: gross +$3,140,
    # ticket fees $7,023, net -$6,004 on $4.45M of traded notional; the
    # decomposition closes to within $3).
    #
    # Break-even needs ~$3,000+ per order. Capital per stream is the binding
    # constraint -- no setting of kelly_multiplier / min_order_notional can
    # clear a flat fee at $10k/stream. The way OUT that preserves a small
    # real account is CONCENTRATION, not more capital: the same $1M funds
    # either 100 streams x $10k (max order $1,000, unusable) or 10 streams
    # x $100k (max order $10,000, viable). That means shrinking the ticker
    # universe (n_envs is one stream per ticker), which needs metadata
    # regeneration and a model reshape -- see the notes in Version_2/AGENTS.md.

    # ---------------------------------------------------------------------
    # P1 VOIDED THE ARITHMETIC ABOVE. Do not act on it without re-deriving.
    # Every cost number in this block was measured against a model charging a
    # flat $1 ticket, 0.5 bps commission, a 1.0 bps proportional spread, and
    # sqrt-impact against 5-MINUTE BAR volume. The venue actually traded
    # (Alpaca) charges none of the first two, the spread is a half-tick, and
    # impact belongs against daily ADV. Measured on the val split with the
    # same fixed policy in both arms: cost_per_turnover 21.704 -> 1.024 bps,
    # an overstatement of 21.2x.
    #
    # So "one-way cost ~29.8 bps vs gross edge ~7.06 bps" is now roughly
    # "~1.0 bps vs an edge that has not been measured honestly yet", and the
    # conclusion that followed from it -- that $10k/stream is structurally
    # unprofitable and only CONCENTRATION can fix it -- no longer follows from
    # its own premise. The flat ticket was 81.5% of all measured losses and it
    # is gone; a flat fee is what made order size binding, and there is no
    # flat fee any more.
    #
    # The value is LEFT AT 10_000 regardless, because it still mirrors the
    # capital this system would deploy, which was always the primary reason.
    # What is retracted is the claim that this number is a structural blocker.
    # Re-measure alpha_per_turnover against cost_per_turnover before treating
    # account size as a constraint again.
    # ---------------------------------------------------------------------
    initial_cash: float = 10_000.0
    max_position_frac: float = 1.0
    tick_size: float = 0.01
    friction_level: str = "realistic"   # "low" | "realistic" | "high" -- see FRICTION_PRESETS above.
                                          # Informational once set via for_friction(); the fields below
                                          # are what VecTradingEnv actually reads.
    # PROPORTIONAL half-spread in bps, charged ON TOP OF the half-tick floor
    # that execution_sim.py applies unconditionally. 1.0 -> 0.0: the minimum
    # quotable US equity spread is one tick, so half a tick is the floor and
    # also, on the liquid large caps in this universe, a fair estimate of the
    # whole thing. Because it is a fixed $0.005 against a per-ticker price,
    # that floor is already per-ticker in bps -- 5.2 bps on a $9.66 name
    # against 0.09 bps on a $557 one. Raise this only for a ticker whose
    # inside market is demonstrably wider than a tick.
    spread_bps: float = 0.0
    # The dimensionless Y prefactor of the square-root impact law,
    # impact/price = Y * sigma_daily * sqrt(Q/ADV). sigma_daily and ADV are
    # measured per ticker by the env from the split being traded, so this is
    # the only free number, and the literature pins it to 0.5-1.0 -- it is not
    # a knob to tune until a strategy looks good.
    #
    # 0.015 -> 0.5 is a UNIT change, not a 33x cost increase. The old value
    # was Y * sigma_daily collapsed into one constant, multiplied by
    # sqrt(order / 5-MINUTE BAR volume). Dividing by a bar rather than by a
    # day overstates participation by roughly the 78 bars in a session, and
    # sqrt(78) ~ 8.8, so the charged impact on a small order FALLS
    # substantially under the new formulation. Kept in sync with
    # FRICTION_PRESETS["realistic"] -- these dataclass defaults (not the
    # preset) are what training actually reads, since train.py builds
    # TrainingConfig() directly rather than going through
    # EnvConfig.for_friction().
    impact_coef: float = 0.5
    max_participation: float = 0.1
    commission_per_share: float = 0.0
    # 0.5 -> 0.0. Alpaca is commission-free on US equities and is the venue
    # live/broker_client.py targets. Charging 0.5 bps modelled a broker this
    # project does not trade through.
    commission_bps: float = 0.0
    min_commission: float = 0.0
    platform_fee_per_trade: float = PLATFORM_FEE_PER_TRADE

    # Which bar-t+1 price the env fills and marks against: "open" (the
    # opening print, default) or "vwap" (the size-aware variant -- what an
    # order worked across the bar would average into). "close" is not an
    # option: marking close[t] -> close[t+1] books the bid-ask bounce, a
    # mean-reverting ~0.47 bps/bar component of the return series that no
    # order can capture, and optimising against it is optimising against an
    # artifact of how the data is recorded. The observation still ends at
    # bar t either way. See vec_trading_env.py's module docstring.
    execution_price_column: str = "open"
    # 0.5 -> 2.0. Scales the vol-normalized step-PnL term, which is the only
    # DIRECTLY PnL-aligned signal in the whole reward. Measured over 200 steps
    # x 100 streams it was the SMALLEST component of the env reward
    # (mean|x| 0.000255) -- smaller than diversity_bonus (0.00118) and
    # hold_loser_penalty (0.000404). Raising it 4x makes PnL the dominant term
    # inside StepResult.reward, which is the point of switching raw_weight on.
    r_step_scale: float = 2.0
    hold_loser_penalty: float = 0.0005
    # mirror_prob defaults to 0.0 as of P1 -- mirroring fabricates a
    # cross-section that does not exist (in-sim pairwise rho 0.001 against a
    # true 0.256). enable_mirroring is left True as the switch that would turn
    # it back on, but the env treats prob 0 as off outright. See paths.py's
    # MIRROR_PROB for the full reasoning before changing either.
    enable_mirroring: bool = True
    mirror_prob: float = MIRROR_PROB
    overtrade_window: int = OVERTRADE_WINDOW
    overtrade_free_trades: int = OVERTRADE_FREE_TRADES
    # NOTE: the overtrading penalty's COEFFICIENT lives in RewardConfig, not
    # here -- it is reward shaping, not a venue cost. The two fields above are
    # what DEFINES the churn window, which is an env property, so they stay.
    bias_window: int = DIVERSITY_WINDOW
    diversity_bonus_coef: float = DIVERSITY_COEF
    # Bars a stream must wait after INCREASING exposure before it may increase
    # again (closes/reduces are never blocked). 12 bars = 1 hour of 5-min bars,
    # matching overtrade_window. 0 disables. See
    # VecTradingEnv._apply_trade_cooldown() for the original measurements.
    #
    # TRIED 0, REVERTED TO 12 ON MEASURED EVIDENCE. Keep it at 12.
    #
    # The argument for removing it was that its only stated justification was
    # cost arithmetic and P1 had voided that: the original case was "a round
    # trip costs ~17 bps against a ~5.6 bps median 5-min move, so hold tens of
    # bars", and under the recalibrated cost model a round trip is ~1.6 bps, so
    # a one-bar hold clears its own cost several times over. That reasoning was
    # correct as far as it went, and it was still the wrong call, because the
    # cooldown was doing a second job nobody was pricing.
    #
    # MEASURED, cooldown=0 vs cooldown=12, matched at 93 rollouts:
    #
    #                        cd=12         cd=0
    #     turnover      $159,799,003  $368,718,253   (2.3x)
    #     fills              240,043       523,958   (2.2x)
    #     cost_per_turnover    0.781         0.840   (off the floor)
    #     alpha_per_turnover  +0.015        -0.033   (did NOT follow)
    #     net worth           -1.20%        -3.12%
    #
    # That is exactly the predicted failure signature: cost climbing off its
    # ~0.78 bps floor while alpha does not follow, i.e. paying more spread for
    # the same (absent) edge. Churn returned the moment the hard brake came
    # off.
    #
    # THE REAL LESSON, worth more than the number: the cooldown was never only
    # a cost constraint. It is the only HARD brake on churn, and it was made
    # hard in the first place precisely because two earlier runs "optimised the
    # reward well while still churning" -- the soft penalty had already failed
    # once. RewardConfig.overtrade_penalty_coef is that soft penalty, and this
    # run is its second failure: it contributed ~0.2% of the reward and did not
    # bind. Retiring a constraint because ONE of its justifications expired,
    # without checking what else it was holding up, is the mistake here.
    #
    # If this is revisited: raise overtrade_penalty_coef until it demonstrably
    # binds BEFORE touching this, and change one of the two at a time.
    trade_cooldown_bars: int = 12

    # Minimum bars a position must be held before the policy may reduce it.
    #
    # trade_cooldown_bars above blocks re-ENTRY; this blocks EXIT. They are
    # complements, and only having the first one is what produced the last
    # run's defining number: 140 of 140 completed round trips lasted exactly
    # one bar -- median, mean, p90 and max all 1.0. Blocking re-entry cut trade
    # frequency but left "open, close next bar, sit out the cooldown" as the
    # policy's best available move.
    #
    # That is the gap between the alpha gate and the equity curve. The gate
    # found its edge at 30min-1hr horizons (6-12 bars). A 1-bar hold captures
    # about one bar of that move (median 9.84 bps) against the full 7.28 bps
    # round trip, every time. Holding h bars scales the move by sqrt(h) while
    # the cost stays fixed.
    #
    # Set this to the horizon of whichever alpha_lab cell actually passed --
    # 6 for a 30min cell, 12 for a 1hr cell. Left at 0 (off) by default so
    # nothing changes until that number is known; enabling it blind would just
    # be a different arbitrary constraint.
    min_hold_bars: int = 0

    # Bars of the session on which the policy may OPEN or ADD exposure, as a
    # half-open [start, end) range of bar-of-day indices (0 == 09:30, 77 is the
    # last RTH bar). Reductions and the session-close flatten are never gated.
    #
    # Median |5-min move| by regime, on the marking price path: open hour
    # 18.6 bps, close ramp 11.9, midday 9.6 -- against a round-trip cost near
    # 7.3 bps that does not vary with time of day. Spreading the cost budget
    # uniformly across all 77 tradable bars spends most of it where the move
    # barely clears the spread.
    #
    # Match this to the regime of whichever alpha_lab cell passed:
    #     open_hour  -> (0, 12)
    #     midday     -> (12, 72)
    #     close_ramp -> (74, 77)
    # None (default) means every bar is tradable, i.e. previous behaviour.
    trading_window: Optional[Tuple[int, int]] = None

    # Close every position on the last bar of each trading session.
    #
    # Before this flag the env had no notion of a session at all, so overnight
    # exposure was carried by default -- not chosen, just never prevented. It
    # was already happening: of only 14 round trips in the last run's tick log
    # (the dead phase, 28 fills out of 257,137 trades) one was opened at 15:55
    # and closed at 09:35 the next morning.
    #
    # Measured on 81 sessions of adjusted data, the overnight bar carries
    # sigma 171.8 bps vs 23.0 bps intraday and a worst case of -3,681 bps vs
    # -581 bps. Every risk control in this project is step-based -- KillSwitch,
    # RiskManager, the drawdown halt -- so none of them run across those 17.5
    # hours. And the apparent reward is not edge: the overnight bar's 7.7x
    # better move-to-cost ratio is exactly its 7.5x larger sigma. A bigger bet
    # is not a better bet.
    #
    # Set False only as a deliberate decision, alongside overnight-specific
    # risk machinery (per-name gap caps, separate overnight sizing) and
    # separate accounting for overnight vs intraday PnL -- the overnight
    # premium is beta, and booking it as alpha is the mistake that cost this
    # project two sessions.
    flatten_at_session_close: bool = True

    @classmethod
    def for_friction(cls, level: str, **overrides: Any) -> "EnvConfig":
        """
        Builds an EnvConfig starting from the named friction preset
        (FRICTION_PRESETS above), then applies any explicit field overrides
        on top (e.g. a custom ticker list). Raises ValueError on an unknown
        level rather than silently falling back to a default -- a typo'd
        friction level should never silently run at the wrong cost regime.
        """
        if level not in FRICTION_PRESETS:
            raise ValueError(f"Unknown friction_level {level!r}; must be one of {list(FRICTION_PRESETS)}")
        cfg = cls(friction_level=level, **FRICTION_PRESETS[level])
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg


@dataclass
class ModelConfig:
    """Mirrors cnn_encoder / lstm_encoder / cross_attention / fusion / hybrid_policy / dual_critic constructors."""

    cnn_channels: Tuple[int, ...] = (64, 64, 128)
    cnn_kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    cnn_dilations: Tuple[int, ...] = (1, 2, 4)
    cnn_dropout: float = 0.1

    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0

    attn_num_heads: int = 4
    attn_ffn_dim: Optional[int] = None  # None -> cross_attention.py defaults to embed_dim * 4
    attn_dropout: float = 0.1

    trunk_dim: int = 256
    fusion_dropout: float = 0.1

    policy_direction_embed_dim: int = 16
    policy_hidden_dim: int = 64
    policy_min_concentration: float = 1.0  # see hybrid_policy.py docstring point 1 before lowering this
    policy_dropout: float = 0.1

    critic_hidden_dim: int = 64
    critic_dropout: float = 0.1

    # ppo_hybrid.py's ppo_update() -- gradient-checkpoints the CNN's
    # T*n_envs mega-batch forward pass (the single largest activation-memory
    # consumer in the replay path, since it scales linearly with n_envs).
    # Trades ~30% more compute (the CNN forward is recomputed during
    # backward instead of its activations being kept) for a real reduction
    # in peak VRAM -- worth it once n_envs gets large enough that memory,
    # not compute, is the constraint (see training/config.py's module-level
    # note on the 100-ticker scaling question). Off by default: verify it
    # doesn't change training behavior on your setup before trusting it,
    # same caution as use_amp below. Only the CNN is checkpointed here, not
    # the LSTM loop -- checkpointing across the sequential, hidden-state-
    # threading LSTM replay is a correctness risk this project's own
    # ppo_hybrid.py docstring warns about at length; not attempted.
    use_gradient_checkpointing: bool = False


@dataclass
class RiskConfig:
    # risk_manager.py -- RiskLimits
    max_position_frac: float = 0.25
    max_gross_exposure_frac: float = 1.0
    max_ticker_concentration_frac: float = 0.35
    # 0.10 -> 0.20. At $10k/stream the old value capped every order at
    # $1,000, BELOW what Kelly wants, so a flat non-edge-aware cap was the
    # binding constraint instead of the edge-aware one. Raising it lets
    # KellySizer bind first, which is the intended pipeline order. Justified
    # by the deployment shape: each stream is an INDEPENDENT single-asset
    # account (n_tickers=1 per PortfolioState), and the plan is to deploy on
    # one ticker -- so these are single-account concentration limits, not
    # portfolio-diversification limits. max_position_frac (0.25) still bounds
    # the resulting position.
    max_order_notional_frac: float = 0.20
    drawdown_halt_frac: float = 0.20

    # Absolute $ floor on exposure-INCREASING orders -- see
    # risk_manager.py's _drop_dust_orders() for the full measured
    # rationale. Short version: a 50-rollout run produced a MEDIAN fill of
    # $0.20 (73.8% of fills under $1.00) while paying a flat $1.00/ticket
    # fee, which was 99.3% of that run's entire loss. Filtering at $100
    # drops ~86% of orders but keeps ~98.8% of traded notional. Applies in
    # training, backtest and live alike. 0.0 disables.
    # MUST stay below max_order_notional_frac * initial_cash (currently
    # 0.10 * 10_000 = $1,000) or NO exposure-increasing order can ever pass
    # this gate and training deadlocks with zero fills -- the hard-caps
    # version of the Kelly chicken-and-egg described under
    # kelly_min_fraction. Re-derive this whenever EnvConfig.initial_cash
    # moves; it is an ABSOLUTE dollar floor, not a fraction.
    # 100 -> 500. Now a genuine FEE FLOOR, not just a dust gate: any order
    # that passes pays at most 1e4/500 = 20 bps of ticket. MUST stay below
    # both max_order_notional_frac * initial_cash ($2,000) and
    # kelly_default_fraction * initial_cash ($800) or nothing can fill and
    # training deadlocks -- re-derive all four together if any moves.
    min_order_notional: float = 500.0
    # Ceiling on the above as a fraction of CURRENT equity, so a drawdown can
    # never make the dust gate unpassable. MUST stay strictly below
    # kelly_min_fraction (0.08) -- RiskManager._drop_dust_orders() carries the
    # proof. The 2026-08-19 run froze all 100 streams at exactly
    # $500/0.08 = $6,250 for want of this.
    min_order_equity_frac: float = 0.04

    # kelly_sizing.py -- KellySizer
    kelly_lookback_trades: int = 30
    kelly_min_trades_for_estimate: int = 10
    # 0.25 -> 0.5 (half-Kelly). At initial_cash=10_000 quarter-Kelly left the
    # typical order at ~$325, where the flat $1 ticket alone is 30.8 bps --
    # against a measured gross edge of only 4-11 bps/trade, i.e. structurally
    # unwinnable. Half-Kelly doubles it to ~$650-800 and, with the
    # max_order_notional_frac change below, brings total one-way cost from
    # ~35.5 bps to ~17.2 bps. Half-Kelly is the standard conservative
    # operating point; full Kelly would size better still but is too
    # high-variance to run unattended.
    kelly_multiplier: float = 0.5
    kelly_cap: float = 1.0
    # PRE-WARM sizing fraction: sets the order cap during exactly the period
    # when no stream has enough closed trades for a real edge estimate. Keep
    # `kelly_default_fraction * initial_cash` comfortably ABOVE
    # min_order_notional (here 0.02 * 10_000 = $200 vs a $100 gate) or every
    # pre-warm order is dropped as dust and the edge estimate can never warm.
    kelly_default_fraction: float = 0.08

    # TRAINING-ONLY floor on the post-multiplier fractional Kelly (see
    # kelly_sizing.py's __init__ docstring). Consumed ONLY by train.py's
    # build_risk_pipeline(), which is called only by train.py /
    # train_ddp.py / hyperparam_sweep.py -- eval/backtest_report.py and
    # live/live_loop.py build their own KellySizer without this argument
    # and keep the strict min_fraction=0.0 behavior, so live/backtest risk
    # semantics are unchanged.
    #
    # Why it exists (measured, not theoretical): with min_fraction=0.0 the
    # first training run to log Kelly diagnostics showed ALL 100 streams at
    # fractional_kelly == 0.0 by the END of rollout 0 -- mean win_rate
    # 0.0156, mean payoff_ratio 0.0672, 73/100 streams with literally zero
    # winning trades. Reconstructing those same round trips at mid price
    # (no slippage) gave a fair 44% win rate, so the edge estimate was
    # measuring execution cost, not policy skill. Once the cap hits 0 no
    # position can be opened, so no trade closes, so the estimate can never
    # update: total_trades froze at 4359 and net worth sat at exactly
    # 994497.19 for 1792 consecutive ticks. Same lock reappeared on every
    # restart. Kept EQUAL to kelly_default_fraction (the size used pre-warm),
    # i.e. "keep exploring at the conservative default size" rather than
    # "stop trading forever the first time the estimate turns negative".
    # If either is retuned, move BOTH, and keep the floor strictly above
    # min_order_notional / initial_cash or the deadlock returns.
    kelly_min_fraction: float = 0.08

    # P2 bullet 4. "realized" is the historical behaviour: f* is accumulated
    # from CLOSED round trips, and a stream needs kelly_min_trades_for_estimate
    # of them before is_warm goes true. That estimator cannot warm inside a
    # cost-aware cross-sectional book -- at ~0.036 gross turnover per bar with
    # ~5 names held at a time, most streams close single-digit round trips per
    # rollout, so the governor sits at kelly_default_fraction for the whole run
    # and is a constant wearing Kelly's name.
    #
    # "model" sizes from the network's own forward-looking edge instead
    # (KellySizer.set_model_edge), which is warm on the first bar. It requires
    # a caller that actually feeds it every step; train.py does this only when
    # the actor-critic carries a pre-trained edge head, and refuses to start
    # otherwise rather than silently sizing on a stale estimate.
    #
    # NOTE, measured: at a per-name edge of a few bps against a ~55 bps
    # 24-bar return sd, continuous Kelly f* = mu/sigma^2 comes out in the tens
    # and saturates kelly_cap. The model source is therefore close to a binary
    # -- cap when the edge clears its cost, zero when it does not -- and its
    # value is the ZERO, not the sizing.
    kelly_edge_source: str = "realized"

    # Skip the Kelly governor entirely during training and keep it for live
    # only (the brief's second option). The cap still exists in
    # live/live_loop.py and eval/backtest_report.py, which build their own
    # sizer; this switch only affects the training pipeline.
    kelly_enabled_in_training: bool = True

    # kill_switch.py -- KillSwitch
    daily_loss_limit_frac: float = 0.03
    broker_error_streak_limit: int = 3
    state_mismatch_tolerance: float = 1e-3

    # TRAINING-ONLY (train.py / train_ddp.py, not live_loop.py): how often,
    # in rollouts, to force kill_switch.reset() + start_new_day() AND
    # portfolio.reset_peak_equity() regardless of episode boundaries.
    # kill_switch.py's daily_loss_limit_frac (3% by default) trips easily
    # against a random early-training policy, and risk_manager.py's
    # drawdown_halt_frac (20%) compares against a peak_equity that is
    # otherwise a true all-time high -- an "episode" here means one full
    # pass through the ENTIRE training dataset, potentially tens of
    # thousands of bars. Without a shorter reset cadence for BOTH of these,
    # a stream tripped/underwater early stays stuck for the rest of that
    # enormous episode: not visibly broken (it still marks-to-market every
    # tick, so its numbers keep moving), just silently contributing zero
    # real trading signal for a very long time. 1 = reset every rollout
    # (treats each 256-bar rollout as its own "day" for both purposes) --
    # aggressive, but appropriate for training where the actual goal is
    # exploration, not a realistic daily-loss simulation (that's what
    # eval/backtest_report.py and live/live_loop.py are for, and neither of
    # those uses this field).
    #
    # DIAGNOSTIC NOTE (episode-76-style "stopped trading, rollouts just
    # roll by" stalls): check tick_record["halted"] in the metrics log
    # (also visible live on the dashboard's HALTED badge/row) AND the
    # per-env drawdown column FIRST. If halted is False everywhere but
    # trades still aren't happening, the culprit is very likely
    # risk_manager.py's drawdown_halt_frac blocking new exposure via
    # reduce_only mode (see PortfolioState.reset_peak_equity()'s docstring)
    # -- that's a SEPARATE mechanism from KillSwitch and won't show up as
    # "halted" at all, only as persistently elevated per-ticker drawdown
    # with a flat/non-climbing trade count. If KillSwitch itself is stuck
    # halted despite this reset cadence, check whether
    # state_mismatch_tolerance or broker_error_streak_limit is tripping
    # repeatedly instead (unlikely during training, since nothing in
    # train.py/train_ddp.py calls record_broker_error() -- that path is
    # live-only).
    kill_switch_reset_every_n_rollouts: int = 1

    # ABSOLUTE BACKSTOP on order size, no longer the primary scale.
    #
    # This used to be the ceiling the Beta size head's (0,1) output was mapped
    # onto directly. At 10,000 shares against $10,000-per-stream equity and a
    # ~$150 median price that is $1.5M of requested notional -- about 161x
    # equity -- while the Kelly cap downstream permits 8% (~$744, 4.96 shares).
    # So 99.95% of the head's output range mapped to "capped", it received no
    # usable gradient, entropy_continuous collapsed to 0.000, and every
    # observed fill across three consecutive runs came in at exactly
    # 0.08 x equity. Sizing was a constant by construction.
    #
    # The mapping ceiling is now derived from equity via
    # HybridPolicyHead.size_cap_shares (max_order_notional_frac x equity /
    # price), which puts the Kelly cap inside the policy's range. This value
    # survives only as a hard clamp against a pathological price or equity.
    max_order_shares: float = 10_000.0
    # 20.0 -> 0.0: DISABLES the limit_offset action, i.e. every order is
    # modelled as marketable and pays the full spread. Not a tuning choice --
    # execution_sim.py cannot currently price a passive order honestly. It
    # fills them unconditionally at the requested price with no rejection and
    # no adverse selection, so ANY non-zero offset is free money: before the
    # clamp in _compute_fill_price() a 20-tick offset filled 207 bps below mid
    # on a $9.66 stock, and even after the clamp it still zeroes the entire
    # spread + impact cost at every price level, which a real resting order
    # never does (it may simply not fill, and when it does it is usually
    # because the market just moved against you).
    #
    # This is what a 151-rollout run farmed to +486% net worth while its
    # per-bar directional hit rate was 41.6% -- WORSE than a coin flip -- with
    # 93/100 streams ending below their starting capital and a -0.921 rank
    # correlation between ticker price and final equity.
    #
    # Assuming every order crosses the spread is the CONSERVATIVE assumption:
    # it can understate performance, never overstate it. Re-enable only
    # alongside a real passive-fill model (fill only when the bar's high/low
    # trades through the limit price -- both columns exist in the parquet,
    # only `close` is loaded today).
    max_limit_offset_ticks: float = 0.0


@dataclass
class RewardConfig:
    """training/reward.py -- DifferentialSharpeReward, plus how it blends with vec_trading_env.py's own shaped reward."""

    dsr_eta: float = 0.01
    dsr_eps: float = 1e-8
    dsr_warmup_steps: int = 2
    dsr_clip: float = 10.0
    sharpe_weight: float = 1.0   # weight on the Sharpe-shaped term
    # 0.0 -> 80.0. StepResult.reward was multiplied by ZERO, so the env's own
    # vol-normalized PnL term, hold_loser_penalty and diversity_bonus never
    # reached the policy at all -- the entire objective was the differential
    # Sharpe term, which measured only +0.147 correlation with actual PnL
    # (R^2 ~ 0.02). 80 is not arbitrary: measured over 200 steps x 100 streams
    # the DSR term averages |0.1447| against the env reward's |0.0018|, a 79x
    # scale gap, so anything of order 1 here contributes ~1% and does nothing.
    # At 80 (with r_step_scale=2.0 and DIVERSITY_COEF=0.015) the blend lands
    # near DSR 50% / step-PnL 28% / hold-penalty 11% / diversity 10%.
    # NOTE: terminal_alpha inside that reward measures exactly 0.000000 here --
    # it only fires on episode end, and an episode is a full pass over the
    # train split (~118k steps) versus ~13k steps in a 51-rollout run.
    raw_weight: float = 80.0     # weight on vec_trading_env.py's own StepResult.reward (vol-normalized step
                                  # reward + hold penalty + diversity bonus + terminal alpha -- see that
                                  # file's docstring). Default 0 means training relies purely on the
                                  # Sharpe-shaped signal; raise this if you want the env's own shaping
                                  # (in particular the directional-bias diversity bonus) to also count.

    # training/reward.py's DifferentialSharpeReward -- symmetric checkpoint
    # bonus/penalty layered on top of D_t whenever a stream's cumulative
    # return since the last reset crosses a NEW multiple of dsr_checkpoint_step
    # (in either direction -- see that file's module docstring). Disable
    # entirely with dsr_enable_checkpoint_bonus=False to recover the plain
    # differential-Sharpe reward.
    dsr_enable_checkpoint_bonus: bool = True
    dsr_checkpoint_step: float = 0.025          # 2.5% cumulative-return milestone spacing
    dsr_checkpoint_bonus_frac: float = 0.10     # +/- 10% of |D_t| per new milestone crossed

    # --- Overtrading penalty (moved here from the cost model, P1) --------
    # Subtracted from vec_trading_env.py's StepResult.reward as
    #     overtrade_penalty_coef * overtrading_factor
    # where overtrading_factor is 0 for a stream inside its free-trade
    # allowance (EnvConfig.overtrade_free_trades within
    # EnvConfig.overtrade_window bars) and 1 for one trading every bar.
    #
    # It used to be EnvConfig.overtrade_surcharge_bps, charged as extra
    # adverse slippage inside execution_sim.py. That made it a fake venue
    # cost, and the damage was to measurement rather than to behaviour: the
    # fill price carried it, so both the measured cost AND the measured edge
    # (which is PnL against that same fill price) were contaminated, and
    # neither could be compared against a broker execution report. Penalising
    # churn is a legitimate thing to want; pricing it as slippage meant the
    # instrument could not be read.
    #
    # THE OLD VALUE OF 3.0 DOES NOT TRANSFER -- that was bps of notional per
    # fill, this is reward units per step. See paths.py's
    # OVERTRADE_PENALTY_COEF for how 0.002 was sized against the measured
    # magnitudes of the reward terms it sits beside. Note it is scaled by
    # RewardConfig.raw_weight along with everything else in StepResult.reward.
    # 0.0 disables it.
    overtrade_penalty_coef: float = OVERTRADE_PENALTY_COEF


@dataclass
class PPOConfig:
    rollout_length: int = 256            # T env-steps per stream, per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    value_clip_range: float = 0.2
    entropy_coef_discrete: float = 0.05    # applied to the direction head's entropy independently.
                                            # Raised 0.02 -> 0.05: 0.02 let PPO collapse to
                                            # always-FLAT ("stopped trading entirely") as a locally
                                            # safe optimum against real transaction costs -- see
                                            # hyperparam_sweep.py's SweptParam docstring. 0.05 keeps
                                            # enough exploration pressure on the SHORT/FLAT/LONG head
                                            # to keep trading, without the 0.10 over-regularization.
    entropy_coef_continuous: float = 0.02  # applied to the (FLAT-masked) size + limit_offset entropy
                                            # independently. Raised 0.01 -> 0.02 to keep the Beta
                                            # size/limit_offset heads exploring instead of collapsing.

    # --- entropy collapse guard -----------------------------------------
    # entropy_coef_discrete above is the INITIAL value when the controller is
    # on; the controller then moves it. See ppo_hybrid.AdaptiveEntropyCoef for
    # why a constant cannot hold: the last run collapsed to
    # entropy_discrete = 0.000 at rollout 81 with entropy_coef_discrete
    # already raised to 0.05, and spent the remaining 65 rollouts (43% of the
    # run, 124 trades total) unable to explore -- having seen 22% of the
    # training split at the moment it stopped.
    #
    # Set target_entropy_discrete to None to restore the old fixed-coefficient
    # behaviour exactly.
    target_entropy_discrete: "Optional[float]" = 0.5   # nats; ln(3)=1.0986 is the max
    entropy_coef_lr: float = 0.1                        # dual-ascent step, per PPO epoch

    # Asymmetry: how much faster the coefficient climbs (entropy BELOW target)
    # than it decays (entropy above). 1.0 restores the symmetric controller.
    #
    # The symmetric version is a pure integrator, so it can only respond to
    # ACCUMULATED error and always lags. Measured on the run 19 trace: entropy
    # crossed the 0.5 target at rollout 46, and the coefficient did not reach
    # the level that actually reversed the fall (~0.20) until rollout 65 --
    # about 19 rollouts of phase lag, during which entropy undershot to 0.156,
    # 31% of target. It then overshot back to 0.988, ~2x target. The run
    # survived (unlike the terminal collapse in report.md Sec 12) but settled
    # into a limit cycle rather than holding near target.
    #
    # The two directions are not equally costly, so a symmetric response is
    # the wrong prior: undershooting collapses the policy into always-FLAT and
    # ends the run's usefulness, while overshooting merely spends some
    # exploration budget. Only the CLIMB is accelerated -- the decay stays at
    # lr, so an overshoot still unwinds at the old rate and cannot latch high.
    #
    # 3.0 rather than something larger. Replaying run 19's measured entropy
    # trace through the controller, the coefficient at rollout 59 reaches:
    #     mult 1.0 -> 0.12   (what actually happened; too slow)
    #     mult 2.0 -> 0.30
    #     mult 4.0 -> 1.80   <- near max_coef, the "policy is just noise" end
    #     mult 6.0 -> 2.00   (saturated)
    # That replay is OPEN-LOOP -- it feeds the historical entropy regardless of
    # what the coefficient does, so it overstates the peak (in a real run the
    # rising coefficient lifts entropy, the error flips sign, and the climb
    # stops). But it brackets the risk, and 4.0+ sits close enough to
    # saturation that the extra speed is not worth it. 3.0 cuts the lag roughly
    # threefold; entropy should turn near 0.30 rather than 0.156.
    #
    # SATURATION SIGNATURE, if this is ever retuned upward: entropy_coef_
    # discrete climbing past ~1.0 while entropy_discrete is STILL falling.
    # That is the controller losing the race, not winning it, and the answer is
    # a lower multiplier or a different lever -- not a higher max_coef.
    entropy_coef_lr_up_mult: float = 3.0

    # entropy_coef_min raised 0.005 -> 0.05, to equal entropy_coef_discrete's
    # init value. A fresh discrete head starts near ln(3), well above the 0.5
    # target, before it has learned anything -- the controller cannot tell
    # that apart from real over-exploration and immediately starts shrinking
    # the coefficient. On the run analysed in report.md Sec 12 that shrink hit
    # the old 0.005 floor by rollout 13-14 while entropy_discrete was still
    # >1.0, stripped out the entropy bonus for the 11+ rollouts where the
    # cost-driven collapse actually took hold, and only started recovering
    # (rollout 25+) after the policy had already committed to always-FLAT.
    # 0.005 was never validated as sufficient on its own -- 0.05 is the value
    # the earlier fixed-coefficient run used, and even that still collapsed
    # eventually (see the docstring above). Letting the controller go 10x
    # below a floor already known to be too weak served no purpose. Combined
    # with entropy_coef_warmup_rollouts below.
    entropy_coef_min: float = 0.05
    entropy_coef_max: float = 2.0

    # Rollouts during which the controller does not move the coefficient at
    # all -- it stays at entropy_coef_discrete's init value regardless of
    # measured entropy. Cold-start entropy near ln(3) is not a signal the
    # controller should react to (see entropy_coef_min's comment above); this
    # gives the policy room to specialize before the dual-ascent step starts
    # trusting what it measures. Redundant with entropy_coef_min in the
    # shrink direction (the floor now equals the init value, so a downward
    # step during warmup would just get clamped back anyway), but it also
    # blocks spurious upward moves from early noise and keeps working if
    # entropy_coef_min is ever retuned below the init value again.
    entropy_coef_warmup_rollouts: int = 15

    # Safety net behind the controller. If the discrete head sits below
    # collapse_entropy_threshold AND the streams place essentially no trades
    # for collapse_patience_rollouts consecutive rollouts, stop the run rather
    # than burn the remainder. A converged do-nothing policy is a result, but
    # it does not need another 65 rollouts of confirmation.
    # Set collapse_patience_rollouts to 0 to disable.
    collapse_patience_rollouts: int = 20
    collapse_entropy_threshold: float = 0.02
    collapse_trades_threshold: int = 10

    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 3                   # full-batch epochs per rollout -- see ppo_hybrid.py docstring
                                            # re: time-axis shuffling being unsafe for a stateful LSTM.
                                            # Lowered 4 -> 3: with no KL early-stop, 4 full-batch epochs
                                            # per rollout overfits the batch and accelerates entropy
                                            # collapse; 3 still learns the batch without overshooting.
    learning_rate: float = 1e-4           # lowered 3e-4 -> 1e-4 for stability: the hotter rate
                                            # overshot the policy into the flat regime on the last run.
    adam_eps: float = 1e-5


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "cuda"   # ppo_hybrid.py falls back to cpu if unavailable
    total_rollouts: int = 1000
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_rollouts: int = 25
    log_every_n_rollouts: int = 1

    # training/ppo_hybrid.py's ppo_update() -- mixed-precision (fp16)
    # autocast, scoped ONLY to the CNN/LSTM/cross-attention/fusion trunk
    # (the memory- and compute-heavy linear algebra), never to
    # model/hybrid_policy.py's Beta-distribution log-prob/entropy math or
    # the PPO loss itself -- those stay fp32 always, per hybrid_policy.py's
    # own "highest-risk file" warning about numerical fragility near the
    # (0,1) boundary (its _EPS=1e-6 guard is close to fp16's precision
    # floor). Off by default -- verify against an fp32 run on your actual
    # data before trusting it for a real training run; this was written
    # without access to a GPU to test it against.
    use_amp: bool = False

    # train.py's tick_callback -- write a "tick" metrics record every Nth
    # real env-step rather than every single one. Counters (global_tick,
    # trade tallies) still advance every real tick regardless -- this only
    # throttles disk writes. 1 = log every tick (max resolution, max I/O).
    #
    # Set to 1 in this revision: the two things that made higher-frequency
    # logging expensive earlier are both gone now -- fsync is already
    # skipped on every tick write (see MetricsWriter.log's docstring), and
    # MetricsReader.tail() no longer re-reads the whole file every poll
    # (chunked reverse read, cost independent of total file size). If you
    # run many parallel Kaggle sessions writing to a network-mounted
    # /kaggle/working and see real I/O contention, raise this back toward
    # 2-5 rather than assuming something else is wrong.
    tick_log_every_n_ticks: int = 1

    # MetricsWriter's tick-log rotation threshold, in bytes -- once
    # metrics_path (the tick-level log, not the whole logs/ dir) would
    # exceed this size, MetricsWriter rotates it (see its own docstring for
    # the exact rotation scheme) rather than letting a single run's file
    # grow unbounded. Was previously hardcoded inside MetricsWriter itself;
    # pulled out here so it's tunable per-run the same way tick_log_every_n_
    # ticks and dashboard_poll_interval_seconds are, without editing that
    # module directly.
    #
    # MEASURED, and the old comment here was wrong: this project's tick record
    # is ~8.2 KB/line at 100 streams, so tick_log_every_n_ticks=1 writes about
    # 2.1 MB per 256-step rollout. A 151-rollout run produces ~320 MB and a
    # 1000-rollout one ~2.1 GB -- the previous note claiming 100 MB "is
    # comfortably larger than a full 1000-rollout run" was off by ~20x. The
    # last 151-rollout run rotated, and because only the live segment was
    # pulled off Kaggle the post-mortem had tick data for rollouts 126-150
    # only: 28 fills out of 257,137 trades, i.e. none of the active phase.
    #
    # Sized below so current + backups span a full run: 64 MB x (1 + 6) =
    # 448 MB, which holds a 151-rollout run whole and is nothing against
    # /kaggle/working's quota. If you raise total_rollouts, raise
    # tick_backup_count with it -- or accept the loss deliberately rather
    # than discovering it during the next post-mortem.
    max_tick_log_bytes: int = 64 * 1024 * 1024    # 64 MB per segment

    # How many rotated tick-log segments to retain alongside the live one.
    # MetricsWriter's own default is 2; at the sizes above that keeps only the
    # tail of a run. ALL segments must be collected to reconstruct a run --
    # metrics.ticks.jsonl plus metrics.ticks.jsonl.1 .. .N, newest-first.
    # Note any log-clearing step must glob "metrics.ticks.jsonl*", not
    # "*.jsonl": the rotated names do not end in .jsonl, so a narrower glob
    # leaves the PREVIOUS run's segments in place for the next one to be
    # mixed with.
    tick_backup_count: int = 6

    # How often the SEPARATE notebook cell running
    # TrainingDashboard.run_polling_loop() (or a manual polling loop) re-reads
    # metrics_path and redraws. Lowered from 1.0 -> 0.5 alongside
    # tick_log_every_n_ticks=1 above -- at this pace the dashboard is polling
    # roughly as often as new tick data actually lands, which is what
    # "frequent updates" actually requires (polling faster than data
    # arrives just re-renders the same frame; the floor below this is Rich's
    # own render cost, not I/O).
    dashboard_poll_interval_seconds: float = 0.5

    # monitoring/dashboard.py -- see resolve_mode()'s precedence (explicit
    # --kaggle/--local CLI flag > this value > env-var auto-detection) and
    # TrainingDashboard.render_once()'s docstring.
    metrics_path: str = "logs/metrics.jsonl"
    display_mode: str = "auto"   # "auto" | "kaggle" | "local" -- see monitoring/dashboard.py's DisplayMode
    dashboard_refresh_every_n_steps: int = 10

    # Best-checkpoint tracking (train.py) -- in addition to the periodic
    # checkpoint_every_n_rollouts saves, train.py separately tracks an
    # EMA-smoothed rollout reward and overwrites checkpoint_best.pt whenever
    # it improves. Raw per-rollout reward (only rollout_length steps) is too
    # noisy to compare directly -- the EMA smooths that out before deciding
    # "best," so a single lucky rollout doesn't get crowned and overwrite a
    # genuinely better policy from a few rollouts earlier.
    best_metric_ema_alpha: float = 0.05          # smoothing factor: higher = reacts faster, noisier
    best_metric_warmup_rollouts: int = 10         # don't start comparing "best" until the EMA has settled


@dataclass
class EvalConfig:
    """eval/metrics.py + eval/backtest_report.py -- go/no-go criteria and annualization."""

    bars_per_year: int = 78 * 252   # 5-min bars, 6.5h US equity session, 252 trading days/year
                                      # -- keep in sync with eval/metrics.py's DEFAULT_BARS_PER_YEAR
    risk_free_rate_annual: float = 0.0
    max_drawdown_limit: float = 0.20
    require_beat_bh: bool = True
    use_risk_pipeline: bool = True   # False bypasses Kelly/RiskManager/KillSwitch -- debugging only, see
                                       # run_backtest()'s docstring on why this should never decide go/no-go


@dataclass
class TrainingConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)