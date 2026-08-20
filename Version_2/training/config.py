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
OVERTRADE_WINDOW, OVERTRADE_FREE_TRADES, OVERTRADE_SURCHARGE_BPS, and
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
    OVERTRADE_SURCHARGE_BPS,
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
# "friction" (spread, impact, participation cap, commissions, platform fee,
# overtrading surcharge) are touched; everything else in EnvConfig (tickers,
# window_size, mirroring, reward shaping, etc.) is untouched by this.
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
#                 cap, real commissions + platform fee, aggressive
#                 overtrading surcharge. If the policy is still profitable
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
        spread_bps=0.1,
        impact_coef=0.01,
        max_participation=0.5,
        commission_per_share=0.0,
        commission_bps=0.0,
        min_commission=0.0,
        platform_fee_per_trade=0.0,
        overtrade_surcharge_bps=0.0,
        overtrade_free_trades=9999,   # effectively disables the overtrading penalty
    ),
    "realistic": dict(
        spread_bps=1.0,
        # 0.1 -> 0.015: recalibrated to the standard square-root impact law,
        # impact ~= Y * sigma_daily * sqrt(Q/V) with Y ~ 0.5-1 and
        # sigma_daily ~ 1.5%, i.e. a coefficient near 0.015 -- NOT 0.1, which
        # is ~7x the literature value and was never calibrated against
        # anything. It matters most at SMALL order sizes, because sqrt(x) >> x
        # for small x: at the ~1.3-share orders this project actually trades
        # (participation ~2e-5 of a consolidated bar) the old 0.1 charged
        # 0.047% one-way -- roughly 5x the half-spread for an order that in
        # reality moves the market not at all. Combined with the separate
        # IEX-volume bug (see paths.py's VOLUME_SCALE) this produced a 0.55%
        # round-trip cost against a 0.056% median 5-minute move. At 0.015 a
        # micro order pays ~0.005% one-way (negligible, correct) while a
        # genuine 10%-of-bar order still pays ~0.24%.
        impact_coef=0.015,
        max_participation=0.1,
        commission_per_share=0.0,
        commission_bps=0.5,
        min_commission=0.0,
        platform_fee_per_trade=PLATFORM_FEE_PER_TRADE,
        overtrade_surcharge_bps=OVERTRADE_SURCHARGE_BPS,
        overtrade_free_trades=OVERTRADE_FREE_TRADES,
    ),
    "high": dict(
        spread_bps=5.0,
        impact_coef=0.35,
        max_participation=0.03,
        commission_per_share=0.005,
        commission_bps=1.5,
        min_commission=1.0,
        platform_fee_per_trade=1.0,
        overtrade_surcharge_bps=10.0,
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
    initial_cash: float = 10_000.0
    max_position_frac: float = 1.0
    tick_size: float = 0.01
    friction_level: str = "realistic"   # "low" | "realistic" | "high" -- see FRICTION_PRESETS above.
                                          # Informational once set via for_friction(); the fields below
                                          # are what VecTradingEnv actually reads.
    spread_bps: float = 1.0
    impact_coef: float = 0.015   # 0.1 -> 0.015, kept in sync with FRICTION_PRESETS["realistic"] --
                                  # see that entry for the square-root-law calibration rationale.
                                  # These dataclass defaults (not the preset) are what training
                                  # actually reads, since train.py builds TrainingConfig() directly
                                  # rather than going through EnvConfig.for_friction().
    max_participation: float = 0.1
    commission_per_share: float = 0.0
    commission_bps: float = 0.5
    min_commission: float = 0.0
    platform_fee_per_trade: float = PLATFORM_FEE_PER_TRADE
    # 0.5 -> 2.0. Scales the vol-normalized step-PnL term, which is the only
    # DIRECTLY PnL-aligned signal in the whole reward. Measured over 200 steps
    # x 100 streams it was the SMALLEST component of the env reward
    # (mean|x| 0.000255) -- smaller than diversity_bonus (0.00118) and
    # hold_loser_penalty (0.000404). Raising it 4x makes PnL the dominant term
    # inside StepResult.reward, which is the point of switching raw_weight on.
    r_step_scale: float = 2.0
    hold_loser_penalty: float = 0.0005
    enable_mirroring: bool = True
    mirror_prob: float = MIRROR_PROB
    overtrade_window: int = OVERTRADE_WINDOW
    overtrade_free_trades: int = OVERTRADE_FREE_TRADES
    overtrade_surcharge_bps: float = OVERTRADE_SURCHARGE_BPS
    bias_window: int = DIVERSITY_WINDOW
    diversity_bonus_coef: float = DIVERSITY_COEF
    # Bars a stream must wait after INCREASING exposure before it may increase
    # again (closes/reduces are never blocked). 12 bars = 1 hour of 5-min bars,
    # matching overtrade_window. 0 disables. See
    # VecTradingEnv._apply_trade_cooldown() for the measurements behind this.
    trade_cooldown_bars: int = 12

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

    # order sizing that hybrid_policy.py's rescale_* helpers need but which
    # isn't itself derived from equity (a hard ceiling the network's
    # normalized (0,1) output gets mapped onto BEFORE kelly/risk clip it
    # further down -- see ppo_hybrid.py's action pipeline)
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
    entropy_coef_min: float = 0.005
    entropy_coef_max: float = 2.0

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