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
    # 10_000 -> 100_000. The platform ticket fee is a FLAT $1 per trade, so
    # its cost as a fraction of notional is set entirely by order size, and
    # order size is capped by fractional_kelly * equity. Measured on a real
    # 50-rollout run: median fill $619, ticket fee therefore 16.2 bps/side,
    # against a measured GROSS edge of 3.29 bps of notional -- i.e. the
    # strategy earned +$1,463 gross and paid $7,023 in tickets to do it.
    # Break-even needs ~$3,040/order; at $10k equity even FULL Kelly
    # (raw 0.147 * $9,270) caps out at $1,363, so no setting of
    # kelly_multiplier / min_order_notional / max_position_frac can clear the
    # fee at that account size -- capital per stream is the binding
    # constraint, not a tuning knob. At $100k the same Kelly fraction gives
    # ~$4.3k orders (2.3 bps) and half-Kelly below gives ~$8.6k (1.2 bps),
    # comfortably under the edge. Observations are scale-free (the two
    # portfolio channels are equity-normalized fractions), so this does not
    # shift the input distribution the policy sees.
    initial_cash: float = 100_000.0
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
    r_step_scale: float = 0.5
    hold_loser_penalty: float = 0.0005
    enable_mirroring: bool = True
    mirror_prob: float = MIRROR_PROB
    overtrade_window: int = OVERTRADE_WINDOW
    overtrade_free_trades: int = OVERTRADE_FREE_TRADES
    overtrade_surcharge_bps: float = OVERTRADE_SURCHARGE_BPS
    bias_window: int = DIVERSITY_WINDOW
    diversity_bonus_coef: float = DIVERSITY_COEF

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
    max_order_notional_frac: float = 0.10
    drawdown_halt_frac: float = 0.20

    # Absolute $ floor on exposure-INCREASING orders -- see
    # risk_manager.py's _drop_dust_orders() for the full measured
    # rationale. Short version: a 50-rollout run produced a MEDIAN fill of
    # $0.20 (73.8% of fills under $1.00) while paying a flat $1.00/ticket
    # fee, which was 99.3% of that run's entire loss. Filtering at $100
    # drops ~86% of orders but keeps ~98.8% of traded notional. Applies in
    # training, backtest and live alike. 0.0 disables.
    # 100.0 -> 2_000.0, tracking the initial_cash 10k -> 100k change above
    # (see EnvConfig.initial_cash). This gate is an ABSOLUTE dollar floor, so
    # leaving it at $100 against a 10x larger account would make it a 0.1%-of-
    # equity threshold that gates nothing. $2,000 keeps it at the same ~2% of
    # stream equity it effectively enforced before, and doubles as a hard
    # backstop on fee drag: any order that passes pays at most 5 bps of
    # ticket, and the typical half-Kelly order (~$8.6k) pays ~1.2 bps.
    # Still exposure-INCREASING orders only -- closes and reduces are never
    # blocked by this, so it can't trap a position.
    min_order_notional: float = 2_000.0

    # kelly_sizing.py -- KellySizer
    kelly_lookback_trades: int = 30
    kelly_min_trades_for_estimate: int = 10
    # 0.25 -> 0.5 (quarter- to half-Kelly). Quarter-Kelly was leaving the
    # typical order at ~$4.3k even after the initial_cash change; half-Kelly
    # puts it near ~$8.6k, cutting ticket drag from 2.3 bps to ~1.2 bps
    # against the 3.29 bps measured gross edge. Half-Kelly is the standard
    # conservative operating point and still sits well inside
    # max_position_frac (0.25) and max_order_notional_frac (0.10).
    kelly_multiplier: float = 0.5
    kelly_cap: float = 1.0
    # 0.02 -> 0.05, forced by the min_order_notional change above. This is the
    # PRE-WARM sizing fraction, so it sets the order cap during exactly the
    # period when no stream has enough closed trades for a real edge estimate.
    # At 0.02 against initial_cash=100_000 the cap is 0.02 * 100k = $2,000 --
    # numerically EQUAL to min_order_notional, so every pre-warm order the
    # policy sized below its own Kelly cap would be dropped as dust, almost
    # nothing would fill, almost nothing would close, and the edge estimate
    # could never warm: a rebuild of the exact chicken-and-egg deadlock
    # kelly_min_fraction below was added to break. 0.05 gives a $5,000
    # pre-warm cap, a clear $2,000-$5,000 band of orders that can actually
    # pass the gate.
    kelly_default_fraction: float = 0.05

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
    # Raised 0.02 -> 0.05 alongside it -- see that field for why the old value
    # collided with min_order_notional at the new account size. If either is
    # retuned, move BOTH, and keep the floor strictly above
    # min_order_notional / initial_cash or the deadlock returns.
    kelly_min_fraction: float = 0.05

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
    max_limit_offset_ticks: float = 20.0


@dataclass
class RewardConfig:
    """training/reward.py -- DifferentialSharpeReward, plus how it blends with vec_trading_env.py's own shaped reward."""

    dsr_eta: float = 0.01
    dsr_eps: float = 1e-8
    dsr_warmup_steps: int = 2
    dsr_clip: float = 10.0
    sharpe_weight: float = 1.0   # weight on the Sharpe-shaped term
    raw_weight: float = 0.0      # weight on vec_trading_env.py's own StepResult.reward (vol-normalized step
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
    # module directly. 100 MB is comfortably larger than a full 1000-rollout
    # run at tick_log_every_n_ticks=1 produces on this project's tick record
    # schema -- lower it if you're disk-constrained (e.g. a small Kaggle
    # /kaggle/working quota), raise it if you extend total_rollouts well
    # past the default or log_every_n_ticks stays at 1 for a much longer run.
    max_tick_log_bytes: int = 100 * 1024 * 1024   # 100 MB

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