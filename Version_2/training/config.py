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
from typing import List, Optional, Tuple

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

FRICTION_PRESETS = {
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
        impact_coef=0.1,
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

    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "SPY", "QQQ", "JPM", "XOM", "UNH", "HD", "V",
    ])  # 14 tickers -- placeholder universe, replace with your actual Alpaca list
    window_size: int = 60
    initial_cash: float = 10_000.0
    max_position_frac: float = 1.0
    tick_size: float = 0.01
    friction_level: str = "realistic"   # "low" | "realistic" | "high" -- see FRICTION_PRESETS above.
                                          # Informational once set via for_friction(); the fields below
                                          # are what VecTradingEnv actually reads.
    spread_bps: float = 1.0
    impact_coef: float = 0.1
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
    def for_friction(cls, level: str, **overrides) -> "EnvConfig":
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


@dataclass
class RiskConfig:
    # risk_manager.py -- RiskLimits
    max_position_frac: float = 0.25
    max_gross_exposure_frac: float = 1.0
    max_ticker_concentration_frac: float = 0.35
    max_order_notional_frac: float = 0.10
    drawdown_halt_frac: float = 0.20

    # kelly_sizing.py -- KellySizer
    kelly_lookback_trades: int = 30
    kelly_min_trades_for_estimate: int = 10
    kelly_multiplier: float = 0.25
    kelly_cap: float = 1.0
    kelly_default_fraction: float = 0.02

    # kill_switch.py -- KillSwitch
    daily_loss_limit_frac: float = 0.03
    broker_error_streak_limit: int = 3
    state_mismatch_tolerance: float = 1e-3

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


@dataclass
class PPOConfig:
    rollout_length: int = 256            # T env-steps per stream, per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coef_discrete: float = 0.02    # applied to the direction head's entropy independently
    entropy_coef_continuous: float = 0.01  # applied to the (FLAT-masked) size + limit_offset entropy independently
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4                   # full-batch epochs per rollout -- see ppo_hybrid.py docstring
                                            # re: time-axis shuffling being unsafe for a stateful LSTM
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "cuda"   # ppo_hybrid.py falls back to cpu if unavailable
    total_rollouts: int = 1000
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_rollouts: int = 25
    log_every_n_rollouts: int = 1

    # monitoring/dashboard.py -- see resolve_mode()'s precedence (explicit
    # --kaggle/--local CLI flag > this value > env-var auto-detection) and
    # TrainingDashboard.render_once()'s docstring.
    metrics_path: str = "logs/metrics.jsonl"
    display_mode: str = "auto"   # "auto" | "kaggle" | "local" -- see monitoring/dashboard.py's DisplayMode
    dashboard_refresh_every_n_steps: int = 10


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