"""Quick calibration: dataset shape, n_envs, and per-rollout wall time. (temp, not committed)"""
import sys, time
sys.path.append(".")
sys.path.append("env")
import torch
torch.set_num_threads(6)
from dataset import MultiTickerRolloutDataset
from training.config import TrainingConfig
from training.ppo_hybrid import HybridActorCritic, collect_rollout, compute_gae, ppo_update
from training.reward import DifferentialSharpeReward
from env.vec_trading_env import VecTradingEnv
from risk.kelly_sizing import KellySizer
from risk.kill_switch import KillSwitch
from risk.risk_manager import RiskLimits, RiskManager

t0 = time.time()
ds = MultiTickerRolloutDataset(window_size=60, split="train", device="cpu")
print(f"load: {time.time()-t0:.1f}s shape={tuple(ds.data_tensor.shape)} features={len(ds.feature_names)}")

cfg = TrainingConfig()
cfg.run.total_rollouts = 3
cfg.run.metrics_path = "calib.jsonl"
cfg.run.checkpoint_every_n_rollouts = 10**9
cfg.run.best_metric_warmup_rollouts = 10**9
cfg.run.tick_log_every_n_ticks = 256

env = VecTradingEnv(dataset=ds, initial_cash=cfg.env.initial_cash,
    max_position_frac=cfg.env.max_position_frac, tick_size=cfg.env.tick_size,
    spread_bps=cfg.env.spread_bps, impact_coef=cfg.env.impact_coef,
    max_participation=cfg.env.max_participation,
    commission_per_share=cfg.env.commission_per_share, commission_bps=cfg.env.commission_bps,
    min_commission=cfg.env.min_commission, platform_fee_per_trade=cfg.env.platform_fee_per_trade,
    r_step_scale=cfg.env.r_step_scale, hold_loser_penalty=cfg.env.hold_loser_penalty,
    enable_mirroring=cfg.env.enable_mirroring, mirror_prob=cfg.env.mirror_prob,
    overtrade_window=cfg.env.overtrade_window, overtrade_free_trades=cfg.env.overtrade_free_trades,
    overtrade_penalty_coef=cfg.reward.overtrade_penalty_coef,
    execution_price_column=cfg.env.execution_price_column,
    bias_window=cfg.env.bias_window, diversity_bonus_coef=cfg.env.diversity_bonus_coef, device="cpu")
print("n_envs:", env.n_envs, "| tickers:", len(env.tickers))
ac = HybridActorCritic(n_features=len(ds.feature_names), cfg=cfg).to("cpu")
opt = torch.optim.Adam(ac.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)
ks = KellySizer(n_envs=env.n_envs, lookback_trades=cfg.risk.kelly_lookback_trades,
    min_trades_for_estimate=cfg.risk.kelly_min_trades_for_estimate,
    kelly_multiplier=cfg.risk.kelly_multiplier, kelly_cap=cfg.risk.kelly_cap,
    default_fraction=cfg.risk.kelly_default_fraction, device="cpu")
rm = RiskManager(RiskLimits(max_position_frac=cfg.risk.max_position_frac,
    max_gross_exposure_frac=cfg.risk.max_gross_exposure_frac,
    max_ticker_concentration_frac=cfg.risk.max_ticker_concentration_frac,
    max_order_notional_frac=cfg.risk.max_order_notional_frac,
    drawdown_halt_frac=cfg.risk.drawdown_halt_frac), device="cpu")
ksw = KillSwitch(n_envs=env.n_envs, daily_loss_limit_frac=cfg.risk.daily_loss_limit_frac,
    broker_error_streak_limit=cfg.risk.broker_error_streak_limit,
    state_mismatch_tolerance=cfg.risk.state_mismatch_tolerance, device="cpu")
rs = DifferentialSharpeReward(n_envs=env.n_envs, eta=cfg.reward.dsr_eta, eps=cfg.reward.dsr_eps,
    warmup_steps=cfg.reward.dsr_warmup_steps, clip=cfg.reward.dsr_clip, device="cpu")

t0 = time.time()
obs = env.reset(); hidden = ac.init_hidden(env.n_envs, "cpu")
for i in range(3):
    buf, obs, fv, hidden = collect_rollout(env, ac, ks, rm, ksw, rs, obs, hidden, cfg)
    compute_gae(buf, fv, cfg.ppo.gamma, cfg.ppo.gae_lambda)
    st = ppo_update(ac, opt, buf, cfg)
dt = time.time() - t0
print(f"3 rollouts in {dt:.1f}s -> {dt/3:.1f}s/rollout")
