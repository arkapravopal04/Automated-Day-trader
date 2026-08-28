"""Benchmark: where does sweep wall-time actually go? (temp file, not committed)"""
import sys, os, time, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

import torch
torch.set_num_threads(6)

from paths import PROCESSED_DIR
print("data exists:", os.path.exists(os.path.join(PROCESSED_DIR, "metadata.json")))

# 1) dataset load cost
from dataset import MultiTickerRolloutDataset
t0 = time.time()
ds = MultiTickerRolloutDataset(window_size=60, split="train", device="cpu")
t_load = time.time() - t0
print(f"DATASET LOAD: {t_load:.1f}s  shape={tuple(ds.data_tensor.shape)}  mem={ds.data_tensor.numel()*4/1e6:.0f}MB")

# 2) per-rollout cost, tick logging every tick (sweep default today)
import train
from training.config import TrainingConfig
from training.ppo_hybrid import HybridActorCritic, collect_rollout, compute_gae, ppo_update
from training.reward import DifferentialSharpeReward
from env.vec_trading_env import VecTradingEnv
from risk.kelly_sizing import KellySizer
from risk.kill_switch import KillSwitch
from risk.risk_manager import RiskLimits, RiskManager
from monitoring.dashboard import MetricsWriter

def build(cfg):
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
    return env, ac, opt, ks, rm, ksw, rs

def run_n(cfg, n_rollouts, label):
    env, ac, opt, ks, rm, ksw, rs = build(cfg)
    mw = MetricsWriter("bench_metrics.jsonl")
    t0 = time.time()
    obs = env.reset(); hidden = ac.init_hidden(env.n_envs, "cpu")
    for i in range(n_rollouts):
        buf, obs, fv, hidden = collect_rollout(env, ac, ks, rm, ksw, rs, obs, hidden, cfg)
        compute_gae(buf, fv, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        st = ppo_update(ac, opt, buf, cfg)
    dt = time.time() - t0
    mw.close()
    print(f"{label}: {n_rollouts} rollouts in {dt:.1f}s -> {dt/n_rollouts:.1f}s/rollout")
    return dt / n_rollouts

cfg = TrainingConfig()
cfg.run.total_rollouts = 3
cfg.run.metrics_path = "bench_metrics.jsonl"
cfg.run.checkpoint_every_n_rollouts = 10**9
cfg.run.best_metric_warmup_rollouts = 10**9
cfg.run.tick_log_every_n_ticks = 1
p1 = run_n(cfg, 3, "ticklog=1 (sweep default)")

cfg2 = TrainingConfig()
cfg2.run.total_rollouts = 3
cfg2.run.metrics_path = "bench_metrics.jsonl"
cfg2.run.checkpoint_every_n_rollouts = 10**9
cfg2.run.best_metric_warmup_rollouts = 10**9
cfg2.run.tick_log_every_n_ticks = 256  # once per rollout
p2 = run_n(cfg2, 3, "ticklog=256 + no ckpts")

print(f"\nEXTRA: 24 runs x 50 rollouts serial:")
print(f"  ticklog=1:      {24*50*p1/3600:.1f}h")
print(f"  ticklog=256:    {24*50*p2/3600:.1f}h")
for nw in (2, 4, 6, 8, 12):
    serial_est = 24 * 50 * p2
    per = serial_est / nw
    print(f"  {nw} workers x {max(1,12//nw)} thr: ~{per/3600:.1f}h + load overhead")
