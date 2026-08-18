"""
train_ddp.py

True dual-GPU data-parallel training for Kaggle's 2x T4 setup.

WHY NOT nn.DataParallel / a naive batch-split:
model/cross_attention.py's CrossAssetAttention needs every ticker in the
SAME forward call to compute cross-asset correlations. nn.DataParallel (or
any scheme that shards n_envs=14 tickers 7/7 across two GPUs) would run
cross-attention over half the ticker universe on each GPU -- there's no
shape-level error to catch this, it would just silently train a model that
never sees the correlations it was designed around.

WHAT THIS DOES INSTEAD -- the correct shape of parallelism for this
architecture:
    - each GPU gets its OWN full replica of HybridActorCritic AND its own
      full VecTradingEnv (all 14 tickers together, every step, so
      cross-attention always operates on the complete ticker set)
    - the two replicas run independent rollouts in parallel -- different
      seeds -> different stochastic act() samples AND a different
      vec_trading_env.py mirror_mask draw on every reset(), so this is
      genuinely 2x the experience per update, not a duplicated copy of one
      trajectory
    - DistributedDataParallel wraps the model so ppo_update()'s
      loss.backward() automatically all-reduces (averages) gradients
      across both GPUs before optimizer.step() -- this is the actual
      "average gradients before each optimizer step" mechanism; DDP does
      it via backward hooks, nothing here hand-rolls an all-reduce.

Net effect: "14 tickers on 1 GPU" becomes "14 tickers x 2 independent
rollout streams, gradient-averaged, on 2 GPUs" -- more experience per PPO
update, not just faster wall-clock on the same experience.

CRITICAL GOTCHA THIS FILE EXISTS TO AVOID: training/ppo_hybrid.py's
collect_rollout() and ppo_update() call actor_critic.cnn(...) / .lstm(...)
/ .policy_head(...) / .critic_head(...) directly -- the SUBMODULES, not
actor_critic(...) itself. DDP's gradient-averaging hooks are registered on
the wrapped module's forward() call. Call the submodules directly through a
DDP wrapper and you bypass those hooks entirely: training "works" (no
error, loss goes down) but each GPU's replica silently trains on its own
gradients only -- never averaged, never in sync, quietly diverging. That
failure mode produces no crash and no obviously-wrong number, which is
exactly why it's dangerous. _ddp_ppo_update() below is ppo_update() rewired
so the actual backward() call happens with DDP's hooks attached; verify this
against ppo_hybrid.py's ppo_update() line-by-line before changing either.

Kaggle-specific: Kaggle notebooks are a single process, so this launches
its own subprocesses via torch.multiprocessing.spawn rather than assuming a
`torchrun` CLI launch is available. Call launch_ddp_training() from a
notebook cell; it blocks until training finishes.

Only rank 0 writes checkpoints and metrics. Two processes both calling
MetricsWriter.log() against the same metrics_path would interleave JSONL
lines mid-write (monitoring/dashboard.py's no-locking guarantee assumes
exactly one writer) -- rank 0 is the only one that ever opens a
MetricsWriter here.
"""

import argparse
import os
import sys
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from dataset import MultiTickerRolloutDataset  # noqa: E402
from paths import is_kaggle  # noqa: E402
from env.vec_trading_env import VecTradingEnv  # noqa: E402

from training.config import TrainingConfig  # noqa: E402
from training.ppo_hybrid import HybridActorCritic, collect_rollout, compute_gae  # noqa: E402
from training.reward import DifferentialSharpeReward  # noqa: E402

from model.dual_critic import DualCriticHead  # noqa: E402

from monitoring.dashboard import MetricsWriter  # noqa: E402

from train import (  # noqa: E402 -- reuse, don't duplicate
    build_risk_pipeline,
    kelly_metrics_fields,
    _TickState,
    make_tick_callback,
    resolve_resume_path,
    _clean_checkpoint_dir,
    _REWARD_SANITY_BOUND,
)


def ddp_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")  # change if this port is already bound in your Kaggle session
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup() -> None:
    dist.destroy_process_group()


def _build_env(cfg: TrainingConfig, device: torch.device) -> VecTradingEnv:
    train_dataset = MultiTickerRolloutDataset(window_size=cfg.env.window_size, split="train", device=str(device))
    env = VecTradingEnv(
        dataset=train_dataset,
        initial_cash=cfg.env.initial_cash,
        max_position_frac=cfg.env.max_position_frac,
        tick_size=cfg.env.tick_size,
        spread_bps=cfg.env.spread_bps,
        impact_coef=cfg.env.impact_coef,
        max_participation=cfg.env.max_participation,
        commission_per_share=cfg.env.commission_per_share,
        commission_bps=cfg.env.commission_bps,
        min_commission=cfg.env.min_commission,
        platform_fee_per_trade=cfg.env.platform_fee_per_trade,
        r_step_scale=cfg.env.r_step_scale,
        hold_loser_penalty=cfg.env.hold_loser_penalty,
        enable_mirroring=cfg.env.enable_mirroring,
        mirror_prob=cfg.env.mirror_prob,
        overtrade_window=cfg.env.overtrade_window,
        overtrade_free_trades=cfg.env.overtrade_free_trades,
        overtrade_surcharge_bps=cfg.env.overtrade_surcharge_bps,
        bias_window=cfg.env.bias_window,
        diversity_bonus_coef=cfg.env.diversity_bonus_coef,
        device=str(device),
    )
    return env


def _worker(
    rank: int,
    world_size: int,
    total_rollouts: Optional[int],
    resume: Optional[str],
    fresh: bool,
    checkpoint_dir: Optional[str] = None,
    metrics_path: Optional[str] = None,
    checkpoint_every_n_rollouts: Optional[int] = None,
    log_every_n_rollouts: Optional[int] = None,
    tick_log_every_n_ticks: Optional[int] = None,
) -> None:
    _setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    cfg = TrainingConfig()
    if total_rollouts is not None:
        cfg.run.total_rollouts = total_rollouts
    if checkpoint_dir is not None:
        cfg.run.checkpoint_dir = checkpoint_dir
    if metrics_path is not None:
        cfg.run.metrics_path = metrics_path
    if checkpoint_every_n_rollouts is not None:
        cfg.run.checkpoint_every_n_rollouts = checkpoint_every_n_rollouts
    if log_every_n_rollouts is not None:
        cfg.run.log_every_n_rollouts = log_every_n_rollouts
    if tick_log_every_n_ticks is not None:
        cfg.run.tick_log_every_n_ticks = tick_log_every_n_ticks
    # Different seed per rank -> genuinely different rollouts (act() samples,
    # vec_trading_env.py's mirror_mask draw), not two copies of one trajectory.
    torch.manual_seed(cfg.run.seed + rank)

    on_kaggle = is_kaggle()
    if on_kaggle and cfg.run.checkpoint_dir == "checkpoints":
        cfg.run.checkpoint_dir = "/kaggle/working/checkpoints"
    # Same redirect as train.py -- see that file's matching comment. The
    # `== default` guard keeps an explicit metrics_path override intact
    # (e.g. hyperparam_sweep.py sets its own per-run metrics_path).
    if on_kaggle and cfg.run.metrics_path == "logs/metrics.jsonl":
        cfg.run.metrics_path = "/kaggle/working/logs/metrics.jsonl"
    if rank == 0:
        os.makedirs(cfg.run.checkpoint_dir, exist_ok=True)
        if fresh:
            # Explicit cold reset -- only rank 0 touches the filesystem.
            _clean_checkpoint_dir(cfg.run.checkpoint_dir)
    dist.barrier()  # rank > 0 must not race ahead of rank 0's mkdir

    # Explicit continue-vs-reset decision (--resume / --fresh / refuse-to-guess).
    # Deterministic on every rank (same dir, same rules) so the two GPUs
    # always agree on the starting weights.
    resume_path = resolve_resume_path(resume, cfg.run.checkpoint_dir, fresh)
    if resume_path is not None:
        if rank == 0:
            print(f"[train_ddp] resuming from {resume_path}")
    elif rank == 0:
        print("[train_ddp] fresh start (random weights)")

    env = _build_env(cfg, device)
    actor_critic = HybridActorCritic(n_features=len(env.feature_names), cfg=cfg).to(device)

    start_rollout = 0
    best_metric = float("-inf")
    ema_reward = None
    # Same _TickState / tick-callback machinery train.py uses -- imported
    # rather than duplicated, so the two entrypoints can't drift apart on
    # what a "tick" record looks like.
    state = _TickState(n_envs=env.n_envs)
    checkpoint = None
    if resume_path is not None:
        # Every rank loads the SAME checkpoint onto ITS OWN device. DDP
        # requires byte-identical starting weights on every replica; loading
        # before the DDP wrap (below) and letting DDP broadcast from rank 0
        # at construction time guarantees that regardless of any local
        # nondeterminism in torch.load across ranks.
        # weights_only=True -- see main.py's matching comment (checkpoints
        # are pickle files; arbitrary-object unpickling is RCE).
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        actor_critic.load_state_dict(checkpoint["actor_critic"])
        start_rollout = checkpoint["rollout_idx"] + 1
        best_metric = checkpoint.get("best_metric", float("-inf"))
        ema_reward = checkpoint.get("ema_reward", None)
        # AUTO-HEAL for the pre-fix reward explosion signature (see
        # train.py's matching comment) -- resets poisoned best/EMA baselines
        # so checkpoint_best.pt tracking works after resuming an old run.
        if best_metric > _REWARD_SANITY_BOUND:
            if rank == 0:
                print(f"[train_ddp] WARNING: loaded best_metric {best_metric:.4f} is the "
                      "pre-fix reward-explosion signature -- resetting best-metric tracking")
            best_metric = float("-inf")
        if ema_reward is not None and ema_reward > _REWARD_SANITY_BOUND:
            if rank == 0:
                print(f"[train_ddp] WARNING: loaded ema_reward {ema_reward:.4f} is the "
                      "pre-fix reward-explosion signature -- restarting the EMA from scratch")
            ema_reward = None
        state.global_tick = checkpoint.get("global_tick", 0)
        state.episode_idx = checkpoint.get("episode_idx", 0)
        saved_per_ticker = checkpoint.get("total_trades_per_ticker", None)
        if isinstance(saved_per_ticker, list) and len(saved_per_ticker) == env.n_envs:
            state.total_trades_per_ticker = saved_per_ticker
        if rank == 0:
            print(f"[train_ddp] resumed from {resume_path} at rollout {start_rollout} "
                  f"(global_tick={state.global_tick}, episode={state.episode_idx}, "
                  f"best_metric so far: {best_metric if best_metric != float('-inf') else 'none yet'})")

    ddp_model = DDP(actor_critic, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.run.use_amp)
    if checkpoint is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    kelly_sizer, risk_manager, kill_switch = build_risk_pipeline(cfg, env.n_envs, device)
    reward_shaper = DifferentialSharpeReward(
        n_envs=env.n_envs, eta=cfg.reward.dsr_eta, eps=cfg.reward.dsr_eps,
        warmup_steps=cfg.reward.dsr_warmup_steps, clip=cfg.reward.dsr_clip, device=str(device),
    )

    # Only rank 0 writes metrics -- two ranks both calling MetricsWriter.log()
    # against the same file would interleave JSONL lines. tick_callback is
    # None on every other rank, and collect_rollout() treats None as "don't
    # bother" (see its own docstring), so non-zero ranks pay zero tick-logging
    # overhead, not just a suppressed write.
    # Only rank 0 owns the metrics writer. The writer itself routes
    # record_type="tick" records to its bounded tick log and keeps rollout
    # records in cfg.run.metrics_path, so this DDP entrypoint does not need
    # separate tick-path or rotation arguments. tick_max_bytes wires up
    # cfg.run.max_tick_log_bytes -- see train.py's matching comment.
    metrics_writer = (
        MetricsWriter(cfg.run.metrics_path, tick_max_bytes=cfg.run.max_tick_log_bytes)
        if rank == 0 else None
    )
    tick_callback = (
        make_tick_callback(env, metrics_writer, state, log_every_n_ticks=cfg.run.tick_log_every_n_ticks)
        if rank == 0 else None
    )

    obs = env.reset()
    hidden = ddp_model.module.init_hidden(env.n_envs, device)
    kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001

    for rollout_idx in range(start_rollout, cfg.run.total_rollouts):
        state.rollout_idx = rollout_idx
        if rank == 0:
            state.start_new_rollout()

        # collect_rollout() takes a HybridActorCritic and calls its
        # submodules directly (no gradients needed for rollout collection,
        # see ppo_hybrid.py's own no_grad note) -- passing ddp_model.module
        # here is correct and does NOT skip any DDP hook, since no
        # backward() happens during collection.
        buffer, obs, final_value, hidden = collect_rollout(
            env, ddp_model.module, kelly_sizer, risk_manager, kill_switch, reward_shaper, obs, hidden, cfg,
            tick_callback=tick_callback,
        )
        # Snapshot BEFORE the episode-boundary kelly_sizer.reset() below can
        # run -- see train.py's matching comment. Only rank 0 logs, so only
        # rank 0 needs the snapshot.
        kelly_diag = kelly_sizer.diagnostics() if rank == 0 else None
        compute_gae(buffer, final_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        stats = _ddp_ppo_update(ddp_model, optimizer, buffer, cfg, scaler=scaler)

        if buffer.done.any():
            if rank == 0:
                state.episode_idx += 1
            obs = env.reset()
            hidden = ddp_model.module.init_hidden(env.n_envs, device)
            kelly_sizer.reset()
            reward_shaper.reset()
            kill_switch.reset()  # training-only departure from live semantics -- see train.py's matching comment
            kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001
        elif (rollout_idx + 1) % cfg.risk.kill_switch_reset_every_n_rollouts == 0:
            # Episode boundaries are far too rare on their own -- see
            # training/config.py's RiskConfig.kill_switch_reset_every_n_rollouts.
            kill_switch.reset()
            kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001
            # See PortfolioState.reset_peak_equity()'s docstring -- same
            # "stuck for the rest of a huge episode" failure mode, in
            # risk_manager.py's drawdown_halt_frac instead of KillSwitch.
            env.portfolio.reset_peak_equity(env._current_prices().unsqueeze(1))  # noqa: SLF001

        rollout_reward_mean = buffer.reward.mean().item()
        alpha = cfg.run.best_metric_ema_alpha
        ema_reward = rollout_reward_mean if ema_reward is None else (
            alpha * rollout_reward_mean + (1 - alpha) * ema_reward
        )

        if rank == 0 and rollout_idx % cfg.run.log_every_n_rollouts == 0:
            current_prices_unsq = env._current_prices().unsqueeze(1)  # noqa: SLF001
            equity_per_ticker = env.portfolio.equity(current_prices_unsq)
            unrealized_per_ticker = env.portfolio.unrealized_pnl(current_prices_unsq)
            peak = env.portfolio.peak_equity.clamp(min=1e-6)
            drawdown_per_ticker = (peak - equity_per_ticker).clamp(min=0.0) / peak
            net_worth = float(equity_per_ticker.sum().item())

            metrics_writer.log(
                step=state.global_tick,
                rollout=rollout_idx,
                episode=state.episode_idx,
                record_type="rollout",
                reward=rollout_reward_mean,
                reward_ema=ema_reward,
                sharpe=None,
                drawdown=float(drawdown_per_ticker.mean().item()),
                drawdown_per_ticker=drawdown_per_ticker.tolist(),
                net_worth=net_worth,
                net_worth_per_ticker=equity_per_ticker.tolist(),
                unrealized_pnl=unrealized_per_ticker.tolist(),
                trades_this_rollout=int(sum(state.trades_this_rollout_per_ticker)),
                trades_per_ticker_this_rollout=list(state.trades_this_rollout_per_ticker),
                total_trades=int(sum(state.total_trades_per_ticker)),
                total_trades_per_ticker=list(state.total_trades_per_ticker),
                tickers=env.tickers,
                position=env.portfolio.positions[:, 0].tolist(),
                world_size=world_size,  # tag records so you can tell DDP runs apart from single-GPU ones in the log
                **kelly_metrics_fields(kelly_diag),
                **stats,
            )

        if rank == 0:
            checkpoint_state = {
                "actor_critic": ddp_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rollout_idx": rollout_idx,
                "best_metric": best_metric,
                "ema_reward": ema_reward,
                "global_tick": state.global_tick,
                "episode_idx": state.episode_idx,
                "total_trades_per_ticker": state.total_trades_per_ticker,
                "scaler": scaler.state_dict(),
            }
            if rollout_idx % cfg.run.checkpoint_every_n_rollouts == 0:
                path = os.path.join(cfg.run.checkpoint_dir, f"checkpoint_{rollout_idx}.pt")
                torch.save(checkpoint_state, path)
                print(f"[train_ddp] rollout {rollout_idx}: checkpoint saved to {path}")
            if rollout_idx >= cfg.run.best_metric_warmup_rollouts and ema_reward > best_metric:
                best_metric = ema_reward
                checkpoint_state["best_metric"] = best_metric
                best_path = os.path.join(cfg.run.checkpoint_dir, "checkpoint_best.pt")
                torch.save(checkpoint_state, best_path)
                print(f"[train_ddp] rollout {rollout_idx}: new best (EMA reward {best_metric:.6f}) -> {best_path}")

    try:
        if metrics_writer is not None:
            metrics_writer.close()
    finally:
        _cleanup()




def _ddp_ppo_update(
    ddp_model: DDP,
    optimizer: torch.optim.Optimizer,
    buffer,
    cfg: TrainingConfig,
    scaler: "Optional[torch.cuda.amp.GradScaler]" = None,
) -> dict:
    """
    Same math as training/ppo_hybrid.py's ppo_update(). The DDP-specific
    difference is that submodules are accessed through `ddp_model.module`,
    while the resulting backward pass remains connected to parameters watched
    by DDP so gradient synchronization still occurs.
    submodule calls), but the resulting `loss.backward()` call below is what
    actually triggers DDP's cross-GPU gradient all-reduce, because the
    tensors in that graph were produced by parameters DDP is watching.
    Includes the same cfg.run.use_amp / cfg.model.use_gradient_checkpointing
    handling as ppo_update() -- see that function's docstring for the full
    rationale (autocast scoped to the CNN/LSTM/attention trunk only, never
    the Beta-distribution policy math). If this drifts from
    ppo_hybrid.py's ppo_update() in anything other than the DDP-specific
    comments, that's a bug -- diff them.
    """
    actor_critic = ddp_model.module
    T, n_envs, window, n_features = buffer.obs.shape
    p = cfg.ppo
    use_amp = bool(getattr(cfg.run, "use_amp", False)) and torch.cuda.is_available()
    use_checkpoint = bool(getattr(cfg.model, "use_gradient_checkpointing", False))

    flat_direction_idx = buffer.direction_idx.reshape(T * n_envs)
    flat_size = buffer.size.reshape(T * n_envs)
    flat_limit_offset = buffer.limit_offset.reshape(T * n_envs)
    flat_log_prob_old = buffer.log_prob_old.reshape(T * n_envs)
    flat_value_old = buffer.value_old.reshape(T * n_envs)
    flat_position_before = buffer.position_before.reshape(T * n_envs)
    flat_advantages = buffer.advantages.reshape(T * n_envs)
    flat_returns = buffer.returns.reshape(T * n_envs)

    adv_mean, adv_std = flat_advantages.mean(), flat_advantages.std()
    flat_advantages_norm = (flat_advantages - adv_mean) / adv_std.clamp(min=1e-8)

    last_stats: dict = {}
    for _ in range(p.ppo_epochs):
        hidden = (buffer.initial_hidden[0].clone(), buffer.initial_hidden[1].clone())

        with torch.cuda.amp.autocast(enabled=use_amp):
            flat_obs = buffer.obs.reshape(T * n_envs, window, n_features)
            if use_checkpoint and torch.is_grad_enabled():
                cnn_seq_all = torch.utils.checkpoint.checkpoint(
                    actor_critic.cnn, flat_obs, use_reentrant=False
                )
            else:
                cnn_seq_all = actor_critic.cnn(flat_obs)
            cnn_seq_all = cnn_seq_all.reshape(T, n_envs, window, -1)

            trunks = []
            for t in range(T):  # MUST stay sequential -- see ppo_hybrid.py's module docstring
                lstm_seq_t, hidden = actor_critic.lstm(cnn_seq_all[t], hidden)
                lstm_last_t = lstm_seq_t[:, -1, :]
                cnn_last_t = cnn_seq_all[t][:, -1, :]
                attn_out_t = actor_critic.cross_attn(lstm_last_t)
                trunks.append(actor_critic.fusion(cnn_last_t, attn_out_t))
            trunk_all = torch.stack(trunks, dim=0)
            flat_trunk = trunk_all.reshape(T * n_envs, -1)

        # Deliberately fp32 from here down -- see this function's and
        # ppo_hybrid.py's ppo_update()'s docstrings.
        flat_trunk = flat_trunk.float()

        log_prob_new, discrete_entropy, continuous_entropy = actor_critic.policy_head.evaluate_actions_split(
            flat_trunk, flat_direction_idx, flat_size, flat_limit_offset
        )
        values_new = actor_critic.critic_head(flat_trunk)
        value_new_selected = DualCriticHead.select(values_new, flat_position_before)

        ratio = torch.exp(log_prob_new - flat_log_prob_old)
        surr1 = ratio * flat_advantages_norm
        surr2 = torch.clamp(ratio, 1.0 - p.clip_range, 1.0 + p.clip_range) * flat_advantages_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        value_clipped = flat_value_old + torch.clamp(
            value_new_selected - flat_value_old, -p.value_clip_range, p.value_clip_range
        )
        value_loss_unclipped = (value_new_selected - flat_returns).pow(2)
        value_loss_clipped = (value_clipped - flat_returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        entropy_bonus = (
            p.entropy_coef_discrete * discrete_entropy + p.entropy_coef_continuous * continuous_entropy
        ).mean()
        loss = policy_loss + p.value_loss_coef * value_loss - entropy_bonus

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()  # DDP all-reduces gradients across both GPUs here, automatically
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), p.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()  # DDP all-reduces gradients across both GPUs here, automatically
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), p.max_grad_norm)
            optimizer.step()

        with torch.no_grad():
            approx_kl = (flat_log_prob_old - log_prob_new).mean().item()
            clip_frac = ((ratio - 1.0).abs() > p.clip_range).float().mean().item()

        last_stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_discrete": discrete_entropy.mean().item(),
            "entropy_continuous": continuous_entropy.mean().item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "grad_norm": float(grad_norm),
        }
    return last_stats


def launch_ddp_training(
    total_rollouts: Optional[int] = None,
    resume: Optional[str] = None,
    fresh: bool = False,
    checkpoint_dir: Optional[str] = None,
    metrics_path: Optional[str] = None,
    checkpoint_every_n_rollouts: Optional[int] = None,
    log_every_n_rollouts: Optional[int] = None,
    tick_log_every_n_ticks: Optional[int] = None,
) -> None:
    """
    Meant to be called from `python train_ddp.py ...` (a real subprocess),
    NOT imported and called in-process from a notebook kernel that has
    already touched CUDA -- mp.spawn() forks from whatever process calls
    it, and forking a process with an active CUDA context is exactly the
    "cannot re-initialize CUDA in forked subprocess" failure. Launch this
    file with `!python train_ddp.py ...` from the notebook shell instead of
    importing launch_ddp_training() into the kernel.

    Falls back to single-GPU train.main() if fewer than 2 GPUs are visible
    -- DDP with world_size=1 is pure overhead with nothing to parallelize,
    and this way the same command works unmodified on a single-GPU Kaggle
    instance, a local single-GPU box, or a dual-T4 session.
    """
    if not ddp_available():
        print("[train_ddp] fewer than 2 CUDA devices visible -- falling back to single-GPU train.py")
        import train
        argv = []
        if total_rollouts is not None:
            argv += ["--total-rollouts", str(total_rollouts)]
        if resume is not None:
            argv += ["--resume", resume]
        if fresh:
            argv += ["--fresh"]
        if tick_log_every_n_ticks is not None:
            argv += ["--tick-log-every-n-ticks", str(tick_log_every_n_ticks)]
        # train.py's own argparse doesn't expose checkpoint_dir/metrics_path
        # overrides -- if you need those on the single-GPU fallback path
        # too, set cfg.run.checkpoint_dir / cfg.run.metrics_path directly in
        # training/config.py, or extend train.py's parse_args() to match.
        train.main(argv)
        return

    world_size = torch.cuda.device_count()
    print(f"[train_ddp] launching {world_size} DDP workers (dual-T4 Kaggle path)")
    mp.spawn(
        _worker,
        args=(world_size, total_rollouts, resume, fresh, checkpoint_dir, metrics_path,
              checkpoint_every_n_rollouts, log_every_n_rollouts, tick_log_every_n_ticks),
        nprocs=world_size,
        join=True,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-GPU DDP entrypoint (Kaggle 2xT4).")
    parser.add_argument("--total-rollouts", type=int, default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from, or 'latest' (highest checkpoint_N.pt) / "
             "'best' (checkpoint_best.pt). Mutually exclusive with --fresh.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Force a cold start from random weights: deletes any existing *.pt in the "
             "checkpoint dir first. Mutually exclusive with --resume. If NEITHER --resume "
             "nor --fresh is given and checkpoints exist, training refuses to start rather "
             "than guess (pass --resume latest to continue, or --fresh to reset).",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--checkpoint-every-n-rollouts", type=int, default=None)
    parser.add_argument("--log-every-n-rollouts", type=int, default=None)
    parser.add_argument(
        "--tick-log-every-n-ticks", type=int, default=None,
        help="Override cfg.run.tick_log_every_n_ticks. 1 = log every env-step (max dashboard resolution).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    launch_ddp_training(
        total_rollouts=args.total_rollouts,
        resume=args.resume,
        fresh=args.fresh,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        checkpoint_every_n_rollouts=args.checkpoint_every_n_rollouts,
        log_every_n_rollouts=args.log_every_n_rollouts,
        tick_log_every_n_ticks=args.tick_log_every_n_ticks,
    )