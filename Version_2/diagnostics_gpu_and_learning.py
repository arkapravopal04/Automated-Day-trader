"""
diagnostics_gpu_and_learning.py

Standalone sanity check, decoupled from real market data and the
VecTradingEnv/dataset pipeline entirely -- it builds a synthetic
RolloutBuffer directly and runs it through training/ppo_hybrid.py's
ppo_update() to answer two narrow, mechanical questions:

    1. Is every relevant tensor actually on the GPU during a PPO update?
       (explicit device assertions on the model's parameters and every
       field of the rollout buffer -- not just torch.cuda.is_available(),
       which tells you nothing about whether THIS run is actually using it)
    2. Does gradient flow reach the parameters, and does the optimizer
       visibly use it? An artificial target: the synthetic buffer's
       advantage is deliberately correlated with which direction was
       "taken" (LONG=high advantage, SHORT=low), so a working PPO update
       run repeatedly on the SAME fixed buffer should show policy_loss
       trending down and a consistently nonzero grad_norm.

WHAT THIS DOES NOT ANSWER: "is the trading strategy any good." That is a
fundamentally different, much harder question that needs real market data
and belongs to eval/backtest_report.py's go/no-go pipeline. This script
only answers "is the training machinery itself doing what it claims" --
device placement and basic gradient flow -- which is necessary but nowhere
near sufficient for the former. A model can pass every check here and
still learn a useless or actively bad trading policy; those are different
failure modes and need different tools to catch.

Run this ON KAGGLE (or wherever the real GPU is):
    python diagnostics_gpu_and_learning.py

Not run or verified against a real GPU as part of writing it -- there is
no CUDA device available in the environment this was authored in. Report
back exactly what it prints (especially any AssertionError, "[FAIL]", or
"[WARN]" line) before trusting a real training run's results.
"""

import copy
import sys

import torch

from training.config import TrainingConfig
from training.ppo_hybrid import HybridActorCritic, RolloutBuffer, ppo_update
from model.dual_critic import DualCriticHead


def check_device_placement(actor_critic: torch.nn.Module, device: torch.device) -> None:
    bad = [name for name, p in actor_critic.named_parameters() if p.device != device]
    if bad:
        raise AssertionError(f"Parameters NOT on {device}: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    n_params = sum(1 for _ in actor_critic.parameters())
    print(f"[ok] all {n_params} parameter tensors are on {device}")


def build_synthetic_buffer(
    n_features: int, T: int, n_envs: int, device: torch.device, cfg: TrainingConfig, actor_critic: HybridActorCritic
) -> RolloutBuffer:
    """
    Fabricates a plausible-shaped rollout with a KNOWN advantage/action
    correlation, entirely bypassing VecTradingEnv and real data. Real
    training will not have this artificial correlation baked in -- this is
    purely a "can the update machinery learn ANYTHING at all" probe.
    """
    window = cfg.env.window_size
    obs = torch.randn(T, n_envs, window, n_features, device=device)

    hidden = actor_critic.init_hidden(n_envs, device)
    direction_idx = torch.randint(0, 3, (T, n_envs), device=device)
    size = torch.rand(T, n_envs, device=device).clamp(1e-3, 1 - 1e-3)
    limit_offset = torch.rand(T, n_envs, device=device).clamp(1e-3, 1 - 1e-3)

    with torch.no_grad():
        log_prob_old = torch.zeros(T, n_envs, device=device)
        value_old = torch.zeros(T, n_envs, device=device)
        for t in range(T):
            trunk, hidden = actor_critic.forward_features(obs[t], hidden)
            lp, _ent = actor_critic.policy_head.evaluate_actions(trunk, direction_idx[t], size[t], limit_offset[t])
            log_prob_old[t] = lp
            values = actor_critic.critic_head(trunk)
            value_old[t] = DualCriticHead.select(values, torch.zeros(n_envs, device=device))

    position_before = torch.zeros(T, n_envs, device=device)
    filled_qty = torch.where(
        direction_idx != 1, torch.ones(T, n_envs, device=device), torch.zeros(T, n_envs, device=device)
    )
    done = torch.zeros(T, n_envs, dtype=torch.bool, device=device)

    # KNOWN, artificial signal: advantage is high wherever direction_idx ==
    # LONG_IDX (2), low wherever SHORT_IDX (0), flat on FLAT_IDX (1).
    advantages = torch.where(
        direction_idx == 2,
        torch.ones(T, n_envs, device=device),
        torch.where(direction_idx == 0, -torch.ones(T, n_envs, device=device), torch.zeros(T, n_envs, device=device)),
    )
    returns = value_old + advantages

    buffer = RolloutBuffer(
        obs=obs,
        direction_idx=direction_idx,
        size=size,
        limit_offset=limit_offset,
        log_prob_old=log_prob_old,
        value_old=value_old,
        position_before=position_before,
        filled_qty=filled_qty,
        reward=advantages.clone(),
        done=done,
        initial_hidden=(hidden[0].clone(), hidden[1].clone()),
    )
    buffer.advantages = advantages
    buffer.returns = returns
    return buffer


def check_buffer_device(buffer: RolloutBuffer, device: torch.device) -> None:
    for name, value in vars(buffer).items():
        if isinstance(value, torch.Tensor) and value.device != device:
            raise AssertionError(f"buffer.{name} is on {value.device}, expected {device}")
        if isinstance(value, tuple):  # initial_hidden = (h, c)
            for i, part in enumerate(value):
                if isinstance(part, torch.Tensor) and part.device != device:
                    raise AssertionError(f"buffer.{name}[{i}] is on {part.device}, expected {device}")
    print(f"[ok] every RolloutBuffer tensor is on {device}")


def main() -> None:
    if not torch.cuda.is_available():
        print(
            "[FAIL] torch.cuda.is_available() is False -- nothing below can test GPU "
            "placement. Check your Kaggle accelerator setting (Settings -> Accelerator -> "
            "GPU T4 x2)."
        )
        sys.exit(1)

    device = torch.device("cuda:0")
    cfg = TrainingConfig()
    n_features = 8  # synthetic -- doesn't need to match your real preprocess.py feature count
    n_envs = 10
    T = 32  # short, just enough to see a trend across a handful of updates

    print(f"torch: {torch.__version__}   cuda: {torch.version.cuda}   device: {torch.cuda.get_device_name(device)}\n")

    actor_critic = HybridActorCritic(n_features=n_features, cfg=cfg).to(device)
    check_device_placement(actor_critic, device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)
    buffer = build_synthetic_buffer(n_features, T, n_envs, device, cfg, actor_critic)
    check_buffer_device(buffer, device)

    print(
        "\nRunning 20 ppo_update() calls on the SAME synthetic buffer -- policy_loss should "
        "trend down and grad_norm should be consistently nonzero if gradients are actually "
        "flowing and the optimizer is using them:\n"
    )
    policy_losses = []
    grad_norms = []
    for i in range(20):
        torch.cuda.reset_peak_memory_stats(device)
        stats = ppo_update(actor_critic, optimizer, buffer, cfg)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        policy_losses.append(stats["policy_loss"])
        grad_norms.append(stats["grad_norm"])
        print(
            f"  update {i:2d}: policy_loss={stats['policy_loss']:+.4f}  "
            f"grad_norm={stats['grad_norm']:.4f}  approx_kl={stats['approx_kl']:+.4f}  "
            f"peak_mem={peak_mb:.1f}MB"
        )

    if all(g == 0.0 for g in grad_norms):
        print(
            "\n[FAIL] grad_norm was exactly 0.0 on every update -- gradients are NOT "
            "flowing. Check for an accidental torch.no_grad() wrapping ppo_update, or a "
            "detached tensor somewhere in the forward path."
        )
        sys.exit(1)

    improved = policy_losses[-1] < policy_losses[0]
    print(f"\npolicy_loss[0]  = {policy_losses[0]:+.4f}")
    print(f"policy_loss[-1] = {policy_losses[-1]:+.4f}")
    if improved:
        print(
            "[ok] policy_loss decreased over 20 updates on a fixed buffer with a known "
            "advantage/action correlation -- gradients are flowing AND the optimizer is "
            "using them to reduce the loss it's actually being asked to minimize."
        )
    else:
        print(
            "[WARN] policy_loss did NOT decrease. This doesn't necessarily mean training is "
            "broken -- PPO's clipped objective on a tiny synthetic buffer can behave "
            "unintuitively -- but it's worth a second look. Re-run this script a couple "
            "times; if it's consistently flat or worse, something in the update path "
            "deserves closer inspection before trusting a real run's numbers."
        )

    print("\n--- gradient checkpointing memory impact (cfg.model.use_gradient_checkpointing) ---")
    cfg_checkpoint = copy.deepcopy(cfg)
    cfg_checkpoint.model.use_gradient_checkpointing = True

    torch.cuda.reset_peak_memory_stats(device)
    ppo_update(actor_critic, optimizer, buffer, cfg)
    peak_off = torch.cuda.max_memory_allocated(device) / 1e6

    torch.cuda.reset_peak_memory_stats(device)
    ppo_update(actor_critic, optimizer, buffer, cfg_checkpoint)
    peak_on = torch.cuda.max_memory_allocated(device) / 1e6

    print(f"  peak memory, checkpointing OFF: {peak_off:.1f} MB")
    print(f"  peak memory, checkpointing ON:  {peak_on:.1f} MB")
    if peak_off > 0:
        print(f"  reduction: {100 * (1 - peak_on / peak_off):.1f}%")
    print(
        "\n  (this synthetic buffer is tiny -- n_envs=10, T=32 -- so the absolute MB numbers "
        "won't look like your real training run. What matters here is the RELATIVE "
        "reduction, which should hold roughly proportionally at your real n_envs/T.)"
    )


if __name__ == "__main__":
    main()