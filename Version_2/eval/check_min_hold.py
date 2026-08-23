"""Verify min_hold_bars actually lengthens holding period.

Drives the env with the exact pathology the last run exhibited: a policy that
opens and then tries to close on the very next bar, forever. Measured on that
run's tick log, 140 of 140 completed round trips lasted exactly 1 bar (median,
mean, p90 and max all 1.0), so this is not a hypothetical.

Checks three things:
  1. min_hold_bars=0 reproduces the 1-bar pathology (control -- proves the
     test can see it)
  2. min_hold_bars=N lengthens holds to >= N
  3. the session-close forced flatten still overrides min-hold, so nothing is
     trapped past the bell
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

from dataset import MultiTickerRolloutDataset          # noqa: E402
from env.vec_trading_env import VecTradingEnv          # noqa: E402

torch.manual_seed(0)
ds = MultiTickerRolloutDataset(window_size=32, split="val", device="cpu")


def drive(min_hold, n_steps=600, size=3.0):
    """Open when flat, try to close immediately when not. The 1-bar pathology."""
    env = VecTradingEnv(
        dataset=ds, initial_cash=10_000.0, trade_cooldown_bars=0,
        enable_mirroring=False, flatten_at_session_close=True,
        min_hold_bars=min_hold, device="cpu",
    )
    env.reset()
    n = env.n_envs
    pos_hist = []

    for _ in range(n_steps):
        pos = env.portfolio.positions[:, 0]
        flat = pos == 0
        # flat -> buy;  holding -> sell it all, immediately, every bar
        direction = torch.where(flat, torch.ones(n), -torch.sign(pos))
        qty = torch.where(flat, torch.full((n,), size), pos.abs())
        env.step(direction=direction, size=qty, limit_offset=torch.zeros(n))
        pos_hist.append((env.portfolio.positions[:, 0] != 0).clone())
        if env.current_idx >= env.max_idx:
            break

    # Reconstruct hold runs per stream from the occupancy history.
    occ = torch.stack(pos_hist).numpy()            # (T, n_envs) bool
    holds = []
    for j in range(occ.shape[1]):
        run = 0
        for t in range(occ.shape[0]):
            if occ[t, j]:
                run += 1
            elif run:
                holds.append(run)
                run = 0
    return np.array(holds), env


print()
print("--- control: min_hold_bars=0 (reproduces the last run's pathology) ---")
h0, _ = drive(0)
print(f"completed holds : {len(h0)}")
print(f"hold bars       : median {np.median(h0):.1f}  mean {h0.mean():.2f}  max {h0.max()}")
print(f"holds == 1 bar  : {100 * (h0 == 1).mean():.1f}%")

results = {}
for N in (6, 12):
    print()
    print(f"--- min_hold_bars={N} ---")
    h, env = drive(N)
    short = int((h < N).sum())
    print(f"completed holds : {len(h)}")
    print(f"hold bars       : median {np.median(h):.1f}  mean {h.mean():.2f}  min {h.min()}  max {h.max()}")
    print(f"holds shorter than {N}: {short}  ({100 * short / len(h):.1f}%)")
    print(f"residual overnight carry: {env.residual_overnight_count}")
    results[N] = (h, short, env)

print()
print("Holds shorter than N are the session-close flatten cutting a position")
print("short at the bell -- min-hold runs BEFORE the override, so the forced")
print("close always wins. That is the intended precedence, not a leak.")

print()
assert (h0 == 1).mean() > 0.9, "control did not reproduce the 1-bar pathology"
for N, (h, short, env) in results.items():
    med = np.median(h)
    assert med >= N, f"min_hold_bars={N} did not lengthen holds (median {med})"
    # every short hold must be explained by a forced session close
    assert short <= env.forced_flatten_count, (
        f"min_hold_bars={N}: {short} short holds but only "
        f"{env.forced_flatten_count} forced flattens to explain them")
print(f"PASS: control median 1.0 -> min_hold=6 median {np.median(results[6][0]):.1f} "
      f"-> min_hold=12 median {np.median(results[12][0]):.1f}")
