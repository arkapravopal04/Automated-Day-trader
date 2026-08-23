"""Verify trading_window gates ENTRY only, never exit.

Three properties, the third being the one that matters:

  1. with no window, entries happen across the whole session (control)
  2. with a window, every ENTRY falls inside it
  3. a position opened inside the window can still be CLOSED outside it

(3) is the property that makes the design safe. Gating exits as well would
trap inventory the moment the window ended and would fight both min_hold and
the session-close flatten -- so the test drives a policy that deliberately
opens late in the window and holds past its end.
"""
import os
import sys

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

WINDOW = (0, 12)          # open_hour


def drive(window, min_hold=0, n_steps=800, size=3.0, hold_bars=3):
    """Open when flat, close after `hold_bars`, repeat.

    Cycling rather than holding all day matters: a policy that never closes
    only ever re-enters right after the session-close flatten, so entries pile
    up on one bar and the control proves nothing about the rest of the day.
    """
    env = VecTradingEnv(
        dataset=ds, initial_cash=10_000.0, trade_cooldown_bars=0,
        enable_mirroring=False, flatten_at_session_close=True,
        min_hold_bars=min_hold, trading_window=window, device="cpu",
    )
    env.reset()
    n = env.n_envs
    entry_bars, exit_bars = [], []
    held = torch.zeros(n)

    for _ in range(n_steps):
        t = env.current_idx + env.window_size - 1
        bod = int(env._bar_of_day[t])
        before = env.portfolio.positions[:, 0].clone()
        flat = before == 0
        want_close = (~flat) & (held >= hold_bars)
        direction = torch.where(flat, torch.ones(n),
                                torch.where(want_close, -torch.sign(before),
                                            torch.zeros(n)))
        qty = torch.where(flat, torch.full((n,), size),
                          torch.where(want_close, before.abs(), torch.zeros(n)))
        env.step(direction=direction, size=qty, limit_offset=torch.zeros(n))
        held = torch.where(env.portfolio.positions[:, 0] == 0,
                           torch.zeros(n), held + 1.0)
        after = env.portfolio.positions[:, 0]
        opened = int(((before == 0) & (after != 0)).sum())
        closed = int(((before != 0) & (after == 0)).sum())
        if opened:
            entry_bars += [bod] * opened
        if closed:
            exit_bars += [bod] * closed
        if env.current_idx >= env.max_idx:
            break

    return np.array(entry_bars), np.array(exit_bars), env


print()
print("--- control: trading_window=None ---")
e0, x0, _ = drive(None)
print(f"entries      : {len(e0)}")
print(f"entry bars   : min {e0.min()}  max {e0.max()}  distinct {len(np.unique(e0))}")

print()
print(f"--- trading_window={WINDOW} (open_hour) ---")
e1, x1, env1 = drive(WINDOW)
lo, hi = WINDOW
outside = int(((e1 < lo) | (e1 >= hi)).sum())
print(f"entries      : {len(e1)}")
print(f"entry bars   : min {e1.min()}  max {e1.max()}  distinct {len(np.unique(e1))}")
print(f"entries OUTSIDE the window: {outside}")
print(f"exits        : {len(x1)}")
if len(x1):
    out_exits = int(((x1 < lo) | (x1 >= hi)).sum())
    print(f"exit bars    : min {x1.min()}  max {x1.max()}")
    print(f"exits OUTSIDE the window: {out_exits}  <- must be > 0, exits are never gated")

print()
print(f"--- window={WINDOW} + min_hold_bars=12 (entry late in window, exit past it) ---")
e2, x2, env2 = drive(WINDOW, min_hold=12)
outside2 = int(((e2 < lo) | (e2 >= hi)).sum())
out_exits2 = int(((x2 < lo) | (x2 >= hi)).sum()) if len(x2) else 0
print(f"entries {len(e2)}, outside window {outside2}")
print(f"exits   {len(x2)}, outside window {out_exits2}")
print(f"residual overnight carry: {env2.residual_overnight_count}")

print()
assert len(np.unique(e0)) > 20, (
    f"control only entered on {len(np.unique(e0))} distinct bars -- it is not "
    "exercising the session, so the window test proves little")
assert outside == 0, f"FAIL: {outside} entries outside the trading window"
assert outside2 == 0, f"FAIL: {outside2} entries outside the window with min_hold on"
assert len(x1) and out_exits > 0, (
    "no exits outside the window -- either nothing was held past it, or exits "
    "are being gated, which would trap inventory")
assert out_exits2 > 0, "min_hold + window: nothing exited past the window end"
print(f"PASS: {len(e1)} entries all inside {WINDOW}; {out_exits} exits outside it "
      f"(never gated); {out_exits2} outside with min_hold=12")
