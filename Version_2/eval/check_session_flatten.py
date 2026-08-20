"""End-to-end check that no position survives a session boundary.

Two policies, both hostile in that neither ever voluntarily closes:

  BOUNDED   buys a fixed size only when flat, so inventory stays at roughly
            what Kelly actually permits (~$740 notional). This is the spec:
            at real position sizes nothing may carry.

  UNBOUNDED accumulates every bar, reaching hundreds of shares by the close
            and exceeding max_participation * bar_volume. Reported, not
            asserted -- a forced close is an order, not a guarantee, and the
            honest answer there is "some residual is unavoidable", not
            "uncap the fill".

A control with the flag off proves the test can detect a carry at all, which
matters more than the pass.
"""
import os
import sys

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


def drive(flatten, size, bounded, close_bars=1, n_steps=900):
    env = VecTradingEnv(
        dataset=ds, initial_cash=10_000.0, trade_cooldown_bars=0,
        enable_mirroring=False, flatten_at_session_close=flatten,
        flatten_close_bars=close_bars, device="cpu",
    )
    env.reset()
    n = env.n_envs
    boundaries = starts = measured = untradeable = 0
    peak_pos = 0.0

    for _ in range(n_steps):
        t = env.current_idx + env.window_size - 1
        is_last = bool(env._session_last_only[t])
        vol = env.volumes[t].clone()
        pos = env.portfolio.positions[:, 0]
        if bounded:
            want = (pos == 0).to(pos.dtype)          # only open when flat
        else:
            want = torch.ones(n)
        env.step(direction=want, size=torch.full((n,), size) * want,
                 limit_offset=torch.zeros(n))
        peak_pos = max(peak_pos, float(env.portfolio.positions[:, 0].abs().max()))
        if is_last:
            boundaries += 1
            left = env.portfolio.positions[:, 0] != 0
            measured += int(left.sum().item())
            untradeable += int((left & (vol <= 0)).sum().item())
        if env.session_just_started:
            starts += 1
        if env.current_idx >= env.max_idx:
            break

    return dict(boundaries=boundaries, starts=starts, measured=measured,
                untradeable=untradeable,
                counted=env.residual_overnight_count,
                forced=env.forced_flatten_count, peak=peak_pos)


print()
print("--- BOUNDED inventory, flatten ON  (this is the spec) ---")
real = drive(True, size=3.0, bounded=True)
print(f"session closes traversed      : {real['boundaries']}")
print(f"session starts flagged        : {real['starts']}")
print(f"forced flattens counted       : {real['forced']:,}")
print(f"peak position held            : {real['peak']:.1f} shares")
print(f"residual carry (env counter)  : {real['counted']}")
print(f"residual carry (independent)  : {real['measured']}")
print(f"   of which the closing bar had ZERO volume: {real['untradeable']}")

print()
print("--- BOUNDED inventory, flatten OFF (control) ---")
ctrl = drive(False, size=3.0, bounded=True)
print(f"residual carry (independent)  : {ctrl['measured']:,}")

print()
print("--- UNBOUNDED accumulation (stress, reported not asserted) ---")
for cb in (1, 3, 6):
    u = drive(True, size=5.0, bounded=False, close_bars=cb)
    print(f"flatten_close_bars={cb}: peak {u['peak']:>7.1f} shares, "
          f"residual {u['measured']:>4d}")
print("   Residual here is the max_participation cap refusing to absorb an")
print("   oversized position into one bar. That cap is never bypassed --")
print("   inventing liquidity is the defect class this project keeps")
print("   rediscovering -- so the residual is counted, not hidden.")

print()
assert real["boundaries"] > 3, "too few session boundaries for this to mean anything"
assert ctrl["measured"] > 0, "control carried nothing -- the test cannot detect a carry"
# The spec is not "residual is zero" -- it is "residual happens only where the
# closing bar had no volume at all". A forced close is an order, and an order
# cannot fill on a bar where nothing traded. Asserting zero would only be
# satisfiable by uncapping the fill, i.e. by inventing liquidity.
assert real["measured"] == real["untradeable"], (
    f"FAIL: {real['measured'] - real['untradeable']} position(s) survived a close "
    f"on a bar that DID have volume -- that is a real leak, not a market limit")
assert real["counted"] == real["measured"], (
    f"env counter {real['counted']} disagrees with independent count {real['measured']}")
assert real["starts"] >= real["boundaries"] - 1, "session start not flagged at every boundary"
print(f"PASS: {real['measured']} carried at bounded inventory (all {real['untradeable']} "
      f"on zero-volume bars) vs {ctrl['measured']:,} with the flag off, across "
      f"{real['boundaries']} session closes")
