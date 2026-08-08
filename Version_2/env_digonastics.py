"""
Diagnostic script for the trading environment (env/ package).
Run this locally or on Kaggle to verify portfolio_state.py, execution_sim.py,
and vec_trading_env.py behave correctly BEFORE wiring them into PPO training.

Specifically validates the HYBRID ACTION contract, since that's the part
most likely to silently misbehave if a policy network hands the env
slightly-off values:

    direction     : must be exactly one of {-1, 0, 1} (discrete head).
                    Anything else (e.g. a raw 0.7 from an un-rounded policy
                    output) is a contract violation -- this env does NOT
                    treat direction as a continuous scaling factor, and if
                    you feed it a fractional direction it will silently
                    scale filled_qty by that fraction instead of raising an
                    error. That's the #1 way this environment gets misused,
                    so we test for it explicitly below.
    size          : continuous, unsigned, >= 0. Requested trade size in
                    shares. Zero means "no size" and combines with any
                    direction to produce no fill.
    limit_offset  : continuous, in ticks, can be any sign/magnitude. Shifts
                    the effective fill price in the agent's favor.

If your policy's output heads don't already enforce this (e.g. a
Categorical/argmax for direction and a Softplus/sigmoid*max for size), this
script will catch it before you burn a training run on garbage fills.
"""

import os
import sys
import json
import traceback

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "env"))


class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(step: str, success: bool, message: str = ""):
    if success:
        print(f"{Colors.OKGREEN}✅ [PASS]{Colors.ENDC} {step}")
    else:
        print(f"{Colors.FAIL}❌ [FAIL]{Colors.ENDC} {step}")
    if message:
        print(f"   └─ {message}")


def print_warn(step: str, message: str = ""):
    print(f"{Colors.WARNING}⚠️  [WARN]{Colors.ENDC} {step}")
    if message:
        print(f"   └─ {message}")


def run_diagnostics():
    print(f"\n{Colors.BOLD}=== RUNNING ENVIRONMENT DIAGNOSTICS ==={Colors.ENDC}\n")

    # ------------------------------------------------------------------
    print(f"{Colors.BOLD}[1/6] Checking Dependencies & Imports...{Colors.ENDC}")
    try:
        import torch
        from portfolio_state import PortfolioState, Fill
        from execution_sim import ExecutionSimulator, SimulatedFill
        print_status("Core env modules imported", True, f"PyTorch: {torch.__version__}")
    except Exception as e:
        print_status("Core env modules imported", False, str(e))
        traceback.print_exc()
        return

    device = "cpu"  # diagnostics run on CPU regardless of training device, for determinism

    # ------------------------------------------------------------------
    print(f"\n{Colors.BOLD}[2/6] PortfolioState: Position Lifecycle Edge Cases...{Colors.ENDC}")
    try:
        ps = PortfolioState(n_envs=6, n_tickers=1, initial_cash=10_000.0, device=device)

        # env0: open long        | env1: open short       | env2: zero-size fill (no-op)
        # env3: open long tiny   | env4: open long        | env5: open short
        qty1 = torch.tensor([10., -10., 0., 0.001, 20., -20.])
        price1 = torch.tensor([10., 10., 10., 10., 5., 5.])
        f1 = Fill(0, qty1, price1, torch.zeros(6))
        r1 = ps.step_apply(f1)
        assert torch.allclose(ps.positions.squeeze(), qty1), "positions after open don't match requested qty"
        assert torch.allclose(r1, torch.zeros(6)), "opening a position should never realize PnL"
        print_status("Open (long/short/zero-size/tiny)", True,
                     f"positions={ps.positions.squeeze().tolist()}")

        # env0: add more long    | env1: add more short   | env2: still zero
        # env3: full close       | env4: partial close    | env5: flip
        qty2 = torch.tensor([5., -5., 0., -0.001, -5., 25.])
        price2 = torch.tensor([12., 8., 10., 10., 6., 6.])
        f2 = Fill(0, qty2, price2, torch.zeros(6))
        r2 = ps.step_apply(f2)
        expected_pos = torch.tensor([15., -15., 0., 0., 15., 5.])
        assert torch.allclose(ps.positions.squeeze(), expected_pos, atol=1e-6), \
            f"expected {expected_pos.tolist()}, got {ps.positions.squeeze().tolist()}"
        # env3 fully closed -> entry price must reset to 0
        assert ps.avg_entry_price.squeeze()[3].item() == 0.0, "entry price should reset to 0 on full close"
        # env5 flipped -> entry price must reset to the NEW fill price, not the old one
        assert ps.avg_entry_price.squeeze()[5].item() == 6.0, \
            f"flip should reset entry to fill price 6.0, got {ps.avg_entry_price.squeeze()[5].item()}"
        print_status("Add / partial close / full close / flip", True,
                     f"positions={ps.positions.squeeze().tolist()}, "
                     f"entries={ps.avg_entry_price.squeeze().tolist()}")

        # Degenerate inputs: NaN/Inf should never silently enter the ledger
        ps2 = PortfolioState(n_envs=2, n_tickers=1, initial_cash=10_000.0, device=device)
        f3 = Fill(0, torch.tensor([10., 0.]), torch.tensor([10., 10.]), torch.zeros(2))
        ps2.step_apply(f3)
        has_nan = torch.isnan(ps2.positions).any() or torch.isnan(ps2.avg_entry_price).any() or torch.isnan(ps2.cash).any()
        assert not has_nan, "NaN leaked into portfolio state from a degenerate (zero-qty) fill"
        print_status("Degenerate / zero-qty fill produces no NaNs", True)

    except AssertionError as e:
        print_status("PortfolioState lifecycle", False, str(e))
    except Exception as e:
        print_status("PortfolioState lifecycle", False, f"Unexpected error: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    print(f"\n{Colors.BOLD}[3/6] ExecutionSimulator: Slippage, Snapping, Partial Fills...{Colors.ENDC}")
    try:
        sim = ExecutionSimulator(tick_size=0.01, spread_bps=5, impact_coef=0.05, max_participation=0.1)

        direction = torch.tensor([1., -1., 0., 1., -1.])
        size = torch.tensor([100., 100., 0., 1e9, 1e9])   # last two request way more than liquidity
        mid = torch.tensor([100., 100., 100., 100., 100.])
        liq = torch.tensor([10_000.] * 5)
        limit_offset = torch.zeros(5)

        res = sim.simulate_fill(direction, size, limit_offset, mid, liq)

        assert res.filled_qty[2].item() == 0.0, "direction=0 must never produce a fill"
        assert res.fill_price[0].item() > mid[0].item(), "buy should slip price UP vs mid"
        assert res.fill_price[1].item() < mid[1].item(), "sell should slip price DOWN vs mid"
        assert res.is_partial[3].item() and res.is_partial[4].item(), "oversized orders must be marked partial"
        assert res.filled_qty[3].item() <= 0.1 * 10_000, "partial fill exceeded max_participation cap"

        # Tick snapping: fill price must be an exact multiple of tick_size
        remainder = (res.fill_price / sim.tick_size).round() * sim.tick_size - res.fill_price
        assert torch.allclose(remainder, torch.zeros_like(remainder), atol=1e-6), "fill price not snapped to tick_size"

        print_status("Slippage direction, partial fills, tick snapping", True,
                     f"fill_price={[round(p,4) for p in res.fill_price.tolist()]}")

        # limit_offset should move price in the agent's favor
        favorable = sim.simulate_fill(torch.tensor([1.]), torch.tensor([100.]), torch.tensor([50.]), torch.tensor([100.]), torch.tensor([10_000.]))
        baseline = sim.simulate_fill(torch.tensor([1.]), torch.tensor([100.]), torch.tensor([0.]), torch.tensor([100.]), torch.tensor([10_000.]))
        assert favorable.fill_price.item() < baseline.fill_price.item(), \
            "a favorable (positive, buy-side) limit_offset should REDUCE the effective buy price"
        print_status("limit_offset moves fill price in agent's favor", True,
                     f"baseline={baseline.fill_price.item():.4f} vs favorable={favorable.fill_price.item():.4f}")

    except AssertionError as e:
        print_status("ExecutionSimulator behavior", False, str(e))
    except Exception as e:
        print_status("ExecutionSimulator behavior", False, f"Unexpected error: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    print(f"\n{Colors.BOLD}[4/6] Hybrid Action Contract Validation...{Colors.ENDC}")
    print("   (This is the check most likely to catch a policy-network integration bug)")
    try:
        sim = ExecutionSimulator(tick_size=0.01, spread_bps=5, impact_coef=0.05, max_participation=0.1)
        mid = torch.tensor([100.])
        liq = torch.tensor([10_000.])

        # Contract violation: a fractional direction (e.g. un-rounded policy output).
        # execution_sim.py now HARD-ENFORCES direction ∈ {-1,0,1} and raises
        # ValueError on violation (previously this silently scaled the fill
        # by the fractional value instead — see the earlier WARN this
        # replaced). This step confirms the raise actually happens.
        bad_direction = torch.tensor([0.7])
        clean_direction = torch.tensor([1.0])
        size = torch.tensor([100.])
        limit_offset = torch.tensor([0.])

        try:
            sim.simulate_fill(bad_direction, size, limit_offset, mid, liq)
            print_status(
                "Fractional direction rejected", False,
                "direction=0.7 was silently accepted instead of raising -- the contract enforcement regressed."
            )
        except ValueError:
            print_status("Fractional direction correctly rejected (raises ValueError)", True)

        clean_fill = sim.simulate_fill(clean_direction, size, limit_offset, mid, liq)
        assert clean_fill.filled_qty.item() > 0, "a valid direction should still fill normally"
        print_status("Valid direction still fills normally after the contract check", True)

        # Negative size should be clamped, not produce a negative fill
        neg_size_fill = sim.simulate_fill(torch.tensor([1.0]), torch.tensor([-50.]), limit_offset, mid, liq)
        assert neg_size_fill.filled_qty.item() >= 0, "negative size leaked through as a negative fill"
        print_status("Negative size is clamped to zero", True)

    except Exception as e:
        print_status("Hybrid action contract checks", False, f"Unexpected error: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    print(f"\n{Colors.BOLD}[5/6] VecTradingEnv: Full Rollout Against Real Project Data...{Colors.ENDC}")
    env = None
    try:
        from paths import PROCESSED_DIR
        meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
        if not os.path.exists(meta_path):
            print_warn("Skipping VecTradingEnv rollout test",
                        f"No metadata.json at {meta_path}. Run fetch_alpaca.py + preprocess.py first, "
                        f"then re-run this diagnostic for the full integration test.")
        else:
            from dataset import MultiTickerRolloutDataset
            from vec_trading_env import VecTradingEnv

            ds = MultiTickerRolloutDataset(window_size=20, split='train', device=device)
            env = VecTradingEnv(ds, initial_cash=10_000.0)
            obs = env.reset()

            expected_shape = (env.n_envs, env.window_size, len(env.feature_names))
            assert tuple(obs.shape) == expected_shape, f"expected obs shape {expected_shape}, got {tuple(obs.shape)}"

            torch.manual_seed(0)
            n_steps = min(30, len(env) - 1)
            saw_nan = False
            saw_inf = False
            for step in range(n_steps):
                # Correctly-discretized hybrid action, as a policy head should produce:
                direction = torch.randint(-1, 2, (env.n_envs,)).float()   # {-1, 0, 1}
                size = torch.rand(env.n_envs) * 10                        # continuous size >= 0
                limit_offset = (torch.rand(env.n_envs) - 0.5) * 4         # continuous, +/- 2 ticks
                result = env.step(direction, size, limit_offset)

                if torch.isnan(result.reward).any() or torch.isnan(result.obs).any() or torch.isnan(result.info["equity"]).any():
                    saw_nan = True
                if torch.isinf(result.reward).any() or torch.isinf(result.info["equity"]).any():
                    saw_inf = True

            assert not saw_nan, "NaN appeared in obs/reward/equity during rollout"
            assert not saw_inf, "Inf appeared in reward/equity during rollout"
            print_status("Full rollout, correctly-discretized hybrid actions", True,
                         f"{n_steps} steps, obs_shape={tuple(obs.shape)}, "
                         f"final equity={result.info['equity'].tolist()}")

            # Same rollout but with a fractional/malformed direction, to confirm
            # the hardened contract from step [4/6] also propagates up through
            # VecTradingEnv.step() -> ExecutionSimulator.simulate_fill(), not
            # just when execution_sim.py is called directly.
            ds2 = MultiTickerRolloutDataset(window_size=20, split='train', device=device)
            env2 = VecTradingEnv(ds2, initial_cash=10_000.0)
            env2.reset()
            bad_direction = torch.full((env2.n_envs,), 0.35)
            size = torch.full((env2.n_envs,), 10.0)
            limit_offset = torch.zeros(env2.n_envs)
            try:
                env2.step(bad_direction, size, limit_offset)
                print_status(
                    "Malformed direction rejected at full-env level", False,
                    "env2.step() silently accepted direction=0.35 instead of raising -- the contract "
                    "enforcement isn't reaching all the way through VecTradingEnv.step()."
                )
            except ValueError:
                print_status("Malformed direction correctly rejected at full-env level (raises ValueError)", True)

    except Exception as e:
        print_status("VecTradingEnv rollout", False, f"Unexpected error: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    print(f"\n{Colors.BOLD}[6/6] Episode Termination...{Colors.ENDC}")
    try:
        if env is None:
            print_warn("Skipping termination check", "VecTradingEnv wasn't constructed in step 5.")
        else:
            from dataset import MultiTickerRolloutDataset
            ds3 = MultiTickerRolloutDataset(window_size=20, split='train', device=device)
            env3 = VecTradingEnv(ds3, initial_cash=10_000.0)
            env3.reset()
            done_any = False
            for _ in range(len(env3) + 2):  # deliberately overshoot the split length
                direction = torch.zeros(env3.n_envs)
                size = torch.zeros(env3.n_envs)
                limit_offset = torch.zeros(env3.n_envs)
                result = env3.step(direction, size, limit_offset)
                if result.done.any():
                    done_any = True
                    break
            assert done_any, "env never signaled done, even after overshooting the split length"
            print_status("Episode correctly signals done at end of split", True)
    except Exception as e:
        print_status("Episode termination", False, f"Unexpected error: {e}")
        traceback.print_exc()

    print(f"\n{Colors.BOLD}=== ENVIRONMENT DIAGNOSTICS COMPLETE ==={Colors.ENDC}\n")


if __name__ == "__main__":
    run_diagnostics()