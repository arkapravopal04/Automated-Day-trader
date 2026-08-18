"""
eval/backtest_report.py

Runs the trained policy on the held-out test set, generates plots (equity
curve vs buy-and-hold, drawdown chart, per-ticker attribution), and produces
the go/no-go verdict: does the strategy beat buy-and-hold with an
acceptable max drawdown.

This runs the FULL deployed action pipeline by default, not just the bare
policy network -- the same path as training/ppo_hybrid.py's
collect_rollout() (rescale -> KellySizer -> RiskManager -> KillSwitch ->
env.step()), with the policy in deterministic mode
(HybridPolicyHead.act(..., deterministic=True)). A backtest that bypasses
the risk pipeline tells you what the model WANTS to do, not what the system
would actually do live -- see run_backtest()'s use_risk_pipeline argument.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless -- this runs on Kaggle/servers, not an interactive desktop
import matplotlib.pyplot as plt
import torch

from model.hybrid_policy import HybridPolicyHead

from risk.kelly_sizing import KellySizer
from risk.risk_manager import RiskManager, RiskLimits
from risk.kill_switch import KillSwitch

from vec_trading_env import VecTradingEnv
from training.ppo_hybrid import HybridActorCritic
from training.config import TrainingConfig

from eval.metrics import (
    MetricsResult,
    compute_metrics,
    buy_and_hold_curve,
    aggregate_equity,
    aggregate_bh_curve,
    aggregate_trade_pnls,
    drawdown_series,
)


@dataclass
class BacktestOutput:
    """Outputs captured from one deterministic backtest pass."""
    equity_curve: torch.Tensor      # [T+1, n_envs]
    price_curve: torch.Tensor       # [T+1, n_envs]
    trade_pnls: torch.Tensor        # [T, n_envs], NaN where no trade closed that step
    per_ticker_metrics: MetricsResult
    portfolio_metrics: MetricsResult
    go_no_go: bool
    go_no_go_reasons: List[str]


def run_backtest(
    env: VecTradingEnv,
    actor_critic: HybridActorCritic,
    cfg: TrainingConfig,
    use_risk_pipeline: Optional[bool] = None,
    max_drawdown_limit: Optional[float] = None,
    require_beat_bh: Optional[bool] = None,
) -> BacktestOutput:
    """
    Walks `env` start to finish exactly once, deterministically, recording
    equity/price/trade-pnl every step. Assumes `env` was freshly constructed
    on a held-out test split (per project convention, a
    MultiTickerRolloutDataset built from data the model never trained on)
    and that its `done` flag fires naturally at the end of that single pass
    -- this function does not loop multiple passes or reset mid-backtest.

    use_risk_pipeline / max_drawdown_limit / require_beat_bh default to
    cfg.eval's values (training/config.py's EvalConfig) when left as None;
    pass an explicit value here only to override cfg for a one-off run.

    use_risk_pipeline=True (default via cfg.eval): applies the full
    KellySizer -> RiskManager -> KillSwitch pipeline exactly as
    training/ppo_hybrid.py's collect_rollout() does, so the go/no-go
    verdict reflects the deployed system, including position caps, the
    drawdown halt, and Kelly sizing. Setting it False feeds the policy's
    rescaled action straight to env.step() -- useful for isolating "is the
    raw policy any good" during debugging, but should NOT be what decides
    go/no-go, since it silently ignores every safety rail the live system
    would actually have on.

    Note: kelly_sizer/risk_manager/kill_switch are constructed FRESH here
    (cold-start Kelly edge estimate, no carried-over halt state from
    training) -- this is a clean, from-scratch simulation of "if we deployed
    today with a blank risk history," not a continuation of whatever state
    those objects were in at the end of training.
    """
    use_risk_pipeline = cfg.eval.use_risk_pipeline if use_risk_pipeline is None else use_risk_pipeline
    max_drawdown_limit = cfg.eval.max_drawdown_limit if max_drawdown_limit is None else max_drawdown_limit
    require_beat_bh = cfg.eval.require_beat_bh if require_beat_bh is None else require_beat_bh

    device = next(actor_critic.parameters()).device
    n_envs = env.n_envs
    T = env.max_idx  # one full deterministic pass over the test split

    kelly_sizer = KellySizer(
        n_envs=n_envs,
        lookback_trades=cfg.risk.kelly_lookback_trades,
        min_trades_for_estimate=cfg.risk.kelly_min_trades_for_estimate,
        kelly_multiplier=cfg.risk.kelly_multiplier,
        kelly_cap=cfg.risk.kelly_cap,
        default_fraction=cfg.risk.kelly_default_fraction,
        device=str(device),
    )
    risk_manager = RiskManager(
        RiskLimits(
            max_position_frac=cfg.risk.max_position_frac,
            max_gross_exposure_frac=cfg.risk.max_gross_exposure_frac,
            max_ticker_concentration_frac=cfg.risk.max_ticker_concentration_frac,
            max_order_notional_frac=cfg.risk.max_order_notional_frac,
            drawdown_halt_frac=cfg.risk.drawdown_halt_frac,
            min_order_notional=cfg.risk.min_order_notional,
        ),
        device=str(device),
    )
    kill_switch = KillSwitch(
        n_envs=n_envs,
        daily_loss_limit_frac=cfg.risk.daily_loss_limit_frac,
        broker_error_streak_limit=cfg.risk.broker_error_streak_limit,
        state_mismatch_tolerance=cfg.risk.state_mismatch_tolerance,
        device=str(device),
    )

    obs = env.reset()
    hidden = actor_critic.init_hidden(n_envs, device)

    start_mid_price = env._current_prices()  # noqa: SLF001 -- same accessor env.step() itself uses internally
    kill_switch.start_new_day(env.portfolio.equity(start_mid_price.unsqueeze(1)))

    equity_hist = [env.portfolio.equity(start_mid_price.unsqueeze(1))]
    price_hist = [start_mid_price]
    trade_pnl_hist = []

    was_training = actor_critic.training
    actor_critic.eval()
    try:
        with torch.no_grad():
            for _ in range(T):
                mid_price = env._current_prices()  # noqa: SLF001
                equity_before = env.portfolio.equity(mid_price.unsqueeze(1))
                current_position_notional = env.portfolio.positions[:, 0] * mid_price

                trunk, hidden = actor_critic.forward_features(obs, hidden)
                action_sample = actor_critic.policy_head.act(trunk, deterministic=True)

                size_shares = HybridPolicyHead.rescale_size(
                    action_sample.size, torch.full_like(action_sample.size, cfg.risk.max_order_shares)
                )
                limit_offset_ticks = HybridPolicyHead.rescale_limit_offset(
                    action_sample.limit_offset, cfg.risk.max_limit_offset_ticks
                )

                if use_risk_pipeline:
                    kelly_result = kelly_sizer.apply(
                        size=size_shares,
                        direction=action_sample.direction,
                        mid_price=mid_price,
                        equity=equity_before,
                        current_position_notional=current_position_notional,
                    )
                    risk_result = risk_manager.apply(
                        direction=action_sample.direction,
                        size=kelly_result.size,
                        limit_offset=limit_offset_ticks,
                        mid_price=mid_price,
                        portfolio=env.portfolio,
                        ticker_idx=0,
                    )
                    final_direction, final_size = kill_switch.apply(risk_result.direction, risk_result.size)
                    final_limit_offset = risk_result.limit_offset
                else:
                    final_direction = action_sample.direction
                    final_size = size_shares
                    final_limit_offset = limit_offset_ticks

                step_result = env.step(direction=final_direction, size=final_size, limit_offset=final_limit_offset)

                if use_risk_pipeline:
                    kelly_sizer.record_realized_pnl(
                        step_result.info["realized_delta"],
                        closed_mask=step_result.info.get("closed_trade"),
                    )
                    kill_switch.check_daily_loss(step_result.info["equity"])

                equity_hist.append(step_result.info["equity"].clone())
                price_hist.append(env._current_prices().clone())  # noqa: SLF001 -- mark AFTER this step, for the next bar
                realized = step_result.info["realized_delta"]
                trade_pnl_hist.append(torch.where(realized != 0, realized, torch.full_like(realized, float("nan"))))

                obs = step_result.obs
                if step_result.done.any():
                    break  # single deterministic pass -- stop rather than auto-reset into a new pass

    finally:
        actor_critic.train(was_training)

    equity_curve = torch.stack(equity_hist, dim=0)   # [T+1, n_envs]
    price_curve = torch.stack(price_hist, dim=0)     # [T+1, n_envs]
    trade_pnls = (
        torch.stack(trade_pnl_hist, dim=0) if trade_pnl_hist
        else torch.full((0, n_envs), float("nan"), device=device)
    )

    bh_curve = buy_and_hold_curve(price_curve, env.initial_cash)
    portfolio_equity = aggregate_equity(equity_curve)
    portfolio_bh = aggregate_bh_curve(price_curve, env.initial_cash)
    portfolio_trade_pnls = aggregate_trade_pnls(trade_pnls)

    per_ticker_metrics = compute_metrics(
        equity_curve, bh_curve, trade_pnls,
        bars_per_year=cfg.eval.bars_per_year, risk_free_rate_annual=cfg.eval.risk_free_rate_annual,
    )
    portfolio_metrics = compute_metrics(
        portfolio_equity, portfolio_bh, portfolio_trade_pnls,
        bars_per_year=cfg.eval.bars_per_year, risk_free_rate_annual=cfg.eval.risk_free_rate_annual,
    )

    reasons: List[str] = []
    beats_bh = bool((portfolio_metrics.alpha_vs_bh > 0).item())
    dd_ok = bool((portfolio_metrics.max_drawdown <= max_drawdown_limit).item())
    if require_beat_bh and not beats_bh:
        reasons.append(f"portfolio alpha vs buy-and-hold is negative ({portfolio_metrics.alpha_vs_bh.item():.4f})")
    if not dd_ok:
        reasons.append(
            f"portfolio max drawdown {portfolio_metrics.max_drawdown.item():.2%} "
            f"exceeds the {max_drawdown_limit:.2%} limit"
        )
    go = (beats_bh or not require_beat_bh) and dd_ok

    return BacktestOutput(
        equity_curve=equity_curve,
        price_curve=price_curve,
        trade_pnls=trade_pnls,
        per_ticker_metrics=per_ticker_metrics,
        portfolio_metrics=portfolio_metrics,
        go_no_go=go,
        go_no_go_reasons=reasons,
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_equity_vs_bh(output: BacktestOutput, out_path: str) -> None:
    """Save portfolio strategy equity versus buy-and-hold as a PNG."""
    portfolio_equity = aggregate_equity(output.equity_curve).cpu().numpy()
    # each stream starts at its own initial_cash, which is the same value
    # across streams by construction (vec_trading_env.py's initial_cash arg
    # applies uniformly) -- recovered from the recorded curve itself rather
    # than threaded through as a separate argument.
    initial_cash_per_ticker = float(output.equity_curve[0, 0].item())
    portfolio_bh = aggregate_bh_curve(output.price_curve, initial_cash_per_ticker).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio_equity, label="Strategy", linewidth=1.5)
    ax.plot(portfolio_bh, label="Buy & Hold", linewidth=1.5, linestyle="--")
    ax.set_title("Portfolio Equity: Strategy vs Buy & Hold")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown(output: BacktestOutput, out_path: str) -> None:
    """Save the portfolio drawdown series as a PNG."""
    portfolio_equity = aggregate_equity(output.equity_curve)
    dd = drawdown_series(portfolio_equity).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(range(len(dd)), -dd * 100, 0, color="firebrick", alpha=0.6)
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_ticker_attribution(output: BacktestOutput, tickers: List[str], out_path: str) -> None:
    """Save per-ticker total-return and alpha attribution as a PNG."""
    total_return = output.per_ticker_metrics.total_return.cpu().numpy()
    alpha = output.per_ticker_metrics.alpha_vs_bh.cpu().numpy()

    x = range(len(tickers))
    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 0.6), 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], total_return * 100, width, label="Total Return")
    ax.bar([i + width / 2 for i in x], alpha * 100, width, label="Alpha vs B&H")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Per-Ticker Attribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt_metrics_row(name: str, m: MetricsResult, idx: Optional[int] = None) -> str:
    """Format one metrics object as a markdown table row."""
    def g(t: torch.Tensor) -> float:
        return float(t[idx].item()) if idx is not None else float(t.item())

    return (
        f"| {name} | {g(m.sharpe):.2f} | {g(m.sortino):.2f} | {g(m.max_drawdown):.2%} | "
        f"{g(m.win_rate):.2%} | {g(m.total_return):.2%} | {g(m.bh_total_return):.2%} | {g(m.alpha_vs_bh):.2%} |"
    )


def generate_report(output: BacktestOutput, tickers: List[str], out_dir: str) -> str:
    """
    Saves three PNGs (equity-vs-B&H, drawdown, per-ticker attribution) plus
    a markdown summary with per-ticker and portfolio metrics tables and the
    go/no-go verdict, into out_dir. Returns the path to the markdown file.
    """
    os.makedirs(out_dir, exist_ok=True)

    equity_png = os.path.join(out_dir, "equity_vs_bh.png")
    drawdown_png = os.path.join(out_dir, "drawdown.png")
    attribution_png = os.path.join(out_dir, "per_ticker_attribution.png")

    plot_equity_vs_bh(output, equity_png)
    plot_drawdown(output, drawdown_png)
    plot_per_ticker_attribution(output, tickers, attribution_png)

    header = "| Ticker | Sharpe | Sortino | Max DD | Win Rate | Total Return | B&H Return | Alpha vs B&H |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [_fmt_metrics_row(t, output.per_ticker_metrics, idx=i) for i, t in enumerate(tickers)]
    portfolio_row = _fmt_metrics_row("PORTFOLIO", output.portfolio_metrics)

    verdict = "GO" if output.go_no_go else "NO-GO"
    reasons_block = (
        "\n".join(f"- {r}" for r in output.go_no_go_reasons) if output.go_no_go_reasons else "- (no blocking issues)"
    )

    report = f"""# Backtest Report

## Verdict: **{verdict}**

{reasons_block}

## Portfolio-Level Metrics

{header}
{sep}
{portfolio_row}

## Per-Ticker Metrics

{header}
{sep}
{chr(10).join(rows)}

## Plots

![Equity vs Buy & Hold](equity_vs_bh.png)

![Drawdown](drawdown.png)

![Per-Ticker Attribution](per_ticker_attribution.png)
"""
    report_path = os.path.join(out_dir, "backtest_report.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)

    return report_path