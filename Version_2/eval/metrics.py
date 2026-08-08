"""
eval/metrics.py

Sharpe ratio, Sortino ratio, max drawdown, win rate, and buy-and-hold (B&H)
benchmark comparison -- computed both per-ticker and at the portfolio level.

All functions operate on CURVES (full time series recorded during a
backtest), since every metric here needs the whole path, not just start/end
values. eval/backtest_report.py is responsible for actually running the
policy and recording equity/price/trade-pnl curves; this file only computes
numbers from them.

Shape convention throughout: a "curve" is [T] for a single stream/portfolio
or [T, n_envs] for all streams (one column per ticker, matching
vec_trading_env.py's n_envs == n_tickers layout). Every function reduces
over dim=0 (time), so the same code path handles both shapes without a
special case.

Bar-to-annual scaling: this project trades 5-minute bars, so annualizing a
Sharpe/Sortino computed on raw per-bar returns needs sqrt(bars_per_year),
NOT sqrt(252) (that constant assumes daily bars). The default
bars_per_year=19656 assumes a 6.5-hour US equity session (390 minutes / 5 =
78 bars/day) times 252 trading days/year -- override this if your bar size,
session length, or trading calendar differs.
"""

from dataclasses import dataclass
from typing import Optional

import torch

DEFAULT_BARS_PER_YEAR = 78 * 252  # 5-min bars, 6.5h US equity session, 252 trading days/year


def simple_returns(equity_curve: torch.Tensor) -> torch.Tensor:
    """equity_curve: [T] or [T, n_envs]. Returns [T-1] or [T-1, n_envs]."""
    return equity_curve[1:] / equity_curve[:-1].clamp(min=1e-8) - 1.0


def sharpe_ratio(
    returns: torch.Tensor,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
) -> torch.Tensor:
    """returns: [T] or [T, n_envs] per-bar simple returns. Returns annualized Sharpe."""
    rf_per_bar = risk_free_rate_annual / bars_per_year
    excess = returns - rf_per_bar
    mean = excess.mean(dim=0)
    std = excess.std(dim=0).clamp(min=1e-8)
    return (mean / std) * (bars_per_year ** 0.5)


def sortino_ratio(
    returns: torch.Tensor,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
) -> torch.Tensor:
    """
    Same as sharpe_ratio() but the denominator only penalizes downside
    deviation (returns below the risk-free/target rate), not upside
    volatility -- distinguishes "risky" from "profitable but lumpy."
    """
    rf_per_bar = risk_free_rate_annual / bars_per_year
    excess = returns - rf_per_bar
    downside = torch.where(excess < 0, excess, torch.zeros_like(excess))
    downside_std = torch.sqrt(downside.pow(2).mean(dim=0)).clamp(min=1e-8)
    mean = excess.mean(dim=0)
    return (mean / downside_std) * (bars_per_year ** 0.5)


def drawdown_series(equity_curve: torch.Tensor) -> torch.Tensor:
    """
    equity_curve: [T] or [T, n_envs].
    Returns the running drawdown fraction at every step (0 = at a new peak,
    0.15 = currently 15% below the running peak), same shape as the input --
    the full path, not just its max. Use this for a drawdown chart;
    max_drawdown() below is just this reduced with .max(dim=0).
    """
    running_peak = torch.cummax(equity_curve, dim=0).values
    return (running_peak - equity_curve) / running_peak.clamp(min=1e-8)


def max_drawdown(equity_curve: torch.Tensor) -> torch.Tensor:
    """equity_curve: [T] or [T, n_envs]. Returns the largest peak-to-trough decline as a positive fraction."""
    return drawdown_series(equity_curve).max(dim=0).values


def win_rate(trade_pnls: torch.Tensor) -> torch.Tensor:
    """
    trade_pnls: [n_trades] or [n_trades, n_envs] -- realized PnL of CLOSED
    trades only. Entries where no trade closed must be NaN, not 0, so they
    get excluded rather than counted as break-even/losing (matches
    risk/kelly_sizing.py's KellySizer convention of NaN-padding empty slots,
    and vec_trading_env.py's info["realized_delta"], which is exactly 0.0
    on any step where nothing closed).
    """
    valid = ~torch.isnan(trade_pnls)
    wins = valid & (trade_pnls > 0)
    n_valid = valid.sum(dim=0).clamp(min=1)
    return wins.sum(dim=0).float() / n_valid.float()


def buy_and_hold_curve(price_curve: torch.Tensor, initial_cash: float) -> torch.Tensor:
    """
    price_curve: [T] or [T, n_envs] raw prices over the backtest window.
    Returns the equity curve of "buy at t=0 with initial_cash, hold to the
    end" -- same shape as price_curve, same starting value (initial_cash)
    at t=0.
    """
    shares = initial_cash / price_curve[0].clamp(min=1e-8)
    return shares * price_curve


def aggregate_equity(equity_curve: torch.Tensor) -> torch.Tensor:
    """
    equity_curve: [T, n_envs] -- one equity path per ticker/stream, each
    started from its own initial_cash (vec_trading_env.py gives every
    stream its own PortfolioState).

    Portfolio-level equity here is the SUM across streams: "run this policy
    on all n_envs tickers at once, one equal-sized capital bucket per
    ticker, no cross-ticker reallocation." If your real portfolio would
    size tickers unevenly, weight the streams before summing rather than
    using this as-is.

    Returns [T] -- summed portfolio equity curve.
    """
    return equity_curve.sum(dim=1)


def aggregate_bh_curve(price_curve: torch.Tensor, initial_cash_per_ticker: float) -> torch.Tensor:
    """
    Portfolio-level buy-and-hold benchmark, built consistently with
    aggregate_equity()'s equal-capital-per-ticker assumption: buy
    initial_cash_per_ticker of EACH ticker at t=0 and sum the resulting
    per-ticker B&H equity curves.

    price_curve: [T, n_envs]. Returns [T].
    """
    per_ticker_bh = buy_and_hold_curve(price_curve, initial_cash_per_ticker)  # [T, n_envs]
    return per_ticker_bh.sum(dim=1)


def aggregate_trade_pnls(trade_pnls: torch.Tensor) -> torch.Tensor:
    """
    trade_pnls: [n_trades, n_envs], NaN-padded per win_rate()'s convention.
    Returns a flattened [n_trades * n_envs] view, for a portfolio-level win
    rate that doesn't care which ticker a closed trade came from.
    """
    return trade_pnls.reshape(-1)


@dataclass
class MetricsResult:
    sharpe: torch.Tensor
    sortino: torch.Tensor
    max_drawdown: torch.Tensor
    win_rate: torch.Tensor
    total_return: torch.Tensor
    bh_total_return: torch.Tensor
    alpha_vs_bh: torch.Tensor       # total_return - bh_total_return
    bh_max_drawdown: torch.Tensor


def compute_metrics(
    equity_curve: torch.Tensor,     # [T] or [T, n_envs]
    bh_curve: torch.Tensor,         # same shape as equity_curve -- from buy_and_hold_curve() / aggregate_bh_curve()
    trade_pnls: torch.Tensor,       # [n_trades] or [n_trades, n_envs], NaN-padded (see win_rate())
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    risk_free_rate_annual: float = 0.0,
) -> MetricsResult:
    """
    Same function computes per-ticker metrics (2D curves) or portfolio-level
    metrics (1D curves, built via aggregate_equity() / aggregate_bh_curve()
    / aggregate_trade_pnls()) -- caller builds the right-shaped curve first,
    this just reduces over the time axis either way.
    """
    returns = simple_returns(equity_curve)

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0].clamp(min=1e-8)
    bh_total_return = (bh_curve[-1] - bh_curve[0]) / bh_curve[0].clamp(min=1e-8)

    return MetricsResult(
        sharpe=sharpe_ratio(returns, bars_per_year, risk_free_rate_annual),
        sortino=sortino_ratio(returns, bars_per_year, risk_free_rate_annual),
        max_drawdown=max_drawdown(equity_curve),
        win_rate=win_rate(trade_pnls),
        total_return=total_return,
        bh_total_return=bh_total_return,
        alpha_vs_bh=total_return - bh_total_return,
        bh_max_drawdown=max_drawdown(bh_curve),
    )