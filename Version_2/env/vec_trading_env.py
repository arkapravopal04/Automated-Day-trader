"""
env/vec_trading_env.py

The vectorized multi-asset environment. Each of the n_tickers instruments in
MultiTickerRolloutDataset is treated as its own independent parallel rollout
stream (n_envs == n_tickers), consistent with the [n_envs, window, features]
shape dataset.py already produces. Each stream trades exactly one
instrument and owns its own single-asset PortfolioState row.

step() accepts a batch of hybrid actions (direction, size, limit_offset) —
one triple per stream — and returns next observation, reward, done, info,
one row per stream.

Wiring:
    dataset.py       -> normalized feature observations [n_envs, window, features]
    execution_sim.py  -> turns actions into simulated fills
    portfolio_state.py -> bookkeeping ledger updated from those fills

Note on prices: dataset.py's tensor is fully normalized (log returns, vol
z-score, etc.) and intentionally has no raw price column (that's correct for
model inputs — you don't want to feed absolute price levels into the net).
But execution and PnL marking need actual prices, so this env independently
loads raw close prices from RAW_DIR and aligns them to the same dates the
dataset uses.

Directional bias mitigation:
The historical window for these tickers is not directionally neutral (some
tickers are net-bullish over the sampled years, some net-bearish), which lets
a PPO agent learn a static long/short bias instead of a context-dependent
policy. Two env-side mitigations are applied here (mirroring + a diversity
reward bonus); the remaining mitigations from the project's directional-bias
writeup (adaptive symmetry penalty, tanh-squashed action head, discrete
direction head, dual critics) live in the policy/loss code, not here.

1. Return mirroring: each stream (ticker) independently has a `mirror_prob`
   chance, decided once per full pass through the dataset, of having its
   price path reflected (log-returns negated, same start price) and its
   return-type obs features sign-flipped to match. A mirrored bull ticker
   becomes a synthetic bear ticker (and vice versa) for that pass, so the
   agent can't just learn "this ticker always goes up."
2. Diversity reward shaping: a rolling window of realized trade directions
   per stream is tracked, and `|mean(direction)|` over that window is
   penalized in the reward, discouraging a policy that always fires the same
   sign regardless of context.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    RAW_DIR,
    MIRROR_PROB,
    DIVERSITY_WINDOW,
    DIVERSITY_COEF,
    OVERTRADE_WINDOW,
    OVERTRADE_FREE_TRADES,
    OVERTRADE_SURCHARGE_BPS,
    PLATFORM_FEE_PER_TRADE,
)
from execution_sim import ExecutionSimulator, SimulatedFill  # noqa: E402
from portfolio_state import PortfolioState, Fill  # noqa: E402


def load_aligned_close_prices(tickers, aligned_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Loads raw close prices per ticker from RAW_DIR and reindexes them onto
    `aligned_dates` (the same timestamp index MultiTickerRolloutDataset
    produced), forward-filling gaps the same way dataset.py's feature
    alignment does, so observation windows and price marks line up 1:1.

    Returns: np.ndarray of shape [T, n_tickers], float32.
    """
    closes = {}
    for ticker in tickers:
        path = os.path.join(RAW_DIR, f"{ticker}.parquet")
        df = pd.read_parquet(path, columns=["close"])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        closes[ticker] = df["close"]

    price_df = pd.DataFrame(closes)
    price_df = price_df.reindex(aligned_dates).ffill().bfill()
    return price_df[tickers].values.astype(np.float32)


@dataclass
class StepResult:
    obs: torch.Tensor       # [n_envs, window, features] - next observation window
    reward: torch.Tensor    # [n_envs]
    done: torch.Tensor      # [n_envs] bool
    info: Dict[str, torch.Tensor]


class VecTradingEnv:
    """
    Vectorized single-asset-per-stream trading environment built on top of a
    MultiTickerRolloutDataset. n_envs == dataset.n_envs == number of tickers.
    """

    def __init__(
        self,
        dataset,                       # a MultiTickerRolloutDataset instance (already split train/val/test)
        initial_cash: float = 100_000.0,
        max_position_frac: float = 1.0,   # max notional as a fraction of initial_cash per stream
        tick_size: float = 0.01,
        spread_bps: float = 1.0,
        impact_coef: float = 0.1,
        max_participation: float = 0.1,
        commission_per_share: float = 0.0,
        commission_bps: float = 0.5,
        min_commission: float = 0.0,
        platform_fee_per_trade: float = PLATFORM_FEE_PER_TRADE,
        r_step_scale: float = 0.5,
        hold_loser_penalty: float = 0.0005,
        enable_mirroring: bool = True,
        mirror_prob: float = MIRROR_PROB,
        return_feature_keywords: Tuple[str, ...] = ("ret", "mom"),
        overtrade_window: int = OVERTRADE_WINDOW,       # ~1hr of 5-min bars by default
        overtrade_free_trades: int = OVERTRADE_FREE_TRADES,   # trades allowed in the window before surcharge kicks in
        overtrade_surcharge_bps: float = OVERTRADE_SURCHARGE_BPS,
        bias_window: int = DIVERSITY_WINDOW,
        diversity_bonus_coef: float = DIVERSITY_COEF,
        device: Optional[str] = None,
    ):
        self.dataset = dataset
        self.window_size = dataset.window_size
        self.n_envs = dataset.n_envs  # == n_tickers, one stream per ticker
        self.tickers = dataset.tickers
        self.feature_names = dataset.feature_names
        self.device = torch.device(device) if device is not None else dataset.device

        self.initial_cash = initial_cash
        self.max_position_frac = max_position_frac
        self.r_step_scale = r_step_scale
        self.hold_loser_penalty = hold_loser_penalty

        self.enable_mirroring = enable_mirroring
        self.mirror_prob = mirror_prob
        self.overtrade_window = overtrade_window
        self.overtrade_free_trades = overtrade_free_trades
        self.bias_window = bias_window
        self.diversity_bonus_coef = diversity_bonus_coef

        # --- Raw prices, aligned to the dataset's own date index
        prices_np = load_aligned_close_prices(self.tickers, dataset.aligned_dates)
        self.prices = torch.tensor(prices_np, device=self.device, dtype=torch.float32)  # [T, n_envs]

        # --- Synthetic mirrored price path per ticker: negate log-returns,
        # keep the same starting price, so a mirrored bull ticker behaves like
        # a bear ticker (and vice versa) for the whole pass. Precomputed once
        # since it only depends on the raw price series, not on the episode.
        log_prices = torch.log(self.prices.clamp(min=1e-6))
        log_returns = torch.diff(log_prices, dim=0, prepend=log_prices[:1])
        mirrored_log_prices = torch.cumsum(-log_returns, dim=0) + log_prices[0]
        self.mirrored_prices = torch.exp(mirrored_log_prices)  # [T, n_envs]

        # --- Feature indices whose sign is direction-sensitive (returns,
        # momentum, etc.) — these get flipped in obs for mirrored streams so
        # the observation stays consistent with the mirrored price path.
        # Volatility-style features (e.g. 'rv') are magnitude-based and are
        # deliberately left untouched.
        self._return_feature_idx = [
            i for i, name in enumerate(self.feature_names)
            if any(kw in name.lower() for kw in return_feature_keywords)
        ]

        # --- Portfolio: one row per stream, each managing exactly 1 instrument
        self.portfolio = PortfolioState(
            n_envs=self.n_envs, n_tickers=1, initial_cash=initial_cash, device=str(self.device)
        )

        # --- Execution simulator, shared config across all streams
        self.execution = ExecutionSimulator(
            tick_size=tick_size,
            spread_bps=spread_bps,
            impact_coef=impact_coef,
            max_participation=max_participation,
            commission_per_share=commission_per_share,
            commission_bps=commission_bps,
            min_commission=min_commission,
            platform_fee_per_trade=platform_fee_per_trade,
            overtrade_surcharge_bps=overtrade_surcharge_bps,
            device=str(self.device),
        )

        # rolling ATR-style volatility proxy for step-reward normalization,
        # reuses the dataset's own 'rv' feature if present (already computed
        # in preprocess.py as annualized realized vol) instead of recomputing
        self._rv_feature_idx = self.feature_names.index("rv") if "rv" in self.feature_names else None

        self.current_idx = 0
        self.max_idx = len(self.dataset) - 1
        self.benchmark_start_price = None

        # --- Per-env rolling trade-direction history (circular buffers),
        # used for both the overtrading slippage surcharge and the
        # bias/diversity reward bonus. Allocated here, populated in reset().
        self._trade_hist = torch.zeros((self.n_envs, self.overtrade_window), device=self.device)
        self._trade_hist_ptr = 0
        self._bias_hist = torch.zeros((self.n_envs, self.bias_window), device=self.device)
        self._bias_hist_ptr = 0
        self.mirror_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        self.active_prices = self.prices

    def reset(self) -> torch.Tensor:
        """Resets the episode to the start of the split and returns the first observation."""
        self.current_idx = 0
        self.portfolio.reset()

        # --- Per-env directional mirroring, decided fresh each pass through
        # the dataset (see module docstring). Independent per stream/ticker.
        if self.enable_mirroring:
            self.mirror_mask = torch.rand(self.n_envs, device=self.device) < self.mirror_prob
        else:
            self.mirror_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        self.active_prices = torch.where(self.mirror_mask.unsqueeze(0), self.mirrored_prices, self.prices)

        self._trade_hist.zero_()
        self._trade_hist_ptr = 0
        self._bias_hist.zero_()
        self._bias_hist_ptr = 0

        first_price_row = self.active_prices[self.window_size - 1]  # last bar in first window == "now"
        self.benchmark_start_price = first_price_row.clone()
        obs = self.dataset[self.current_idx].to(self.device)  # [n_envs, window, features]
        return self._apply_mirror_to_obs(obs)

    def _apply_mirror_to_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Flips the sign of direction-sensitive features (returns/momentum) for
        streams whose price path was mirrored this pass, so the observation
        stays consistent with `active_prices`. Magnitude-based features
        (volatility, etc.) are left untouched. No-op if mirroring is off or
        no matching feature names were found.
        """
        if not self.enable_mirroring or not self._return_feature_idx or not self.mirror_mask.any():
            return obs
        obs = obs.clone()
        flip = self.mirror_mask.view(-1, 1)  # broadcast over the window dim
        for f in self._return_feature_idx:
            obs[:, :, f] = torch.where(flip, -obs[:, :, f], obs[:, :, f])
        return obs

    def _current_prices(self) -> torch.Tensor:
        """Mark/mid price for 'now' = last bar of the current window."""
        t = self.current_idx + self.window_size - 1
        return self.active_prices[t]  # [n_envs]

    def _bar_liquidity_proxy(self) -> torch.Tensor:
        """
        Cheap liquidity proxy without needing raw volume loaded separately:
        uses a fixed large notional cap scaled by price so partial fills only
        kick in for outsized orders. If you want true volume-based partial
        fills, pass real bar volume in here instead (see docstring note in
        execution_sim.py: bar_liquidity_proxy is meant to be bar volume).
        """
        return torch.full((self.n_envs,), 1e6, device=self.device, dtype=torch.float32)

    def step(
        self,
        direction: torch.Tensor,      # [n_envs], in {-1, 0, 1}
        size: torch.Tensor,           # [n_envs], requested shares (unsigned)
        limit_offset: torch.Tensor,   # [n_envs], in ticks
    ) -> StepResult:
        """
        Advances every stream by one window-step. Each stream i trades only
        its own instrument i (portfolio ticker index is always 0 within each
        stream's single-asset ledger).
        """
        mid_price = self._current_prices()
        liquidity_proxy = self._bar_liquidity_proxy()

        equity_before = self.portfolio.equity(mid_price.unsqueeze(1))

        overtrading_factor = self._overtrading_factor()

        sim_fill: SimulatedFill = self.execution.simulate_fill(
            direction=direction,
            size=size,
            limit_offset=limit_offset,
            mid_price=mid_price,
            bar_liquidity_proxy=liquidity_proxy,
            overtrading_factor=overtrading_factor,
        )
        commission = self.execution.compute_commission(sim_fill.filled_qty, sim_fill.fill_price)
        platform_fee = self.execution.compute_platform_fee(sim_fill.filled_qty)
        total_fees = commission + platform_fee

        fill = Fill(ticker_idx=0, qty=sim_fill.filled_qty, price=sim_fill.fill_price, commission=total_fees)
        realized_delta = self.portfolio.step_apply(fill)

        # --- update rolling trade-direction history (overtrading + diversity)
        trade_sign = torch.sign(sim_fill.filled_qty)
        self._trade_hist[:, self._trade_hist_ptr % self.overtrade_window] = trade_sign
        self._trade_hist_ptr += 1
        self._bias_hist[:, self._bias_hist_ptr % self.bias_window] = trade_sign
        self._bias_hist_ptr += 1

        self.current_idx += 1
        done_time = self.current_idx >= self.max_idx
        next_mid_price = self._current_prices() if not done_time else mid_price

        equity_after = self.portfolio.equity(next_mid_price.unsqueeze(1))
        drawdown = self.portfolio.update_drawdown_tracking(next_mid_price.unsqueeze(1))

        next_obs = self.dataset[min(self.current_idx, self.max_idx)].to(self.device)
        next_obs = self._apply_mirror_to_obs(next_obs)

        reward, reward_info = self._compute_reward(
            equity_before, equity_after, mid_price, next_mid_price, done_time, next_obs
        )

        done = torch.full((self.n_envs,), done_time, device=self.device, dtype=torch.bool)

        info = {
            "equity": equity_after,
            "realized_delta": realized_delta,
            "cash": self.portfolio.cash.clone(),
            "position": self.portfolio.positions[:, 0].clone(),
            "realized_pnl": self.portfolio.realized_pnl.clone(),
            "drawdown": drawdown,
            "is_partial_fill": sim_fill.is_partial,
            "slippage_bps": sim_fill.slippage_bps,
            "commission": commission,
            "platform_fee": platform_fee,
            "overtrading_factor": overtrading_factor,
            **reward_info,
        }

        return StepResult(obs=next_obs, reward=reward, done=done, info=info)

    def _overtrading_factor(self) -> torch.Tensor:
        """
        Fraction in [0, 1] of how far each stream is over its "free" trade
        budget within the rolling `overtrade_window` (in bars — 5-minute bars
        by default, so overtrade_window=12 is roughly the last hour). 0 means
        at or under the free-trade allowance, 1 means trading every bar.
        Fed into execution_sim's overtrading slippage surcharge.
        """
        recent_trade_count = (self._trade_hist != 0).float().sum(dim=1)
        denom = max(self.overtrade_window - self.overtrade_free_trades, 1)
        factor = (recent_trade_count - self.overtrade_free_trades) / denom
        return factor.clamp(min=0.0, max=1.0)

    def _compute_reward(
        self,
        equity_before: torch.Tensor,
        equity_after: torch.Tensor,
        mid_price_before: torch.Tensor,
        mid_price_after: torch.Tensor,
        is_terminal: bool,
        next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vol-normalized step reward + terminal alpha-vs-buy-and-hold reward +
        a hold-loser penalty + a directional-diversity bonus, matching the
        reward shape described in project notes: position_bar_return /
        current_atr, benchmarked terminal PnL, a small drag on holding an
        underwater position, and a penalty on persistent one-sided direction
        (see module docstring on directional bias mitigation).

        `next_obs` is the already-fetched, already-mirror-adjusted observation
        for `self.current_idx` (passed in from step() so we don't redundantly
        re-slice/re-mirror the dataset here).
        """
        step_pnl = equity_after - equity_before

        if self._rv_feature_idx is not None:
            # next_obs: [n_envs, window, features] -> take last timestep's rv feature per stream
            current_vol = next_obs[:, -1, self._rv_feature_idx].abs().clamp(min=1e-4)
        else:
            current_vol = torch.full_like(step_pnl, 1.0)

        step_reward = self.r_step_scale * (step_pnl / (equity_before.clamp(min=1e-6) * current_vol))

        # Hold-loser penalty: small drag applied when holding a position that's
        # currently underwater, scaled by |position| so bigger losers cost more
        pos = self.portfolio.positions[:, 0]
        unrealized = self.portfolio.unrealized_pnl(mid_price_after.unsqueeze(1))
        is_loser = (pos != 0) & (unrealized < 0)
        hold_penalty = torch.where(
            is_loser, self.hold_loser_penalty * pos.abs(), torch.zeros_like(pos)
        )

        # Directional diversity bonus: penalize streams whose recent trade
        # direction has been persistently one-sided (|mean sign| close to 1),
        # regardless of whether that's because the underlying data (or its
        # mirrored counterpart) happens to be trending. This is a light-touch
        # complement to the mirroring above, not a substitute for it.
        bias_mean = self._bias_hist.mean(dim=1)
        diversity_bonus = -self.diversity_bonus_coef * bias_mean.abs()

        reward = step_reward - hold_penalty + diversity_bonus

        terminal_alpha = torch.zeros_like(step_pnl)
        if is_terminal and self.benchmark_start_price is not None:
            strategy_return = (equity_after - self.initial_cash) / self.initial_cash
            benchmark_return = (mid_price_after - self.benchmark_start_price) / self.benchmark_start_price.clamp(min=1e-6)
            terminal_alpha = strategy_return - benchmark_return
            reward = reward + terminal_alpha

        return reward, {
            "step_pnl": step_pnl,
            "terminal_alpha": terminal_alpha,
            "hold_penalty": hold_penalty,
            "diversity_bonus": diversity_bonus,
        }

    def __len__(self) -> int:
        return len(self.dataset)