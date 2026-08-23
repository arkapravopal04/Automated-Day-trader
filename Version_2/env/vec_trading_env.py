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
from typing import Callable, Dict, Optional, Sequence, Tuple

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
    VOLUME_SCALE,
)
from execution_sim import ExecutionSimulator, SimulatedFill  # noqa: E402
from portfolio_state import PortfolioState, Fill  # noqa: E402

Tensor = torch.Tensor


def _load_aligned_column(
    tickers: Sequence[str],
    aligned_dates: pd.DatetimeIndex,
    column: str,
    fill: Callable[[pd.DataFrame], pd.DataFrame],
) -> np.ndarray:
    """
    Shared loader: reads `column` from each ticker's parquet in RAW_DIR,
    reindexes onto `aligned_dates` (the same timestamp index
    MultiTickerRolloutDataset produced), and applies the caller-supplied
    `fill` strategy for gaps -- the only real difference between the price
    and volume loaders below.

    Returns: np.ndarray of shape [T, n_tickers], float32.
    """
    series_by_ticker = {}
    for ticker in tickers:
        path = os.path.join(RAW_DIR, f"{ticker}.parquet")
        df = pd.read_parquet(path, columns=[column])
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        series_by_ticker[ticker] = df[column]

    # NaN-POISONING FIX: reindex onto the UNION of each ticker's own history
    # and `aligned_dates`, fill there, and only then select the split's dates.
    # Reindexing straight onto `aligned_dates` (which is already sliced to one
    # train/val/test split) discards every bar before the split starts, so
    # ffill() has nothing to carry forward for a ticker whose data ENDS before
    # that split -- the whole column stays NaN even though ffill().bfill()
    # "looks" total.
    #
    # That is not hypothetical: SQ (Block, ticker retired to XYZ) stops at
    # 2025-01-17, while the test split runs 2026-01-21 -> 2026-08-11, so SQ's
    # price column was entirely NaN on both the val and test splits. Because
    # equity is `cash + (positions * prices).sum()` and IEEE gives
    # 0 * NaN = NaN, that one dead ticker made its stream's equity NaN from
    # the very first bar -- and therefore the SUMMED portfolio net worth NaN
    # -- which would have silently turned eval/backtest_report.py's entire
    # go/no-go verdict into garbage. dataset.py's feature path never tripped
    # on it because that path ends in an extra .fillna(0) this one lacks.
    #
    # After this fix such a ticker forward-fills its last real price as a
    # constant for the split: a dead but INERT stream (zero volume already
    # makes it unfillable via load_aligned_volumes) rather than one that
    # poisons every aggregate. `_warn_on_dead_streams` below makes that
    # visible instead of silent.
    frame = pd.DataFrame(series_by_ticker)
    frame = frame.reindex(frame.index.union(aligned_dates))
    frame = fill(frame)
    frame = frame.reindex(aligned_dates)
    return frame[tickers].values.astype(np.float32)


def _warn_on_dead_streams(tickers: Sequence[str], prices: np.ndarray) -> None:
    """
    Prints a warning for any ticker whose price never moves across the split.

    Such a stream is inert -- it contributes no PnL and no gradient signal,
    but it still occupies one of the n_envs rollout slots and dilutes every
    cross-sectional average. Two known causes, both real in this universe:
    a ticker whose history ENDS before the split (SQ, see the fix note in
    _load_aligned_column) and one whose history BEGINS after the split start,
    which dataset.py backward-fills to its first-ever print (COIN, listed
    2021-04-15, against a union grid starting 2020-08-13).
    """
    if prices.size == 0:
        return
    constant = (prices.max(axis=0) - prices.min(axis=0)) == 0
    dead = [tickers[i] for i in np.nonzero(constant)[0]]
    if dead:
        print(
            f"[env] WARNING: {len(dead)}/{len(tickers)} streams have a CONSTANT price across this "
            f"split and are inert (no PnL, no signal, still consuming a rollout slot): {dead}"
        )


def load_aligned_close_prices(tickers: Sequence[str], aligned_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Loads raw close prices per ticker from RAW_DIR and reindexes them onto
    `aligned_dates` (the same timestamp index MultiTickerRolloutDataset
    produced), forward-filling gaps the same way dataset.py's feature
    alignment does, so observation windows and price marks line up 1:1.

    Returns: np.ndarray of shape [T, n_tickers], float32.
    """
    # NON-POSITIVE PRICES ARE CORRUPT DATA, NOT REAL QUOTES. 15 bars across
    # CRM/KO/LOW/T carry close == 0.00 with non-zero volume (e.g. CRM
    # 2021-05-27 20:05 UTC, 179 shares at $0.00), which no venue can print.
    # Left in, a zero price makes that stream's position worth nothing for one
    # bar -- equity collapses to cash and recovers next bar, a pure fabricated
    # PnL swing -- and _precompute_mirrored_prices() reflects around
    # log(clamp(0, min=1e-6)) = -13.8, manufacturing absurd mirrored prices
    # (max observed $3,489 on a universe whose real median is ~$131).
    # Masking to NaN first makes ffill treat them as the missing bars they
    # are, exactly like any other gap.
    #
    # NOTE: this repairs the EXECUTION/marking path only. The same zero bars
    # are still baked into data/processed/*_features.parquet as extreme
    # log-return outliers (z-scores of -15.8 and +57.9 vs a typical +/-5);
    # clearing those needs preprocess.py re-run, which is a separate step.
    return _load_aligned_column(
        tickers, aligned_dates, "close", lambda df: df.mask(df <= 0).ffill().bfill()
    )


def load_aligned_volumes(tickers: Sequence[str], aligned_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Same alignment as load_aligned_close_prices(), but for raw bar volume --
    used as the real liquidity proxy for execution_sim.py's partial-fill
    sizing (see _bar_liquidity_proxy() below), replacing the earlier fixed
    1e6 placeholder.

    Deliberately fillna(0), NOT ffill like price: a missing/halted bar
    genuinely traded zero shares in that window, it did not trade "the same
    volume as the last real bar." Zero volume makes that bar unfillable in
    execution_sim.py, which is the right behavior for a halt, not an
    artifact to paper over. Scaling below preserves that: 0 * anything is
    still 0.

    NOTE: this was only true in the docstring until execution_sim.py's
    ZERO-LIQUIDITY FIX -- that file clamped the proxy to 1e-6 BEFORE the
    fill-size cap, so a zero-volume bar yielded a 1e-07-share fill (charged
    a full ticket fee) instead of no fill. See the comment in its
    simulate_fill() for the measurement. Roughly 11% of cells on the
    aligned union grid are no-bar cells, so this path is common, not rare.

    Volume is then scaled by paths.VOLUME_SCALE to convert single-venue
    (IEX) prints into an approximation of consolidated tape volume -- see
    that constant's comment in paths.py for the measured justification and
    for how to disable it (TRADING_VOLUME_SCALE=1.0, or switch to the SIP
    feed which is already consolidated).

    Returns: np.ndarray of shape [T, n_tickers], float32.
    """
    volumes = _load_aligned_column(tickers, aligned_dates, "volume", lambda df: df.fillna(0.0))
    return (volumes * VOLUME_SCALE).astype(np.float32)


@dataclass
class StepResult:
    obs: Tensor              # [n_envs, window, features] - next observation window
    reward: Tensor           # [n_envs]
    done: Tensor              # [n_envs] bool
    info: Dict[str, Tensor]


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
        return_feature_keywords: Tuple[str, ...] = ("ret", "mom", "vwap", "pres", "resid"),
        cross_sectional_feature_keywords: Tuple[str, ...] = ("resid",),
        overtrade_window: int = OVERTRADE_WINDOW,       # ~1hr of 5-min bars by default
        overtrade_free_trades: int = OVERTRADE_FREE_TRADES,   # trades allowed in the window before surcharge kicks in
        overtrade_surcharge_bps: float = OVERTRADE_SURCHARGE_BPS,
        bias_window: int = DIVERSITY_WINDOW,
        diversity_bonus_coef: float = DIVERSITY_COEF,
        trade_cooldown_bars: int = 0,   # see _apply_trade_cooldown()
        min_hold_bars: int = 0,         # see _apply_min_hold()
        trading_window: "Optional[Tuple[int, int]]" = None,  # see _apply_trading_window()
        flatten_at_session_close: bool = True,  # see _session_close_override()
        flatten_close_bars: int = 1,   # how many trailing bars force flat

        device: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self.window_size = dataset.window_size
        self.n_envs = dataset.n_envs  # == n_tickers, one stream per ticker
        self.tickers = dataset.tickers
        # Market features from the dataset PLUS the portfolio-state channels
        # this env appends to every observation (see
        # _augment_obs_with_portfolio_state). Model construction sites size
        # the network off env.feature_names, so they pick up the extra width
        # automatically -- anything sizing off dataset.feature_names directly
        # would build a network 2 channels too narrow.
        self.market_feature_names = list(dataset.feature_names)
        self.portfolio_feature_names = ["pos_frac", "unrealized_frac"]
        self.feature_names = self.market_feature_names + self.portfolio_feature_names
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
        self.trade_cooldown_bars = int(trade_cooldown_bars)
        self.min_hold_bars = max(0, int(min_hold_bars))
        self.trading_window = (
            None if trading_window is None
            else (int(trading_window[0]), int(trading_window[1]))
        )
        self.diversity_bonus_coef = diversity_bonus_coef

        self._load_market_data()
        self._precompute_mirrored_prices()

        # --- Feature indices whose sign is direction-sensitive (returns,
        # momentum, etc.) — these get flipped in obs for mirrored streams so
        # the observation stays consistent with the mirrored price path.
        # Volatility-style features (e.g. 'rv') are magnitude-based and are
        # deliberately left untouched.
        #
        # Keyword -> feature, as of preprocess.FEATURE_COLUMNS:
        #   "ret"   -> log_ret, log_ret_3/6/12    (flip)
        #   "vwap"  -> vwap_dev                   (flip)
        #   "pres"  -> intrabar_pres              (flip)
        #   "resid" -> xs_resid                   (flip)
        # Deliberately unmatched, and must stay that way: 'rv', 'vol_z',
        # 'time_sin', 'time_cos' are magnitudes or absolute clock position,
        # and flipping them would corrupt mirrored observations rather than
        # correct them. Confirm against the "[env] mirroring will sign-flip"
        # line printed below on every construction.
        # market_feature_names, NOT feature_names: mirroring is applied to the
        # raw dataset observation BEFORE the portfolio channels are appended,
        # so these indices must address the market block only.
        self._return_feature_idx = [
            i for i, name in enumerate(self.market_feature_names)
            if any(kw in name.lower() for kw in return_feature_keywords)
        ]

        # L4 visibility: print exactly which features mirroring will flip, so
        # adding a new direction-sensitive feature (e.g. RSI/MACD) without
        # extending return_feature_keywords shows up immediately instead of
        # silently training on inconsistent mirrored observations.
        # Cross-sectional features cannot be mirrored coherently AT ALL --
        # see _apply_mirror_to_obs(). Zeroed for mirrored streams instead of
        # flipped. Matched by name so a renamed column fails loudly here
        # rather than silently training on a corrupted channel.
        self._cross_sectional_feature_idx = [
            i for i, name in enumerate(self.market_feature_names)
            if any(kw in name.lower() for kw in cross_sectional_feature_keywords)
        ]

        if self._return_feature_idx:
            flipped = [
                self.feature_names[i] for i in self._return_feature_idx
                if i not in self._cross_sectional_feature_idx
            ]
            print(f"[env] mirroring will sign-flip direction-sensitive features: {flipped}")
        if self._cross_sectional_feature_idx:
            zeroed = [self.feature_names[i] for i in self._cross_sectional_feature_idx]
            print(f"[env] mirroring will ZERO cross-sectional features (cannot be mirrored): {zeroed}")
        elif self.enable_mirroring:
            print(
                f"[env] WARNING: mirroring is enabled but NO features matched "
                f"return_feature_keywords={return_feature_keywords} -- mirrored streams "
                "will get flipped prices with unflipped features (inconsistent obs)."
            )

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
        self._rv_feature_idx = (
            self.market_feature_names.index("rv") if "rv" in self.market_feature_names else None
        )

        # REWARD-NORMALIZATION FIX: the 'rv' feature stored in
        # *_features.parquet is a per-ticker Z-SCORE (preprocess.py
        # normalizes every feature with train-split mean/std), not raw
        # annualized vol. Dividing the step reward by the raw z-score
        # (clamped to 1e-4) meant that whenever realized vol sat near its
        # historical mean -- the common case -- the denominator collapsed
        # to ~1e-4 and the step reward exploded by ~4 orders of magnitude.
        # Denormalize back to raw vol using metadata.json's per-ticker
        # constants. 'rv' is a magnitude feature, never sign-flipped by
        # mirroring, so this is mirror-safe.
        self._rv_norm_mean: Optional[Tensor] = None
        self._rv_norm_std: Optional[Tensor] = None
        if self._rv_feature_idx is not None:
            constants = dataset.metadata["norm_constants"]
            self._rv_norm_mean = torch.tensor(
                [constants[t]["mean"]["rv"] for t in self.tickers],
                device=self.device, dtype=torch.float32,
            )
            self._rv_norm_std = torch.tensor(
                [constants[t]["std"]["rv"] for t in self.tickers],
                device=self.device, dtype=torch.float32,
            )

        # --- Session boundaries ---------------------------------------
        # Until this existed the env had NO concept of a trading session. The
        # RTH filter in preprocess.py leaves bar 77 of one day adjacent to
        # bar 0 of the next with nothing marking the seam, so a position open
        # at 15:55 was simply still open at 09:30 -- overnight exposure that
        # nobody chose and no risk control could act on. Measured on 81
        # sessions of adjusted data, the overnight bar carries sigma 171.8 bps
        # against 23.0 bps intraday (7.5x) with a worst case of -3,681 bps
        # against -581 bps: the tail you cannot exit is six times the tail you
        # can. KillSwitch, RiskManager and the drawdown halt are all
        # step-based and none of them run for those 17.5 hours.
        #
        # Derived from real timestamps rather than from a bar counter: a
        # ticker missing a bar, a half-day session, or a holiday would all
        # break a modulo-78 assumption, and dataset.aligned_dates is already
        # sliced to this split.
        dates = getattr(dataset, "aligned_dates", None)
        if dates is not None and len(dates):
            ny = dates.tz_convert("America/New_York") if dates.tz is not None else dates
            sess = np.asarray(ny.normalize().view("int64"))
            last = np.empty(len(sess), dtype=bool)
            last[:-1] = sess[:-1] != sess[1:]
            last[-1] = True          # nothing follows the final bar
            first = np.empty(len(sess), dtype=bool)
            first[0] = True
            first[1:] = sess[1:] != sess[:-1]
            self._is_session_first = torch.as_tensor(first, device=self.device)
            self._n_sessions = int(first.sum())
            # Bar-of-day index (0 == 09:30) from the clock, not a row counter,
            # for the same reason the session masks are: a missing bar, a
            # half-day or a holiday would shift every later label.
            bod = np.asarray(
                ((ny.hour * 60 + ny.minute) - (9 * 60 + 30)) // 5, dtype=np.int16
            )
            self._bar_of_day = torch.as_tensor(bod.astype(np.int64), device=self.device)
            if trading_window is None:
                self._is_tradable_bar = None
            else:
                lo, hi = int(trading_window[0]), int(trading_window[1])
                self._is_tradable_bar = torch.as_tensor(
                    (bod >= lo) & (bod < hi), device=self.device
                )
            # Widen the force-flat window to the trailing N bars of each
            # session, so an oversized position gets more than one
            # attempt at the participation cap.
            wide = last.copy()
            for _s in range(1, max(1, int(flatten_close_bars))):
                wide[:-_s] |= last[_s:]
            self._is_session_last = torch.as_tensor(wide, device=self.device)
            self._session_last_only = torch.as_tensor(last, device=self.device)
        else:
            self._is_session_last = None
            self._session_last_only = None
            self._bar_of_day = None
            self._is_tradable_bar = None
            self._is_session_first = None
            self._n_sessions = 0

        self.flatten_at_session_close = bool(flatten_at_session_close)
        self.flatten_close_bars = max(1, int(flatten_close_bars))
        if self.flatten_at_session_close and self._is_session_last is None:
            raise ValueError(
                "flatten_at_session_close=True but the dataset exposes no "
                "aligned_dates, so session boundaries cannot be located. Pass "
                "flatten_at_session_close=False only if carrying overnight "
                "exposure is a deliberate choice."
            )
        # Set by step(): True on the step that lands on a session's FIRST bar.
        # collect_rollout() uses it to re-anchor KillSwitch's daily-loss
        # reference on real sessions instead of every N rollouts.
        self.session_just_started = False
        self.forced_flatten_count = 0
        # Positions that were STILL open after a session's final bar. A
        # forced close is an order, not a guarantee: execution_sim caps
        # fills at max_participation * bar volume, and a zero-volume bar
        # fills nothing at all. Exempting the close from that cap would
        # be inventing liquidity -- the same class of defect as Session
        # 1's zero-liquidity clamp that manufactured 1e-07-share fills.
        # So the cap stands and the residual is COUNTED instead. At real
        # position sizes (~$740 against bar volumes in the millions of
        # dollars) participation is ~7e-5 and this never binds; if it
        # ever does, raise flatten_close_bars to wind down over more
        # bars rather than uncapping the fill.
        self.residual_overnight_count = 0
        if self.flatten_at_session_close:
            print(f"[env] flatten_at_session_close=True -- positions are closed on "
                  f"the last bar of each of {self._n_sessions} session(s); no "
                  f"overnight exposure is carried")
        else:
            print("[env] flatten_at_session_close=False -- OVERNIGHT EXPOSURE IS "
                  "CARRIED. No step-based risk control runs between sessions.")

        if self.trading_window is not None and self._is_tradable_bar is None:
            raise ValueError(
                "trading_window was set but the dataset exposes no aligned_dates, "
                "so bar-of-day cannot be located."
            )
        if self.trading_window is not None:
            _lo, _hi = self.trading_window
            _n = int(self._is_tradable_bar.sum())
            print("[env] trading_window=[%d,%d) -- new exposure may only be opened"
                  % (_lo, _hi))
            print("      on bars %d-%d of the session (%d of 78 bars, %d rows)."
                  % (_lo, _hi - 1, _hi - _lo, _n))
            print("      Reductions and the session-close flatten are unaffected.")


        self.current_idx = 0
        self.max_idx = len(self.dataset) - 1
        self.benchmark_start_price = None

        # --- Per-env rolling trade-direction history (circular buffers),
        # used for both the overtrading slippage surcharge and the
        # bias/diversity reward bonus. Allocated here, populated in reset().
        # Bars since this stream last INCREASED exposure; seeded above the
        # cooldown so nothing is blocked on the very first bar.
        self._bars_since_open = torch.full(
            (self.n_envs,), float(self.trade_cooldown_bars + 1), device=self.device
        )
        # Bars since this stream last OPENED from flat. Seeded above the
        # threshold so nothing is trapped on the very first bar.
        self._bars_since_entry = torch.full(
            (self.n_envs,), float(self.min_hold_bars + 1), device=self.device
        )
        self._trade_hist = torch.zeros((self.n_envs, self.overtrade_window), device=self.device)
        self._trade_hist_ptr = 0
        self._bias_hist = torch.zeros((self.n_envs, self.bias_window), device=self.device)
        self._bias_hist_ptr = 0
        self.mirror_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        self.active_prices = self.prices

    def _load_market_data(self) -> None:
        """Loads and aligns raw close prices and bar volume onto the dataset's date index."""
        prices_np = load_aligned_close_prices(self.tickers, self.dataset.aligned_dates)
        _warn_on_dead_streams(self.tickers, prices_np)
        if not np.isfinite(prices_np).all():
            bad = [self.tickers[i] for i in np.nonzero(~np.isfinite(prices_np).all(axis=0))[0]]
            raise ValueError(
                f"Non-finite aligned close prices for {bad}. Equity is cash + positions*price and "
                f"0 * NaN = NaN, so a single such column makes the summed portfolio net worth NaN "
                f"and every downstream metric meaningless -- see _load_aligned_column()'s "
                f"NaN-POISONING FIX note. Refusing to build the env."
            )
        self.prices = torch.tensor(prices_np, device=self.device, dtype=torch.float32)  # [T, n_envs]

        # Real liquidity proxy for execution_sim.py's partial-fill sizing
        # (see _bar_liquidity_proxy()), replacing the earlier fixed placeholder.
        volumes_np = load_aligned_volumes(self.tickers, self.dataset.aligned_dates)
        self.volumes = torch.tensor(volumes_np, device=self.device, dtype=torch.float32)  # [T, n_envs]

    def _precompute_mirrored_prices(self) -> None:
        """
        Builds the synthetic mirrored price path per ticker: negate log-
        returns, keep the same starting price, so a mirrored bull ticker
        behaves like a bear ticker (and vice versa) for the whole pass.
        Precomputed once since it only depends on the raw price series, not
        on the episode.
        """
        log_prices = torch.log(self.prices.clamp(min=1e-6))
        log_returns = torch.diff(log_prices, dim=0, prepend=log_prices[:1])
        mirrored_log_prices = torch.cumsum(-log_returns, dim=0) + log_prices[0]
        self.mirrored_prices = torch.exp(mirrored_log_prices)  # [T, n_envs]

    def reset(self) -> Tensor:
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

        self._bars_since_open.fill_(float(self.trade_cooldown_bars + 1))
        self._bars_since_entry.fill_(float(self.min_hold_bars + 1))
        self._trade_hist.zero_()
        # current_idx is back at 0, so the first bar of the split is by
        # definition the first bar of a session -- collect_rollout() should
        # anchor the daily-loss reference there rather than waiting for the
        # next boundary.
        self.session_just_started = True
        self.forced_flatten_count = 0
        self._trade_hist_ptr = 0
        self._bias_hist.zero_()
        self._bias_hist_ptr = 0

        first_price_row = self.active_prices[self.window_size - 1]  # last bar in first window == "now"
        self.benchmark_start_price = first_price_row.clone()
        obs = self.dataset[self.current_idx].to(self.device)  # [n_envs, window, features]
        obs = self._apply_mirror_to_obs(obs)
        return self._augment_obs_with_portfolio_state(obs, first_price_row)

    def _augment_obs_with_portfolio_state(self, obs: Tensor, mark_price: Tensor) -> Tensor:
        """
        Appends the stream's OWN portfolio state to the market observation.

        Until this existed the actor saw ONLY the 8 market features
        preprocess.py produces (log_ret, rv, vol_z, time_sin, time_cos,
        log_ret_3/6/12) -- no position, no unrealized PnL, no equity. That
        made the policy a stateless market-signal function that could not
        perceive its own inventory, with three measured consequences:

          * "go flat" was not expressible. The discrete head's 0 action
            means "the market looks neutral", not "exit my position",
            because the actor did not know a position existed. Over
            4,860 ticks x 100 streams a real run produced 6,043 flips and
            exactly 0 opens and 0 closes, with 99/100 streams never flat
            for even one tick.
          * a persistent 96.1%-long / 2.9%-short occupancy, which is NOT
            explained by mirroring (mirror_prob=0.5 means ~half the streams
            see inverted prices, and every one of the 99 active streams was
            measured taking BOTH signs at some point) -- it is what a
            direction head collapsing toward a constant looks like.
          * dual_critic.py's DualCriticHead.select() conditions the VALUE
            estimate on position sign while the actor could not observe it,
            so the critic was fitting a function of a variable hidden from
            the actor -- consistent with value_loss sitting flat at ~1.0-1.5
            for 50 straight rollouts instead of falling.

        Two channels, both scale-free (so they are unaffected by
        initial_cash) and both broadcast across the window dim, since they
        describe "now" rather than a history:

            pos_frac         signed position notional / equity
            unrealized_frac  unrealized PnL / equity

        Sign convention is deliberately the MIRRORED frame, matching
        _apply_mirror_to_obs(): the ledger trades `active_prices`, so a
        mirrored stream's position sign is already expressed in the same
        frame as its sign-flipped return features. No extra flip here.
        """
        equity = self.portfolio.equity(mark_price.unsqueeze(1)).clamp(min=1e-6)
        position_notional = self.portfolio.positions[:, 0] * mark_price
        unrealized = self.portfolio.unrealized_pnl(mark_price.unsqueeze(1))

        pos_frac = (position_notional / equity).clamp(-10.0, 10.0)
        unrealized_frac = (unrealized / equity).clamp(-10.0, 10.0)

        extra = torch.stack((pos_frac, unrealized_frac), dim=-1)          # [n_envs, 2]
        extra = extra.unsqueeze(1).expand(-1, obs.shape[1], -1)            # [n_envs, window, 2]
        return torch.cat((obs, extra), dim=-1)

    def _session_close_override(self, direction, size):
        """Force every stream flat on the last bar of a trading session.

        This is a genuine forced liquidation, and it is deliberately NOT
        expressed by setting direction to 0. step()'s own note explains why:
        a zero direction arriving at the env has three indistinguishable
        causes -- real policy intent, risk_manager zeroing a dust-clipped
        order, and kill_switch halting the stream -- and an earlier version
        that reinterpreted it as "close" turned every dust rejection and every
        halt into a liquidation. Here the target is stated positively:
        direction = -sign(position), size = |position|. There is nothing to
        misread.

        Applied AFTER _apply_trade_cooldown so the cooldown cannot suppress
        it. That is belt-and-braces -- the cooldown only blocks exposure-
        INCREASING orders and a full close is a pure reduction -- but a forced
        close must not depend on that remaining true.

        It also runs after risk_manager and kill_switch, which live upstream
        in _run_action_pipeline, so min_order_notional cannot reject it as
        dust and a halted stream still gets flattened. Flattening a halted
        stream is the safer reading of "halt": KillSwitch's docstring
        explicitly leaves flatten-vs-freeze to the caller, and freezing a
        halted stream into an overnight gap is the one outcome nobody wants.
        """
        if not self.flatten_at_session_close:
            return direction, size
        t = self.current_idx + self.window_size - 1
        if t >= len(self._is_session_last) or not bool(self._is_session_last[t]):
            return direction, size

        position = self.portfolio.positions[:, 0]
        open_pos = position != 0

        direction = direction.to(device=self.device, dtype=position.dtype).clone()
        size = size.to(device=self.device, dtype=position.dtype).clone()

        # Close what is open...
        direction = torch.where(open_pos, -torch.sign(position), direction)
        size = torch.where(open_pos, position.abs(), size)

        # ...and refuse to open what is flat. Without this second half the
        # window is "wind down", not "wind down and STAY down": a stream fully
        # closed on an early bar of the window sees position == 0 on the next
        # one, takes no override, and the policy simply re-opens it -- so with
        # an even number of bars left the session can END open. Measured
        # before this branch existed, residual carry went the wrong way with
        # window width (47 -> 390 -> 830 for 1 -> 3 -> 6 bars), which is what
        # exposed it.
        flat = ~open_pos
        direction = torch.where(flat, torch.zeros_like(direction), direction)
        size = torch.where(flat, torch.zeros_like(size), size)

        self.forced_flatten_count += int(open_pos.sum().item())
        return direction, size


    @staticmethod
    def _increasing_mask(direction: Tensor, size: Tensor, position: Tensor) -> Tensor:
        """Orders that GROW exposure, as opposed to a pure reduction.

        Opening from flat, adding in the same direction, or flipping (which
        opens fresh opposite exposure). Shared by _apply_trade_cooldown and
        _apply_trading_window so the two cannot drift apart -- they gate the
        same class of order on different clocks (time-since-last-increase vs
        time-of-day).
        """
        opening = (position == 0) & (direction != 0)
        adding = (torch.sign(direction) == torch.sign(position)) & (position != 0)
        flipping = (
            (torch.sign(direction) == -torch.sign(position))
            & (position != 0)
            & (size > position.abs())
        )
        return opening | adding | flipping

    def _apply_trade_cooldown(self, direction: Tensor, size: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Blocks EXPOSURE-INCREASING orders for `trade_cooldown_bars` bars after
        the stream last increased exposure. Closes and pure reduces always
        pass, so this can never trap a position.

        Why this exists: with a flat action finally reachable
        (training/ppo_hybrid.py's _apply_flat_intent) a 51-rollout run produced
        2,506 opens and 2,507 closes but a MEDIAN TIME FLAT OF 1 BAR -- the
        policy closed and re-entered five minutes later, churning through flat
        rather than resting in it. Flat occupancy barely moved (1.0% -> 1.9%)
        while fills went 2,809 -> 10,156, ticket cost 15.3 -> 27.4 bps, and
        gross edge collapsed 11.20 -> 4.36 bps.

        The economics this targets: the platform fee is FLAT per trade, so
        cost-per-trade is fixed while edge-per-trade grows with holding
        period. At ~$800 orders a round trip costs ~17 bps against a ~5.6 bps
        median 5-minute move, so a position must be held on the order of tens
        of bars before its expected move can clear its own ticket. Capping
        re-entry frequency is what makes edge-per-trade able to exceed
        cost-per-trade at this account size at all -- no sizing knob can,
        since even full Kelly on $10k caps an order near $1,300.

        Deliberately a hard env-level constraint rather than another reward
        penalty: the previous two runs showed the policy optimizing the reward
        well while still churning, so this removes the option instead of
        pricing it.
        """
        if self.trade_cooldown_bars <= 0:
            return direction, size
        position = self.portfolio.positions[:, 0]
        direction = direction.to(device=self.device, dtype=position.dtype)
        size = size.to(device=self.device, dtype=position.dtype)

        increasing = self._increasing_mask(direction, size, position)

        blocked = increasing & (self._bars_since_open < self.trade_cooldown_bars)
        direction = torch.where(blocked, torch.zeros_like(direction), direction)
        size = torch.where(blocked, torch.zeros_like(size), size)
        return direction, size

    def _apply_trading_window(self, direction, size):
        """Allow new exposure only on bars inside the configured window.

        Gates ENTRY, never exit. A position opened inside the window can always
        be reduced or closed outside it -- gating exits too would trap
        inventory the moment the window ended, and would fight both
        _apply_min_hold and the session-close flatten.

        WHY. Median |5-min move| by regime, measured on the marking price path:
        open hour 18.6 bps, close ramp 11.9, midday 9.6, against a round-trip
        cost of ~7.3 bps that does not vary with time of day. The policy was
        spreading its cost budget uniformly over all 77 tradable bars, so most
        of it was spent in the stretch where the available move barely clears
        the spread. time_sin/time_cos have been in the feature set the whole
        time and the policy demonstrably did not use them, so this removes the
        option rather than pricing it -- the same call made for
        trade_cooldown_bars.

        Set this to the regime of whichever alpha_lab cell passed:
        open_hour = (0, 12), midday = (12, 72), close_ramp = (74, 77).
        None means every bar is tradable, which is the previous behaviour.
        """
        if self._is_tradable_bar is None:
            return direction, size
        t = self.current_idx + self.window_size - 1
        if t < len(self._is_tradable_bar) and bool(self._is_tradable_bar[t]):
            return direction, size

        position = self.portfolio.positions[:, 0]
        direction = direction.to(device=self.device, dtype=position.dtype)
        size = size.to(device=self.device, dtype=position.dtype)
        increasing = self._increasing_mask(direction, size, position)
        direction = torch.where(increasing, torch.zeros_like(direction), direction)
        size = torch.where(increasing, torch.zeros_like(size), size)
        return direction, size


    def _apply_min_hold(self, direction, size):
        """Block exposure-REDUCING orders for min_hold_bars after an entry.

        The exact complement of _apply_trade_cooldown, and the mechanism that
        was missing. trade_cooldown_bars=12 blocks re-ENTRY for 12 bars, which
        cuts trade FREQUENCY but says nothing about holding period -- so the
        policy's optimal response was to open, close on the very next bar, and
        sit out the cooldown. Measured on the last run's tick log: 140 of 140
        completed round trips lasted exactly 1 bar. Median, mean, p90 and max
        all 1.0, not a single 2-bar hold.

        That is the gap between the alpha gate and the equity curve. The gate
        found its edge at 30min-1hr horizons (6-12 bars); a 1-bar hold captures
        roughly one bar of that move -- median 9.84 bps -- while paying the
        full 7.28 bps round trip every time. Holding 12 bars scales the move by
        sqrt(12) to ~34 bps against the same fixed cost. Cutting frequency was
        the wrong lever; holding is the right one.

        A hard env constraint rather than a reward term, following the
        precedent set for trade_cooldown_bars: two earlier runs optimised the
        reward well while still churning, so the option is removed rather than
        priced.

        SAFETY. This runs BEFORE _session_close_override, so the forced
        session-close flatten always wins and no position can be trapped past
        the bell. Within a session it can delay a risk_manager-mandated
        reduction by up to min_hold_bars, because by the time an action reaches
        the env a reduction's origin is no longer distinguishable (the same
        ambiguity step() documents for direction == 0). That exposure is
        bounded -- at most min_hold_bars, and always released at the close --
        but it is real, so keep min_hold_bars well under a session.
        """
        if self.min_hold_bars <= 0:
            return direction, size
        position = self.portfolio.positions[:, 0]
        direction = direction.to(device=self.device, dtype=position.dtype)
        size = size.to(device=self.device, dtype=position.dtype)

        # "Reducing" = trading against an existing position. Opening from flat
        # and adding in the same direction are untouched; the cooldown governs
        # those.
        holding = position != 0
        against = torch.sign(direction) == -torch.sign(position)
        reducing = holding & against & (direction != 0)

        blocked = reducing & (self._bars_since_entry < self.min_hold_bars)
        direction = torch.where(blocked, torch.zeros_like(direction), direction)
        size = torch.where(blocked, torch.zeros_like(size), size)
        return direction, size

    def _advance_min_hold(self, positions_before: Tensor) -> None:
        """Ticks the hold clock, resetting it wherever a position opened from flat."""
        if self.min_hold_bars <= 0:
            return
        after = self.portfolio.positions[:, 0]
        opened = (positions_before == 0) & (after != 0)
        self._bars_since_entry = self._bars_since_entry + 1.0
        self._bars_since_entry = torch.where(
            opened, torch.zeros_like(self._bars_since_entry), self._bars_since_entry
        )


    def _advance_trade_cooldown(self, positions_before: Tensor, filled_qty: Tensor) -> None:
        """Ticks the cooldown clock, resetting it wherever exposure actually grew."""
        if self.trade_cooldown_bars <= 0:
            return
        after = self.portfolio.positions[:, 0]
        grew = (filled_qty != 0) & (after.abs() > positions_before.abs())
        self._bars_since_open = self._bars_since_open + 1.0
        self._bars_since_open = torch.where(
            grew, torch.zeros_like(self._bars_since_open), self._bars_since_open
        )

    def _apply_mirror_to_obs(self, obs: Tensor) -> Tensor:
        """
        Flips the sign of direction-sensitive features (returns/momentum) for
        streams whose price path was mirrored this pass, so the observation
        stays consistent with `active_prices`. Magnitude-based features
        (volatility, etc.) are left untouched. No-op if mirroring is off or
        no matching feature names were found.

        CROSS-SECTIONAL FEATURES ARE ZEROED, NOT FLIPPED. A residual is
        `own_return - market_return`. Mirroring inverts a stream's own price
        path, but the market return is a fact about the real universe and
        does not invert with it, so for a mirrored stream:

            correct:      (-own) - market
            if flipped:  -(own  - market) = -own + market   ->  error 2*market
            if left:      (own  - market)                   ->  error 2*own

        Neither is right, because a mirrored stream is a synthetic asset with
        no place in the real cross-section. Measured on 40 tickers of
        regular-hours bars: std(xs_resid)=0.0019 against std(2*market)=0.0025
        -- the flip error is 1.30 residual-sigmas, i.e. LARGER than the
        signal, and systematically anti-correlated with it. With
        MIRROR_PROB=0.5 that corrupted roughly half of all streams in the
        151-rollout run of 2026-08-19, whose win rate came in at 0.29 --
        materially BELOW chance, which is the signature of an inverted input
        rather than merely an unprofitable one.

        Zeroing costs the mirrored half of the batch this one channel (error
        = |signal| = 1.0 sigma, less than either alternative) while leaving
        the unmirrored half fully intact and correct.
        """
        if not self.enable_mirroring or not self.mirror_mask.any():
            return obs
        if not self._return_feature_idx and not self._cross_sectional_feature_idx:
            return obs
        obs = obs.clone()
        flip = self.mirror_mask.view(-1, 1)  # broadcast over the window dim
        for f in self._return_feature_idx:
            if f in self._cross_sectional_feature_idx:
                continue  # zeroed below -- flipping it is worse than useless
            obs[:, :, f] = torch.where(flip, -obs[:, :, f], obs[:, :, f])
        for f in self._cross_sectional_feature_idx:
            obs[:, :, f] = torch.where(flip, torch.zeros_like(obs[:, :, f]), obs[:, :, f])
        return obs

    def _current_prices(self) -> Tensor:
        """Mark/mid price for 'now' = last bar of the current window."""
        t = self.current_idx + self.window_size - 1
        return self.active_prices[t]  # [n_envs]

    def _bar_liquidity_proxy(self) -> Tensor:
        """
        Real per-bar liquidity proxy: this stream's actual traded volume for
        the current bar (same time index _current_prices() uses), fed into
        execution_sim.py's max_participation cap for genuine volume-based
        partial fills. A zero-volume bar (halt/gap -- see
        load_aligned_volumes()) correctly yields zero fillable shares that
        step via execution_sim's max_fillable = max_participation * proxy.
        """
        t = self.current_idx + self.window_size - 1
        return self.volumes[t]  # [n_envs]

    def step(
        self,
        direction: Tensor,      # [n_envs], in {-1, 0, 1}
        size: Tensor,           # [n_envs], requested shares (unsigned)
        limit_offset: Tensor,   # [n_envs], in ticks
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

        # NOTE on direction == 0 (do NOT reinterpret it here as "close the
        # position"): by the time an action reaches this method, a zero
        # direction has THREE indistinguishable causes, only one of which is
        # the policy's own intent --
        #   1. the policy's discrete head genuinely sampled "flat";
        #   2. risk_manager.py zeroed it (`direction = where(size == 0, 0,
        #      direction)`) because a cap or the min_order_notional dust gate
        #      clipped size to 0;
        #   3. kill_switch.py zeroed it because that stream is halted --
        #      whose docstring explicitly leaves "flatten vs. merely freeze"
        #      to the caller.
        # Treating all three as a full close turns every dust rejection,
        # every drawdown-halt reduce_only block, and every KillSwitch halt
        # into a forced liquidation. Any change to flat-semantics has to be
        # made where raw policy intent is still known (see
        # training/ppo_hybrid.py's _run_action_pipeline) -- and note that the
        # actor cannot currently observe its own position at all, so a
        # sampled "flat" means "the market looks neutral", not "exit my
        # position". Position state is now fed to the actor via
        # _augment_obs_with_portfolio_state() below, which is what makes
        # inventory intent representable in the first place.
        direction, size = self._apply_trading_window(direction, size)
        direction, size = self._apply_trade_cooldown(direction, size)
        direction, size = self._apply_min_hold(direction, size)
        direction, size = self._session_close_override(direction, size)

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
        positions_before = self.portfolio.positions[:, 0].clone()
        realized_delta = self.portfolio.step_apply(fill)

        # L2 fix: whether ANY part of an existing position was CLOSED this
        # step (position magnitude shrank, including full closes and the
        # closing leg of a flip) -- distinct from realized_delta == 0, which
        # is also what a break-even close produces. KellySizer uses this to
        # count break-even round trips in its win-rate estimate instead of
        # silently dropping them (see kelly_sizing.py's record_realized_pnl).
        closed_trade = positions_before.abs() > self.portfolio.positions[:, 0].abs()

        self._record_trade_direction(sim_fill.filled_qty)
        self._advance_trade_cooldown(positions_before, sim_fill.filled_qty)
        self._advance_min_hold(positions_before)

        # Residual overnight carry: anything still open after this bar, when
        # this bar is a session's LAST. Must be read BEFORE current_idx
        # advances -- computing it after made t the NEXT bar, which counted
        # positions one bar early, i.e. before the forced close had run.
        # That reported 1,188 carries where the true number was 10.
        if self._session_last_only is not None and self.flatten_at_session_close:
            _tc = self.current_idx + self.window_size - 1
            if _tc < len(self._session_last_only) and bool(self._session_last_only[_tc]):
                self.residual_overnight_count += int(
                    (self.portfolio.positions[:, 0] != 0).sum().item()
                )
        self.current_idx = min(self.current_idx + 1, self.max_idx)
        # Flag the NEW bar, so collect_rollout() can re-anchor the
        # KillSwitch daily-loss reference on a real session boundary.
        if self._is_session_first is not None:
            _t = self.current_idx + self.window_size - 1
            self.session_just_started = (
                _t < len(self._is_session_first)
                and bool(self._is_session_first[_t])
            )
        done_time = self.current_idx >= self.max_idx
        next_mid_price = self._current_prices() if not done_time else mid_price

        equity_after = self.portfolio.equity(next_mid_price.unsqueeze(1))
        drawdown = self.portfolio.update_drawdown_tracking(next_mid_price.unsqueeze(1))

        next_obs = self.dataset[min(self.current_idx, self.max_idx)].to(self.device)
        next_obs = self._apply_mirror_to_obs(next_obs)

        reward, reward_info = self._compute_reward(
            equity_before, equity_after, mid_price, next_mid_price, done_time, next_obs
        )

        # Portfolio channels are appended AFTER _compute_reward, which indexes
        # next_obs by self._rv_feature_idx -- appending at the end of the
        # feature dim leaves every existing index valid, but keeping the
        # reward path on the pure market obs makes that independence explicit.
        # Reflects post-fill position, so the next action is conditioned on
        # the inventory it will actually be acting on.
        next_obs = self._augment_obs_with_portfolio_state(next_obs, next_mid_price)

        done = torch.full((self.n_envs,), done_time, device=self.device, dtype=torch.bool)

        info = {
            "equity": equity_after,
            "realized_delta": realized_delta,
            "filled_qty": sim_fill.filled_qty.clone(),
            "cash": self.portfolio.cash.clone(),
            "position": self.portfolio.positions[:, 0].clone(),
            "realized_pnl": self.portfolio.realized_pnl.clone(),
            "drawdown": drawdown,
            "is_partial_fill": sim_fill.is_partial,
            "slippage_bps": sim_fill.slippage_bps,
            "commission": commission,
            "platform_fee": platform_fee,
            "overtrading_factor": overtrading_factor,
            "closed_trade": closed_trade,
            **reward_info,
        }

        return StepResult(obs=next_obs, reward=reward, done=done, info=info)

    def _record_trade_direction(self, filled_qty: Tensor) -> None:
        """Pushes this step's realized trade sign into both rolling circular buffers."""
        trade_sign = torch.sign(filled_qty)
        self._trade_hist[:, self._trade_hist_ptr % self.overtrade_window] = trade_sign
        self._trade_hist_ptr += 1
        self._bias_hist[:, self._bias_hist_ptr % self.bias_window] = trade_sign
        self._bias_hist_ptr += 1

    def _overtrading_factor(self) -> Tensor:
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
        equity_before: Tensor,
        equity_after: Tensor,
        mid_price_before: Tensor,
        mid_price_after: Tensor,
        is_terminal: bool,
        next_obs: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
        step_reward = self._vol_normalized_step_reward(step_pnl, equity_before, next_obs)
        hold_penalty = self._hold_loser_penalty(mid_price_after)
        diversity_bonus = self._diversity_bonus()

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

    def _vol_normalized_step_reward(self, step_pnl: Tensor, equity_before: Tensor, next_obs: Tensor) -> Tensor:
        """position_bar_return / current_atr, using the dataset's 'rv' feature as the vol proxy when available."""
        if self._rv_feature_idx is not None and self._rv_norm_std is not None:
            # next_obs: [n_envs, window, features] -> take last timestep's rv
            # feature per stream, DENORMALIZED back to raw annualized vol
            # (see __init__'s REWARD-NORMALIZATION FIX comment).
            rv_z = next_obs[:, -1, self._rv_feature_idx]
            raw_rv = (rv_z * self._rv_norm_std + self._rv_norm_mean).abs().clamp(min=1e-4)
            current_vol = raw_rv
        else:
            current_vol = torch.full_like(step_pnl, 1.0)
        # REWARD-EXPLOSION DEFENSE (same class as ppo_hybrid.py's step_return
        # clamp): a near-zero-equity stream turns step_pnl / (1e-6 * vol) into
        # an astronomically large ratio. Clamping the ratio to [-1, 1] is free
        # in practice (real vol-normalized bar moves are << 1) and bounds the
        # env reward path used when RewardConfig.raw_weight > 0.
        ratio = (step_pnl / (equity_before.clamp(min=1e-6) * current_vol)).clamp(-1.0, 1.0)
        return self.r_step_scale * ratio

    def _hold_loser_penalty(self, mid_price_after: Tensor) -> Tensor:
        """Small drag applied when holding a position that's currently underwater, scaled by |position|."""
        pos = self.portfolio.positions[:, 0]
        unrealized = self.portfolio.unrealized_pnl(mid_price_after.unsqueeze(1))
        is_loser = (pos != 0) & (unrealized < 0)
        return torch.where(is_loser, self.hold_loser_penalty * pos.abs(), torch.zeros_like(pos))

    def _diversity_bonus(self) -> Tensor:
        """
        Penalizes streams whose recent trade direction has been persistently
        one-sided (|mean sign| close to 1), regardless of whether that's
        because the underlying data (or its mirrored counterpart) happens to
        be trending. A light-touch complement to mirroring, not a substitute.
        """
        bias_mean = self._bias_hist.mean(dim=1)
        return -self.diversity_bonus_coef * bias_mean.abs()

    def __len__(self) -> int:
        return len(self.dataset)