"""
live/live_loop.py

Real-time bar polling -> model inference -> risk layer -> order execution
loop, running against the Alpaca paper account by default (or live, if
broker_client.py's AlpacaBrokerClient is constructed with paper=False --
nothing in THIS file changes either way).

Runs the SAME action pipeline as training/ppo_hybrid.py's collect_rollout()
and eval/backtest_report.py's run_backtest() -- rescale -> KellySizer ->
RiskManager -> KillSwitch -> order submission -- except:
    - the policy runs in DETERMINISTIC mode always (act(deterministic=True)).
      PPO's stochastic exploration has no business happening with real
      capital, paper or not.
    - live/reconciliation.py's Reconciler.sync() runs every cycle BEFORE any
      order is placed, and an unreachable/mismatched broker blocks trading
      that cycle regardless of what the kill switch itself decided.

THE BIGGEST GAP THIS FILE CANNOT CLOSE ON ITS OWN: feature engineering.
preprocess.py's feature computation (log returns, vol z-score, whatever
else cnn_encoder.py expects) was never shared when this file was written,
so `feature_builder` below is an INJECTED DEPENDENCY, not an
implementation -- you must supply a callable that turns a rolling window of
raw bars into exactly the same normalized feature tensor preprocess.py
produces offline. Any drift between that callable and the real
preprocess.py (different rolling-window length for a z-score, different
NaN/fill handling, different feature ordering) is train/serve skew, and it
will NOT throw an error -- it will just make the model act on inputs it was
never trained on. This is the live-trading equivalent of
model/hybrid_policy.py's "highest risk file" warning: verify
feature_builder's output against preprocess.py's OFFLINE output on a known
historical window before ever pointing this at a funded account.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from live.broker_client import BrokerAPIError, BrokerClient, OrderRequest, OrderSide, OrderType
from live.reconciliation import Reconciler

from model.hybrid_policy import HybridPolicyHead
from training.ppo_hybrid import HybridActorCritic
from training.config import TrainingConfig

from risk.kelly_sizing import KellySizer
from risk.kill_switch import KillSwitch
from risk.risk_manager import RiskLimits, RiskManager

from portfolio_state import Fill, PortfolioState

from monitoring.dashboard import MetricsWriter


@dataclass
class RawBarWindow:
    """Rolling window of raw OHLCV bars for one ticker -- exactly what feature_builder needs to reproduce preprocess.py's features."""

    timestamps: List
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]


# obs: [n_envs, window, n_features], matching MultiTickerRolloutDataset's
# offline layout -- see module docstring's warning about this being an
# injected dependency, not an implementation.
FeatureBuilder = Callable[[Dict[str, RawBarWindow]], torch.Tensor]


class AlpacaBarPoller:
    """
    Minimal market-data polling: fetches the most recent `window_size`
    completed 5-minute bars per ticker. This is a plain REST poll on a fixed
    interval, NOT a streaming/websocket client -- fine for a 5-minute-bar
    strategy where nothing needs sub-bar latency, and it keeps this file's
    dependency surface small. Swap in Alpaca's streaming API later if
    intra-bar reaction time ever matters.
    """

    def __init__(self, tickers: List[str], window_size: int, paper: bool = True):
        key = os.environ.get("TRADING_ALPACA_PAPER_KEY" if paper else "TRADING_ALPACA_LIVE_KEY")
        secret = os.environ.get("TRADING_ALPACA_PAPER_SECRET" if paper else "TRADING_ALPACA_LIVE_SECRET")
        if not key or not secret:
            raise RuntimeError("AlpacaBarPoller: matching Alpaca API key/secret env vars are not set.")
        self.tickers = tickers
        self.window_size = window_size
        self._client = StockHistoricalDataClient(key, secret)

    def poll(self) -> Dict[str, RawBarWindow]:
        request = StockBarsRequest(
            symbol_or_symbols=self.tickers,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            limit=self.window_size,
        )
        bar_set = self._client.get_stock_bars(request)
        windows: Dict[str, RawBarWindow] = {}
        for ticker in self.tickers:
            bars = bar_set[ticker] if ticker in bar_set.data else []
            windows[ticker] = RawBarWindow(
                timestamps=[b.timestamp for b in bars],
                open=[float(b.open) for b in bars],
                high=[float(b.high) for b in bars],
                low=[float(b.low) for b in bars],
                close=[float(b.close) for b in bars],
                volume=[float(b.volume) for b in bars],
            )
        return windows


class LiveLoop:
    def __init__(
        self,
        broker: BrokerClient,
        bar_poller: AlpacaBarPoller,
        feature_builder: FeatureBuilder,
        actor_critic: HybridActorCritic,
        cfg: TrainingConfig,
        tickers: List[str],
        metrics_writer: Optional[MetricsWriter] = None,
        bar_interval_seconds: float = 300.0,
    ):
        self.broker = broker
        self.bar_poller = bar_poller
        self.feature_builder = feature_builder
        self.actor_critic = actor_critic
        self.cfg = cfg
        self.tickers = tickers
        self.n_envs = len(tickers)
        self.metrics_writer = metrics_writer
        self.bar_interval_seconds = bar_interval_seconds

        self.device = next(actor_critic.parameters()).device

        # Internal ledger -- reconciliation.py compares broker state against
        # THIS object every cycle; live_loop.py updates it optimistically
        # right after each submitted order (see _submit_and_apply()).
        self.portfolio = PortfolioState(
            n_envs=self.n_envs, n_tickers=1, initial_cash=cfg.env.initial_cash, device=str(self.device)
        )

        self.kelly_sizer = KellySizer(
            n_envs=self.n_envs,
            lookback_trades=cfg.risk.kelly_lookback_trades,
            min_trades_for_estimate=cfg.risk.kelly_min_trades_for_estimate,
            kelly_multiplier=cfg.risk.kelly_multiplier,
            kelly_cap=cfg.risk.kelly_cap,
            default_fraction=cfg.risk.kelly_default_fraction,
            device=str(self.device),
        )
        self.risk_manager = RiskManager(
            RiskLimits(
                max_position_frac=cfg.risk.max_position_frac,
                max_gross_exposure_frac=cfg.risk.max_gross_exposure_frac,
                max_ticker_concentration_frac=cfg.risk.max_ticker_concentration_frac,
                max_order_notional_frac=cfg.risk.max_order_notional_frac,
                drawdown_halt_frac=cfg.risk.drawdown_halt_frac,
            ),
            device=str(self.device),
        )
        self.kill_switch = KillSwitch(
            n_envs=self.n_envs,
            daily_loss_limit_frac=cfg.risk.daily_loss_limit_frac,
            broker_error_streak_limit=cfg.risk.broker_error_streak_limit,
            state_mismatch_tolerance=cfg.risk.state_mismatch_tolerance,
            device=str(self.device),
        )
        self.reconciler = Reconciler(
            broker=broker,
            portfolio=self.portfolio,
            kill_switch=self.kill_switch,
            tickers=tickers,
            mismatch_tolerance=cfg.risk.state_mismatch_tolerance,
            device=str(self.device),
        )

        self.hidden = actor_critic.init_hidden(self.n_envs, self.device)
        self._step = 0
        self._day_started = False
        self._total_trades = 0

    def step_once(self) -> None:
        # --- 1. poll bars first -- everything else this cycle needs a price
        # or a feature tensor derived from them.
        raw_windows = self.bar_poller.poll()
        mid_price = self._extract_mid_price(raw_windows)

        # --- 2. reconcile BEFORE placing any order -- a stale/unreachable
        # broker or a state mismatch should block new orders even if
        # nothing else looks wrong yet.
        report = self.reconciler.sync()

        if not self._day_started:
            self.kill_switch.start_new_day(self.portfolio.equity(mid_price.unsqueeze(1)))
            self._day_started = True

        # --- 3. features + inference (deterministic -- see module docstring)
        obs = self.feature_builder(raw_windows).to(self.device)
        equity_before = self.portfolio.equity(mid_price.unsqueeze(1))
        current_position_notional = self.portfolio.positions[:, 0] * mid_price

        with torch.no_grad():
            trunk, self.hidden = self.actor_critic.forward_features(obs, self.hidden)
            action_sample = self.actor_critic.policy_head.act(trunk, deterministic=True)

        size_shares = HybridPolicyHead.rescale_size(
            action_sample.size, torch.full_like(action_sample.size, self.cfg.risk.max_order_shares)
        )
        limit_offset_ticks = HybridPolicyHead.rescale_limit_offset(
            action_sample.limit_offset, self.cfg.risk.max_limit_offset_ticks
        )

        # --- 4. risk pipeline, same order as training/eval
        kelly_result = self.kelly_sizer.apply(
            size=size_shares,
            direction=action_sample.direction,
            mid_price=mid_price,
            equity=equity_before,
            current_position_notional=current_position_notional,
        )
        risk_result = self.risk_manager.apply(
            direction=action_sample.direction,
            size=kelly_result.size,
            limit_offset=limit_offset_ticks,
            mid_price=mid_price,
            portfolio=self.portfolio,
            ticker_idx=0,
        )
        final_direction, final_size = self.kill_switch.apply(risk_result.direction, risk_result.size)

        # --- 5. explicit reconciliation gate -- redundant with the kill
        # switch in most cases (check_state_mismatch/record_broker_error
        # already feed it), but kept as a second, explicit guard rather than
        # relying on that implicitly.
        if not report.broker_reachable or not report.ok:
            final_direction = torch.zeros_like(final_direction)
            final_size = torch.zeros_like(final_size)

        # --- 6. submit orders for any nonzero size, update ledger optimistically
        for i, ticker in enumerate(self.tickers):
            qty = float(final_size[i].item())
            direction = float(final_direction[i].item())
            if qty <= 0 or direction == 0:
                continue
            self._submit_and_apply(i, ticker, direction, qty, float(mid_price[i].item()), float(limit_offset_ticks[i].item()))

        # --- 7. metrics (structured log only -- see monitoring/dashboard.py;
        # this process never renders anything itself)
        if self.metrics_writer is not None:
            equity_after = self.portfolio.equity(mid_price.unsqueeze(1))
            unrealized = self.portfolio.unrealized_pnl(mid_price.unsqueeze(1))
            self.metrics_writer.log(
                step=self._step,
                tickers=self.tickers,
                position=self.portfolio.positions[:, 0].tolist(),
                unrealized_pnl=unrealized.tolist(),
                equity=float(equity_after.sum().item()),      # kept for backward compat
                net_worth=float(equity_after.sum().item()),   # dashboard.py's expected key -- see train.py's matching field
                total_trades=self._total_trades,
                halted=self.kill_switch.is_halted().tolist(),
                broker_reachable=report.broker_reachable,
                mismatched_tickers=report.mismatched_tickers,
            )

        self._step += 1

    def _submit_and_apply(
        self, i: int, ticker: str, direction: float, qty: float, mid_price: float, limit_offset_ticks: float
    ) -> None:
        side = OrderSide.BUY if direction > 0 else OrderSide.SELL
        tick_size = self.cfg.env.tick_size
        limit_price = round(mid_price + direction * limit_offset_ticks * tick_size, 2)
        order = OrderRequest(symbol=ticker, side=side, qty=qty, order_type=OrderType.LIMIT, limit_price=limit_price)

        try:
            result = self.broker.submit_order(order)
        except BrokerAPIError:
            mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            mask[i] = True
            self.kill_switch.record_broker_error(mask)
            return

        # Optimistic ledger update from whatever the broker reports FILLED
        # right now (a limit order may be partial or not filled at all --
        # reconciliation.py's NEXT sync() is what catches drift between this
        # optimistic update and the broker's actual settled state).
        filled_qty = result.filled_qty or 0.0
        if filled_qty == 0.0:
            return
        signed_qty = filled_qty if direction > 0 else -filled_qty
        fill_price = result.filled_avg_price if result.filled_avg_price is not None else mid_price

        qty_vec = torch.zeros(self.n_envs, device=self.device)
        qty_vec[i] = signed_qty
        price_vec = torch.zeros(self.n_envs, device=self.device)
        price_vec[i] = fill_price
        commission_vec = torch.zeros(self.n_envs, device=self.device)

        fill = Fill(ticker_idx=0, qty=qty_vec, price=price_vec, commission=commission_vec)
        self.portfolio.step_apply(fill)
        self._total_trades += 1

    def _extract_mid_price(self, raw_windows: Dict[str, RawBarWindow]) -> torch.Tensor:
        prices = []
        for ticker in self.tickers:
            closes = raw_windows[ticker].close
            if not closes:
                raise RuntimeError(
                    f"AlpacaBarPoller returned no bars for {ticker} -- refusing to trade this "
                    "cycle rather than guess a price. Check market hours / symbol validity."
                )
            prices.append(closes[-1])
        return torch.tensor(prices, device=self.device, dtype=torch.float32)

    def run_forever(self) -> None:
        while True:
            try:
                self.step_once()
            except Exception as e:  # noqa: BLE001 -- a live trading loop must never die silently
                print(f"[live_loop] step_once() raised: {e!r} -- halting all streams as a precaution")
                self.kill_switch.trip_manual(torch.ones(self.n_envs, dtype=torch.bool, device=self.device))
            time.sleep(self.bar_interval_seconds)