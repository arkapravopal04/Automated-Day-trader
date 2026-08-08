"""
live/reconciliation.py

Syncs internal position/state tracking (PortfolioState, one row per
ticker/stream -- see portfolio_state.py) against the broker's actual
reported state, handles API hiccups via broker_client.py's retry wrapper,
and is the ONLY place that wires risk/kill_switch.py's
check_state_mismatch() / record_broker_error() / record_broker_success() to
something real. Everywhere else in this project, KillSwitch was fed
backtest-simulated conditions (see eval/backtest_report.py); this file is
where it starts meaning something in the real world.

A state mismatch here means "we can no longer trust our own book" -- a
DIFFERENT failure mode from a market-risk breach (risk_manager.py) or a
PnL-based halt (KillSwitch.check_daily_loss()). See kill_switch.py's module
docstring. sync() should be called every live_loop.py cycle, not just
reactively after an error.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from live.broker_client import BrokerAPIError, BrokerClient
from portfolio_state import PortfolioState
from risk.kill_switch import KillSwitch


@dataclass
class ReconciliationReport:
    ok: bool
    broker_reachable: bool
    mismatched_tickers: List[str]
    internal_positions: Dict[str, float]
    broker_positions: Dict[str, float]


class Reconciler:
    def __init__(
        self,
        broker: BrokerClient,
        portfolio: PortfolioState,
        kill_switch: KillSwitch,
        tickers: List[str],
        mismatch_tolerance: float = 1e-3,
        device: Optional[str] = None,
    ):
        """
        portfolio: the SAME PortfolioState instance live_loop.py updates
            optimistically after each order -- this class only READS it
            (torch.sign / .positions) via sync(); it never mutates it as a
            side effect of sync(). Correcting a mismatched ledger is a
            separate, explicit call (reconcile_and_correct() below), not
            something that happens implicitly during a routine sync.
        tickers: MUST be in the same column order as portfolio's ticker
            axis (i.e. env.tickers from vec_trading_env.py / this project's
            convention) -- positions are compared index-by-index against
            this list.
        """
        self.broker = broker
        self.portfolio = portfolio
        self.kill_switch = kill_switch
        self.tickers = tickers
        self.mismatch_tolerance = mismatch_tolerance
        self.device = torch.device(device) if device is not None else portfolio.device

    def sync(self) -> ReconciliationReport:
        """
        Pulls broker positions, compares against the internal ledger, and
        updates the kill switch accordingly. Call this every live_loop.py
        cycle BEFORE deciding whether to place new orders -- an unreachable
        or mismatched broker connection should block new orders even if
        nothing else looks wrong yet.
        """
        all_true = torch.ones(len(self.tickers), dtype=torch.bool, device=self.device)

        try:
            broker_positions_raw = self.broker.get_all_positions()
        except BrokerAPIError:
            # A dead/unreachable connection is exactly what the broker
            # error-streak counter exists for -- feed all streams, since we
            # have no per-ticker information when the WHOLE call failed.
            self.kill_switch.record_broker_error(all_true)
            return ReconciliationReport(
                ok=False,
                broker_reachable=False,
                mismatched_tickers=[],
                internal_positions={},
                broker_positions={},
            )

        internal_qty = self.portfolio.positions[:, 0]  # [n_envs] -- one ticker per stream, see vec_trading_env.py
        broker_qty = torch.tensor(
            [broker_positions_raw[t].qty if t in broker_positions_raw else 0.0 for t in self.tickers],
            device=self.device,
            dtype=torch.float32,
        )

        self.kill_switch.check_state_mismatch(internal_qty, broker_qty)
        self.kill_switch.record_broker_success(all_true)

        mismatch_mask = (internal_qty - broker_qty).abs() > self.mismatch_tolerance
        mismatched_tickers = [t for t, m in zip(self.tickers, mismatch_mask.tolist()) if m]

        return ReconciliationReport(
            ok=len(mismatched_tickers) == 0,
            broker_reachable=True,
            mismatched_tickers=mismatched_tickers,
            internal_positions={t: float(q) for t, q in zip(self.tickers, internal_qty.tolist())},
            broker_positions={t: float(q) for t, q in zip(self.tickers, broker_qty.tolist())},
        )

    def reconcile_and_correct(self, report: ReconciliationReport) -> None:
        """
        Explicit, separate step: overwrites the internal ledger's POSITION
        ONLY (not cash, not avg_entry_price -- the broker's position report
        doesn't give us enough to reconstruct either of those correctly) for
        every mismatched ticker, to match the broker's reported qty.

        Deliberately does NOT clear the kill switch halt that
        check_state_mismatch() would have triggered during sync() --
        correcting the ledger and deciding it's safe to resume trading are
        two different decisions. Call kill_switch.reset() yourself once
        you've actually reviewed why the mismatch happened.
        """
        if report.ok:
            return
        for i, ticker in enumerate(self.tickers):
            if ticker in report.mismatched_tickers:
                self.portfolio.positions[i, 0] = report.broker_positions[ticker]