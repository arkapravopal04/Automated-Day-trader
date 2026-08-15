"""
upstox_paper_broker.py — paper trading broker for Upstox.

Uses Upstox's official sandbox environment for order simulation
(no real money, no real market impact).

Mirrors paper_broker.py's interface exactly so live_runner.py
works with both Binance and Upstox by just swapping the import.

Sandbox docs:
    https://upstox.com/developer/api-documentation/sandbox
"""

import csv
import os
import time
import numpy as np
from datetime import datetime, timezone

import upstox_client
from upstox_client.rest import ApiException

from live_data import ACCESS_TOKEN, _ticker_to_key, INSTRUMENT_KEYS

# ── Realistic NSE intraday friction ───────────────────────────────────────────
TAKER_FEE  = 0.0003   # 0.03% (Upstox intraday equity brokerage ~₹10/order,
                       # approximated as % for the simulator)
SLIPPAGE   = 0.0002   # 0.02% — NSE is more liquid than crypto, tighter spread
STT        = 0.00025  # 0.025% Securities Transaction Tax on sell side
STAMP_DUTY = 0.00003  # 0.003% stamp duty on buy side


def _get_sandbox_config() -> upstox_client.Configuration:
    config = upstox_client.Configuration(sandbox=True)
    config.access_token = ACCESS_TOKEN
    return config


class Fill:
    __slots__ = ("timestamp", "symbol", "side", "price", "quantity",
                 "fee", "pnl", "balance_after")

    def __init__(self, timestamp, symbol, side, price,
                 quantity, fee, pnl, balance_after):
        self.timestamp    = timestamp
        self.symbol       = symbol
        self.side         = side
        self.price        = price
        self.quantity     = quantity
        self.fee          = fee
        self.pnl          = pnl
        self.balance_after= balance_after


class UpstoxPaperBroker:
    """
    Paper trading broker for Upstox / NSE equities.

    Mirrors PaperBroker from paper_broker.py — same interface,
    same portfolio model as env.py. NSE-specific friction applied.

    Note: NSE equities are long-only (no shorting in CNC).
          Short signals are treated as close/flat signals.
          Use F&O instruments if you want short exposure.

    Usage
    -----
        broker = UpstoxPaperBroker(initial_balance=50000.0, ticker="RELIANCE")
        broker.execute(action, current_price)
        print(broker.net_worth(current_price))
    """

    def __init__(self, initial_balance: float = 50000.0,
                 ticker: str = "RELIANCE",
                 log_path: str = "paper_trades_upstox.csv",
                 trade_threshold: float = 0.515,
                 neutral_zone: float = 0.10,
                 cooldown_bars: int = 8,
                 max_drawdown_limit: float = 0.20):

        self.ticker            = ticker
        self.instrument_key    = _ticker_to_key(ticker)
        self.initial_balance   = float(initial_balance)
        self.balance           = float(initial_balance)
        self.position          = 0.0   # ₹ value of open position
        self.entry_price       = 0.0
        self.shares_held       = 0     # integer share count (NSE is lot-based)
        self.cooldown          = 0

        self.trade_threshold   = trade_threshold
        self.neutral_zone      = neutral_zone
        self.cooldown_bars     = cooldown_bars
        self.max_drawdown_limit= max_drawdown_limit

        self.peak_balance      = float(initial_balance)
        self.n_trades          = 0
        self.winning_trades    = 0
        self.fills: list[Fill] = []
        self.log_path          = log_path

        # optional: sandbox order API for logging orders to Upstox sandbox
        self._sandbox_api = None
        try:
            cfg = _get_sandbox_config()
            self._sandbox_api = upstox_client.OrderApiV3(
                upstox_client.ApiClient(cfg)
            )
            print("[UpstoxPaperBroker] sandbox order API connected")
        except Exception as e:
            print(f"[UpstoxPaperBroker] sandbox API unavailable ({e}) "
                  f"— running fully local simulation")

        self._init_log()

    # ── portfolio ─────────────────────────────────────────────────────────────

    def net_worth(self, current_price: float) -> float:
        if self.position == 0 or self.entry_price == 0:
            return self.balance
        # long only: position value scales with price
        pos_value = self.position * (current_price / self.entry_price)
        return self.balance + pos_value

    def drawdown(self, current_price: float) -> float:
        nw = self.net_worth(current_price)
        self.peak_balance = max(self.peak_balance, nw)
        return (self.peak_balance - nw) / (self.peak_balance + 1e-8)

    def win_rate(self) -> float:
        return self.winning_trades / self.n_trades if self.n_trades > 0 else 0.0

    # ── core ──────────────────────────────────────────────────────────────────

    def execute(self, action: np.ndarray, current_price: float) -> dict:
        """
        Execute one agent action at current_price.

        NSE note: negative direction (short) is treated as SELL/flat
        since intraday equity shorting requires F&O or specific MIS products.
        Direction < 0 → close any long position and stay flat.
        """
        direction = float(action[0])
        size      = float(np.clip(action[1], 0.0, 1.0))

        # hard stop
        if self.drawdown(current_price) >= self.max_drawdown_limit:
            if self.position > 0:
                self._close_position(current_price, reason="max_drawdown")
            return self._status(current_price, trade_occurred=False)

        result = {"trade_occurred": False, "side": None, "pnl": 0.0}

        should_close = (
            self.position > 0 and self.entry_price > 0 and (
                abs(direction) < self.neutral_zone or
                direction < -self.trade_threshold   # short signal → go flat
            )
        )

        if should_close:
            pnl = self._close_position(current_price, reason="signal")
            result.update({"trade_occurred": True, "side": "SELL", "pnl": pnl})

        # only go long (direction > threshold)
        if (self.cooldown == 0 and self.position == 0 and
                direction > self.trade_threshold and size > 0.01):
            self._open_long(direction, size, current_price)
            result.update({"trade_occurred": True, "side": "BUY"})

        if self.cooldown > 0:
            self.cooldown -= 1

        return self._status(current_price, **result)

    # ── internals ─────────────────────────────────────────────────────────────

    def _open_long(self, direction: float, size: float, current_price: float):
        investment = self.balance * size
        if investment < 100.0:   # minimum ₹100
            return

        # slippage: buy slightly higher
        fill_price = current_price * (1 + SLIPPAGE)

        # shares: floor to integer
        shares = int(investment / fill_price)
        if shares < 1:
            return

        actual_investment = shares * fill_price
        fee  = actual_investment * (TAKER_FEE + STAMP_DUTY)
        cost = actual_investment + fee

        if cost > self.balance:
            shares = int((self.balance * (1 - TAKER_FEE - STAMP_DUTY)) / fill_price)
            if shares < 1:
                return
            actual_investment = shares * fill_price
            fee  = actual_investment * (TAKER_FEE + STAMP_DUTY)
            cost = actual_investment + fee

        self.balance     -= cost
        self.entry_price  = fill_price
        self.position     = actual_investment   # ₹ value at entry
        self.shares_held  = shares
        self.cooldown     = self.cooldown_bars

        self._place_sandbox_order("BUY", shares, fill_price)

        fill = Fill(_now(), self.ticker, "BUY", fill_price,
                    shares, fee, 0.0, self.balance)
        self.fills.append(fill)
        self._log_fill(fill)
        print(f"[broker] BUY   {shares} shares @ ₹{fill_price:.2f}"
              f"  cost=₹{cost:.2f}  fee=₹{fee:.2f}")

    def _close_position(self, current_price: float, reason: str = "") -> float:
        if self.position == 0 or self.shares_held == 0:
            return 0.0

        # slippage: sell slightly lower
        fill_price  = current_price * (1 - SLIPPAGE)
        proceeds    = self.shares_held * fill_price
        fee         = proceeds * (TAKER_FEE + STT)   # STT on sell
        net_proceeds= proceeds - fee

        pnl = (fill_price - self.entry_price) / (self.entry_price + 1e-8)

        self.balance += net_proceeds
        self.balance  = max(0.0, self.balance)
        self.n_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        self._place_sandbox_order("SELL", self.shares_held, fill_price)

        fill = Fill(_now(), self.ticker, "SELL", fill_price,
                    self.shares_held, fee, pnl, self.balance)
        self.fills.append(fill)
        self._log_fill(fill)

        pnl_str = f"+{pnl:.4f}" if pnl >= 0 else f"{pnl:.4f}"
        print(f"[broker] SELL  {self.shares_held} shares @ ₹{fill_price:.2f}"
              f"  pnl={pnl_str}  balance=₹{self.balance:.2f}  reason={reason}")

        self.position    = 0.0
        self.entry_price = 0.0
        self.shares_held = 0
        return pnl

    def _place_sandbox_order(self, side: str, quantity: int, price: float):
        """Fire order to Upstox sandbox for logging (optional)."""
        if self._sandbox_api is None:
            return
        try:
            body = upstox_client.PlaceOrderV3Request(
                quantity         = quantity,
                product          = "MIS",           # intraday
                validity         = "DAY",
                price            = round(price, 2),
                tag              = "ppo_agent",
                instrument_token = self.instrument_key,
                order_type       = "LIMIT",
                transaction_type = side,
                disclosed_quantity=0,
                trigger_price    = 0.0,
                is_amo           = False,
                slice            = False,
            )
            self._sandbox_api.place_order(body, algo_name="ppo-trader")
        except ApiException as e:
            # sandbox errors are non-fatal — local simulation continues
            print(f"[broker] sandbox order failed (non-fatal): {e.status}")

    # ── status ────────────────────────────────────────────────────────────────

    def _status(self, current_price: float, trade_occurred=False,
                side=None, pnl=0.0) -> dict:
        nw = self.net_worth(current_price)
        dd = self.drawdown(current_price)
        return {
            "trade_occurred": trade_occurred,
            "side"          : side,
            "pnl"           : pnl,
            "net_worth"     : nw,
            "balance"       : self.balance,
            "position"      : self.position,
            "shares_held"   : self.shares_held,
            "drawdown"      : dd,
            "n_trades"      : self.n_trades,
            "win_rate"      : self.win_rate(),
        }

    # ── CSV logging ───────────────────────────────────────────────────────────

    def _init_log(self):
        os.makedirs(os.path.dirname(self.log_path)
                    if os.path.dirname(self.log_path) else ".", exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=[
                    "timestamp", "symbol", "side", "price",
                    "quantity", "fee", "pnl", "balance_after"
                ]).writeheader()

    def _log_fill(self, fill: Fill):
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "timestamp", "symbol", "side", "price",
                "quantity", "fee", "pnl", "balance_after"
            ]).writerow({
                "timestamp"    : fill.timestamp,
                "symbol"       : fill.symbol,
                "side"         : fill.side,
                "price"        : round(fill.price, 4),
                "quantity"     : fill.quantity,
                "fee"          : round(fill.fee, 4),
                "pnl"          : round(fill.pnl, 6),
                "balance_after": round(fill.balance_after, 2),
            })

    def summary(self, current_price: float) -> str:
        nw  = self.net_worth(current_price)
        ret = (nw / self.initial_balance - 1) * 100
        return (
            f"\n{'─'*55}\n"
            f"  Ticker         : {self.ticker}\n"
            f"  Initial balance: ₹{self.initial_balance:,.2f}\n"
            f"  Net worth      : ₹{nw:,.2f}  ({ret:+.2f}%)\n"
            f"  Balance        : ₹{self.balance:,.2f}\n"
            f"  Shares held    : {self.shares_held}\n"
            f"  Total trades   : {self.n_trades}\n"
            f"  Win rate       : {self.win_rate():.1%}\n"
            f"  Max drawdown   : {self.drawdown(current_price):.1%}\n"
            f"  Trade log      : {self.log_path}\n"
            f"{'─'*55}"
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")