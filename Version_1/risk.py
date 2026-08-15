"""
Risk Management System
- Dynamic Kelly Criterion positioning
- ATR-based stop loss
- Black Swan detection circuit breaker
- All console output routed to stderr (Rich-safe)

Changes (v4):
- on_open_position / on_close_position: stdout → stderr
- check_stop_loss: already stderr, unchanged
- Added vol regime label and Kelly debug line on open
- kelly_fraction_cap default kept at 0.50 (TradingEnvironment passes 0.25 explicitly)
"""

import sys
import numpy as np

RISK_FEATURE_SIZE = 8

def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class RiskManager:
    def __init__(self, initial_balance=10000.0, max_drawdown_limit=0.30,
                 kelly_fraction_cap=0.50, atr_multiplier=2.0,
                 black_swan_threshold=4.0, vol_window=30):
        self.initial_balance      = float(initial_balance)
        self.max_drawdown_limit   = float(max_drawdown_limit)
        self.kelly_fraction_cap   = float(kelly_fraction_cap)
        self.atr_multiplier       = float(atr_multiplier)
        self.black_swan_threshold = float(black_swan_threshold)
        self.vol_window           = int(vol_window)
        self.reset()

    def reset(self):
        self.current_drawdown  = 0.0
        self.peak_balance      = self.initial_balance
        self.current_position  = 0.0
        self.entry_price       = 0.0
        self.direction         = 0.0
        self.stop_loss         = 0.0

        self.current_volatility = 0.0
        self.current_atr        = 0.001
        self.current_kelly      = 0.15
        self.drawdown_velocity  = 0.0
        self.vol_regime         = 0.0
        self.black_swan_z       = 0.0

        self.trade_history  = []
        self.price_history  = []
        self.return_history = []

    # ── market state ──────────────────────────────────────────────────────────

    def update_market(self, current_price, prev_price, net_worth):
        self.price_history.append(current_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)

        log_return = np.log(current_price / (prev_price + 1e-8))
        self.return_history.append(log_return)
        if len(self.return_history) > self.vol_window:
            self.return_history.pop(0)

        if len(self.return_history) >= 5:
            self.current_volatility = float(np.std(self.return_history))
            self.current_atr        = float(np.mean(np.abs(self.return_history[-15:])))
        else:
            self.current_volatility = 0.001
            self.current_atr        = 0.001

        self.current_volatility = max(self.current_volatility, 1e-6)
        self.current_atr        = max(self.current_atr, 1e-6)

        if self.current_volatility < 0.0015:
            self.vol_regime = 0.0
        elif self.current_volatility < 0.0035:
            self.vol_regime = 0.5
        else:
            self.vol_regime = 1.0

        if net_worth > self.peak_balance:
            self.peak_balance = net_worth
        self.current_drawdown = float(
            np.clip((self.peak_balance - net_worth) / (self.peak_balance + 1e-8), 0.0, 1.0)
        )

        if len(self.return_history) >= 15:
            hist_mean          = np.mean(self.return_history[:-1])
            hist_std           = np.std(self.return_history[:-1]) + 1e-8
            self.black_swan_z  = abs(log_return - hist_mean) / hist_std
        else:
            self.black_swan_z = 0.0

    # ── position lifecycle ─────────────────────────────────────────────────────

    def on_open_position(self, entry_price, direction):
        self.entry_price      = float(entry_price)
        self.direction        = float(np.sign(direction))
        self.current_position = 1.0

        stop_loss_noise_floor = 0.015
        atr_buffer            = self.current_atr * self.atr_multiplier
        total_buffer          = max(stop_loss_noise_floor, atr_buffer)

        if self.direction > 0:
            self.stop_loss = self.entry_price * (1.0 - total_buffer)
        else:
            self.stop_loss = self.entry_price * (1.0 + total_buffer)

        vol_label = {0.0: "LOW", 0.5: "MED", 1.0: "HIGH"}.get(self.vol_regime, "?")
        _log(
            f"[RISK]  OPEN {'LONG' if self.direction > 0 else 'SHORT':5s} | "
            f"Entry: ${self.entry_price:.2f} | "
            f"Stop: ${self.stop_loss:.2f} ({total_buffer:.2%}) | "
            f"Vol: {self.current_volatility:.5f} [{vol_label}] | "
            f"ATR: {self.current_atr:.5f} | "
            f"Kelly: {self.current_kelly:.2%} | "
            f"Drawdown: {self.current_drawdown:.2%}"
        )

    def on_close_position(self, reason: str = "agent"):
        if self.entry_price != 0.0:
            side = "LONG" if self.direction > 0 else "SHORT"
            _log(
                f"[RISK]  CLOSE {side:5s} | "
                f"Entry: ${self.entry_price:.2f} | "
                f"Reason: {reason.upper():15s} | "
                f"Drawdown at close: {self.current_drawdown:.2%}"
            )
        self.entry_price      = 0.0
        self.direction        = 0.0
        self.stop_loss        = 0.0
        self.current_position = 0.0

    # ── stop-loss check ───────────────────────────────────────────────────────

    def check_stop_loss(self, current_price) -> bool:
        if self.entry_price == 0.0 or self.stop_loss == 0.0:
            return False

        triggered = False
        if self.direction > 0 and current_price <= self.stop_loss:
            triggered = True
            loss_pct  = (self.entry_price - current_price) / (self.entry_price + 1e-8)
            _log(
                f"[STOP]  LONG stop triggered | "
                f"Entry: ${self.entry_price:.2f} | Stop: ${self.stop_loss:.2f} | "
                f"Price: ${current_price:.2f} | Loss: {loss_pct:.2%}"
            )
        elif self.direction < 0 and current_price >= self.stop_loss:
            triggered = True
            loss_pct  = (current_price - self.entry_price) / (self.entry_price + 1e-8)
            _log(
                f"[STOP]  SHORT stop triggered | "
                f"Entry: ${self.entry_price:.2f} | Stop: ${self.stop_loss:.2f} | "
                f"Price: ${current_price:.2f} | Loss: {loss_pct:.2%}"
            )

        return triggered

    # ── trade recording ───────────────────────────────────────────────────────

    def record_trade(self, pnl: float, reason: str = ""):
        self.trade_history.append(float(pnl))
        if len(self.trade_history) > 50:
            self.trade_history.pop(0)

        if len(self.trade_history) > 1:
            self.drawdown_velocity = pnl - float(np.mean(self.trade_history[-10:]))
        else:
            self.drawdown_velocity = 0.0

        wins   = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t <= 0]
        wr     = len(wins) / len(self.trade_history) if self.trade_history else 0.0
        avg_w  = float(np.mean(wins))   if wins   else 0.0
        avg_l  = float(np.mean(losses)) if losses else 0.0
        _log(
            f"[RISK]  Trade recorded | PnL: {pnl:+.3%} | "
            f"WR(last {len(self.trade_history)}): {wr:.0%} | "
            f"Avg W: {avg_w:+.3%} | Avg L: {avg_l:+.3%} | "
            f"Kelly: {self.current_kelly:.2%}"
        )

    # ── action adjustment ─────────────────────────────────────────────────────

    def adjust_action(self, action, current_price: float):
        """
        Emergency circuit-breaker only — does NOT continuously rescale size so
        PPO's credit assignment is not corrupted.
        """
        direction      = float(action[0])
        size           = float(action[1])
        current_price  = float(current_price)

        # Dynamic Kelly update
        if len(self.trade_history) >= 5:
            wins   = [t for t in self.trade_history if t > 0]
            losses = [t for t in self.trade_history if t <= 0]
            win_rate = len(wins) / len(self.trade_history)

            if wins and losses:
                avg_win       = np.mean(wins)
                avg_loss      = abs(np.mean(losses)) + 1e-8
                win_loss_ratio = avg_win / avg_loss
                kelly         = win_rate - (1.0 - win_rate) / win_loss_ratio
                self.current_kelly = float(np.clip(kelly, 0.02, self.kelly_fraction_cap))
            else:
                self.current_kelly = 0.15
        else:
            self.current_kelly = 0.15

        # Hard cap: Kelly ceiling
        size = min(size, self.current_kelly)
        size = min(size, self.kelly_fraction_cap)

        # Circuit breaker: Black Swan
        if self.black_swan_z > self.black_swan_threshold:
            size = min(size, 0.05)
            _log(
                f"[RISK]  BLACK SWAN active! z={self.black_swan_z:.2f} > {self.black_swan_threshold:.2f} | "
                f"Size capped to 5%"
            )

        return np.array([direction, size])

    # ── feature vector ────────────────────────────────────────────────────────

    def get_risk_features(self):
        if self.entry_price != 0.0 and self.stop_loss != 0.0 and self.price_history:
            current_price = self.price_history[-1]
            stop_distance = abs(current_price - self.stop_loss) / (self.entry_price + 1e-8)
        else:
            stop_distance = 0.0

        return np.array([
            float(np.clip(self.current_volatility * 200.0,  0.0,  1.0)),
            float(np.clip(self.current_atr         * 200.0,  0.0,  1.0)),
            float(np.clip(self.current_kelly,                0.0,  1.0)),
            float(np.clip(self.current_drawdown,             0.0,  1.0)),
            float(np.clip(self.drawdown_velocity   * 200.0, -1.0,  1.0)),
            float(self.vol_regime),
            float(np.clip(stop_distance            * 200.0,  0.0,  1.0)),
            float(np.clip(self.black_swan_z / (self.black_swan_threshold + 1e-8), 0.0, 2.0)),
        ], dtype=np.float64)