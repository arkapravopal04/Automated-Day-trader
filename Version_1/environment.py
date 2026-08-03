'''
Trading environment — fast-reacting intra-day trade setups.
Fixes applied:
  - Neutral-zone close now respects MIN_HOLD_STEPS (was bypassing it silently)
  - benchmark B&H uses actual price ratio, not the EMA proxy
  - All risk/trade console prints routed to stderr so Rich telemetry is clean
  - Richer per-step and per-trade console logging added
'''
import sys
import numpy as np
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Linear, Attention, FusionLayers, RegimeDetector
from nlp import NLPEncoder

from risk import RiskManager, RISK_FEATURE_SIZE

from dataclasses import dataclass, field
from typing import Set


@dataclass
class FrictionConfig:
    fee:              float    = 0.0001
    slippage:         float    = 0.0001
    liquid_symbols:   Set[str] = field(default_factory=lambda: {"SPY", "QQQ"})
    liquid_fee:       float    = 0.0001
    liquid_slippage:  float    = 0.0001
    min_trade_pct:    float    = 0.0


def low_friction() -> FrictionConfig:
    return FrictionConfig(
        fee=0.0001, slippage=0.0001,
        liquid_fee=0.0001, liquid_slippage=0.0001,
        min_trade_pct=0.0,
    )

def realistic_friction() -> FrictionConfig:
    return FrictionConfig(
        fee=0.00025, slippage=0.00025,
        liquid_fee=0.0001, liquid_slippage=0.0001,
        min_trade_pct=0.02,
    )

def high_friction() -> FrictionConfig:
    return FrictionConfig(
        fee=0.001, slippage=0.001,
        liquid_fee=0.0005, liquid_slippage=0.0005,
        min_trade_pct=0.02,
    )

RED = "\033[91m"
GREEN  = "\033[92m"
ORANGE = "\033[33m"
BLUE = "\033[94m"
RESET  = "\033[0m"

TRADE_THRESHOLD = 0.25
NEUTRAL_ZONE    = 0.10
MIN_HOLD_STEPS  = 1
MAX_HOLD_STEPS  = 60

R_STRESS_SCALE          = 0.05
R_BANKRUPT              = 100.0
R_CLIP                  = 15.0
R_GROWTH_SCALE          = 20.0
R_PREMATURE_CLOSE_PENALTY = 0.01

R_SHARPE_SCALE          = 0.5

DRIFT_EMA_ALPHA         = 0.10
BENCHMARK_EMA_ALPHA     = 0.10

R_HOLD_LOSER_SCALE      = 0.05
R_HOLD_LOSER_CAP        = 0.10
LOSER_THRESHOLD         = 0.005

MILESTONES        = [12500, 15000, 20000, 30000, 50000]
MILESTONE_REWARDS = [5.0,   10.0,  30.0,  60.0,  80.0]

# ── console helpers (all to stderr so Rich is undisturbed) ─────────────────────
def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class TradingEnvironment:
    def __init__(self, X, y, lstm, attention, cnn, flatten, regime, fusion, nlp,
                 prices, initial_balance=10000,
                 friction: FrictionConfig = None,
                 symbol: str = "",
                 mirror_data: bool = False):
        self._lstm_c = None
        self._lstm_h = None
        self.lstm      = lstm
        self.attention = attention
        self.cnn       = cnn
        self.flatten   = flatten
        self.regime    = regime
        self.fusion    = fusion
        self.nlp       = nlp
        self.initial_balance = float(initial_balance)
        self.symbol    = symbol
        self.mirror_data = mirror_data

        prices_copied = prices.copy().astype(np.float64)
        X_copied      = X.copy()
        y_copied      = y.copy()

        if self.mirror_data:
            p0 = prices_copied[0] + 1e-8
            self.prices = (p0 ** 2) / (prices_copied + 1e-8)
            if X_copied.ndim == 3:
                X_copied[:, :, 0:4] = (p0 ** 2) / (np.abs(X_copied[:, :, 0:4]) + 1e-8)
            elif X_copied.ndim == 2:
                X_copied = -X_copied
        else:
            self.prices = prices_copied

        self.X = X_copied
        self.y = y_copied

        _friction  = friction if friction is not None else low_friction()
        _is_liquid = symbol.upper() in _friction.liquid_symbols
        self.fee           = _friction.liquid_fee      if _is_liquid else _friction.fee
        self.slippage      = _friction.liquid_slippage if _is_liquid else _friction.slippage
        self.min_trade_pct = _friction.min_trade_pct
        self.friction      = _friction

        self.total_steps  = len(X_copied)
        self.current_step = 0
        self.balance      = self.initial_balance
        self.position     = 0.0
        self.entry_price  = 0.0
        self.cooldown     = 0

        self.last_trade_pnl        = None
        self.n_trades_this_episode = 0
        self.milestones_crossed    = set()

        self.precomputed_nlp = None
        self.current_text    = "market news headline"

        self.stress_threshold     = 0.93 * self.initial_balance
        self.death_threshold      = 0.80 * self.initial_balance
        self.stress_penalty_accum = 0.0
        self.stress_penalty_cap   = 5.0
        self.hold_loser_steps     = 0
        self.hold_steps           = 0
        self.ema_log_return       = 0.0
        self.benchmark_ema        = 0.0

        self.short_only_mode = False
        self.long_only_mode  = False

        self.last_reward_breakdown = {
            'trade': 0.0, 'step': 0.0, 'hold_loser': 0.0, 'stress': 0.0,
            'premature_close': 0.0, 'milestone': 0.0, 'terminal': 0.0, 'total': 0.0
        }

        self.risk_manager = RiskManager(
            initial_balance=self.initial_balance,
            max_drawdown_limit=0.30,
            kelly_fraction_cap=0.25,
            atr_multiplier=2.5,
            black_swan_threshold=5.0,
            vol_window=30,
        )

    # ── net-worth helpers ──────────────────────────────────────────────────────

    def get_net_worth_at_price(self, price):
        if self.position == 0 or self.entry_price == 0:
            return self.balance

        if self.position > 0:
            exec_price  = price * (1.0 - self.slippage)
            close_value = self.position * (exec_price / self.entry_price)
            net_return  = close_value * (1.0 - self.fee)
            return self.balance + net_return
        else:
            exec_price  = price * (1.0 + self.slippage)
            close_value = abs(self.position) * (
                1.0 - (exec_price - self.entry_price) / self.entry_price
            )
            close_value = max(0.0, close_value)
            net_return  = close_value * (1.0 - self.fee)
            return self.balance + net_return

    @property
    def net_worth(self):
        idx = min(self.current_step, self.total_steps - 1)
        return self.get_net_worth_at_price(self.prices[idx])

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self):
        self.current_step          = 0
        self.balance               = self.initial_balance
        self.position              = 0.0
        self.entry_price           = 0.0
        self.cooldown              = 0
        self.last_trade_pnl        = None
        self.n_trades_this_episode = 0
        self.milestones_crossed    = set()
        self.stress_penalty_accum  = 0.0
        self.hold_loser_steps      = 0
        self.hold_steps            = 0
        self.ema_log_return        = 0.0
        self.benchmark_ema         = 0.0
        self.risk_manager.reset()
        return self._get_state()

    # ── step ───────────────────────────────────────────────────────────────────

    def step(self, action):
        direction_in = float(action[0])
        size_in      = float(np.clip(action[1], 0.0, 1.0))

        idx      = min(self.current_step, self.total_steps - 1)
        next_idx = min(self.current_step + 1, self.total_steps - 1)

        prev_price    = self.prices[idx]
        current_price = self.prices[next_idx]

        _raw_bar_return = np.log(current_price / (prev_price + 1e-8))
        self.ema_log_return = (DRIFT_EMA_ALPHA * _raw_bar_return
                               + (1.0 - DRIFT_EMA_ALPHA) * self.ema_log_return)
        self.benchmark_ema  = (BENCHMARK_EMA_ALPHA * _raw_bar_return
                               + (1.0 - BENCHMARK_EMA_ALPHA) * self.benchmark_ema)

        reward         = 0.0
        prev_net_worth = self.get_net_worth_at_price(prev_price)
        self.risk_manager.update_market(current_price, prev_price, prev_net_worth)

        adjusted = self.risk_manager.adjust_action(
            np.array([direction_in, size_in]), current_price
        )
        direction = float(adjusted[0])
        size      = float(adjusted[1])

        danger_factor     = 0.0
        trade_occurred    = False
        premature_penalty = 0.0
        self.last_trade_pnl = None

        stop_loss_hit = self.risk_manager.check_stop_loss(current_price)
        max_hold_hit  = (self.position != 0 and self.hold_steps >= MAX_HOLD_STEPS)
        risk_override = stop_loss_hit or max_hold_hit

        agent_wants_close = (
            (self.position > 0 and direction < -NEUTRAL_ZONE) or
            (self.position < 0 and direction >  NEUTRAL_ZONE)
        )
        # Neutral drift: direction is ambiguous, not an explicit close signal
        neutral_drift = (self.position != 0 and abs(direction) < NEUTRAL_ZONE)

        min_hold_satisfied = self.hold_steps >= MIN_HOLD_STEPS

        # Premature-close penalty only for explicit reversal signals before min hold
        if agent_wants_close and self.position != 0 and not min_hold_satisfied:
            premature_penalty = R_PREMATURE_CLOSE_PENALTY
            reward -= premature_penalty
            print(f"\033[33mPREMATURE CLOSE ATTEMPT DETECTED...REDIRECTING...\033[0m")

        # FIX (Bug 5): neutral-drift close now also respects MIN_HOLD_STEPS.
        should_close = (
            risk_override
            or (agent_wants_close and min_hold_satisfied)
            or (neutral_drift and min_hold_satisfied)
        )

        close_pnl_pct = 0.0
        if should_close and self.position != 0 and self.entry_price != 0:
            if stop_loss_hit:
                close_reason = "stop_loss"
            elif max_hold_hit:
                close_reason = "max_hold"
            elif agent_wants_close:
                close_reason = "agent_reversal"
            else:
                close_reason = "neutral_drift"

            if self.position > 0:
                exec_price  = current_price * (1.0 - self.slippage)
                close_value = self.position * (exec_price / self.entry_price)
            else:
                exec_price  = current_price * (1.0 + self.slippage)
                close_value = abs(self.position) * (
                    1.0 - (exec_price - self.entry_price) / self.entry_price
                )
                close_value = max(0.0, close_value)

            investment  = abs(self.position) / (1.0 - self.fee)
            net_return  = close_value * (1.0 - self.fee)
            close_pnl_pct = (net_return - investment) / (investment + 1e-8)

            self.balance        += net_return
            self.last_trade_pnl  = close_pnl_pct
            self.risk_manager.record_trade(close_pnl_pct, reason=close_reason)
            self.risk_manager.on_close_position(reason=close_reason)

            _log(
                f"[{self.symbol}] CLOSE {('LONG' if self.position > 0 else 'SHORT')} | "
                f"{BLUE}Reason: {close_reason.upper():15s}{RESET} | "
                f"{GREEN if close_pnl_pct > 0 else RED}PnL: {close_pnl_pct:+.3%}{RESET} | "
                f"Hold: {self.hold_steps} steps | "
                f"NW: ${self.balance:,.0f}"
            )

            self.position         = 0.0
            self.entry_price      = 0.0
            self.cooldown         = 8
            self.hold_steps       = 0
            self.hold_loser_steps = 0
            trade_occurred        = True
            self.n_trades_this_episode += 1

        # ── open new position ──────────────────────────────────────────────────
        if self.cooldown > 0:
            self.cooldown -= 1
        elif (self.position == 0 and not trade_occurred
              and abs(direction) >= TRADE_THRESHOLD and size > 0.01):
            if not ((self.short_only_mode and direction > 0)
                    or (self.long_only_mode and direction < 0)):
                investment     = self.balance * size
                min_investment = max(self.balance * self.min_trade_pct, 2.0)

                if investment >= min_investment:
                    effective_investment = investment * (1.0 - self.fee)

                    if direction > 0:
                        self.entry_price = current_price * (1.0 + self.slippage)
                        self.position    = effective_investment
                    else:
                        self.entry_price = current_price * (1.0 - self.slippage)
                        self.position    = -effective_investment

                    self.balance         -= investment
                    self.hold_steps       = 0
                    self.hold_loser_steps = 0
                    self.risk_manager.on_open_position(self.entry_price, direction)
                    trade_occurred = True

                    _log(
                        f"[{self.symbol}] OPEN  {'LONG' if direction > 0 else 'SHORT':5s} | "
                        f"Entry: ${self.entry_price:.2f} | "
                        f"Size: {size:.2%} of balance | "
                        f"Kelly: {self.risk_manager.current_kelly:.2%} | "
                        f"Vol: {self.risk_manager.current_volatility:.4f} | "
                        f"NW: ${self.balance + abs(self.position):,.0f}"
                    )

        # ── hold-loser penalty ─────────────────────────────────────────────────
        hold_loser_penalty = 0.0
        if self.position != 0 and self.entry_price != 0:
            self.hold_steps += 1
            side       = np.sign(self.position)
            unrealised = side * (current_price - self.entry_price) / (self.entry_price + 1e-8)

            if unrealised < -LOSER_THRESHOLD:
                loss_depth  = abs(unrealised) - LOSER_THRESHOLD
                self.hold_loser_steps += 1
                time_factor = 1.0 + 0.05 * min(self.hold_loser_steps, 20)
                hold_loser_penalty = min(
                    loss_depth * R_HOLD_LOSER_SCALE * time_factor,
                    R_HOLD_LOSER_CAP
                )
                reward -= hold_loser_penalty
                if self.hold_loser_steps % 10 == 1:   # log every 10 loser steps
                    _log(
                        f"[{self.symbol}] HOLD_LOSER | Unrealised: {unrealised:+.3%} | "
                        f"Penalty: {hold_loser_penalty:.4f} | Steps losing: {self.hold_loser_steps}"
                    )
            else:
                self.hold_loser_steps = 0

        self.current_step += 1
        current_net_worth  = self.net_worth

        # ── base reward: vol-normalised portfolio return ────────────────────────
        portfolio_return = (current_net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        market_vol       = max(self.risk_manager.current_volatility, 1e-5)
        base_reward      = (portfolio_return / market_vol) * R_SHARPE_SCALE
        reward          += base_reward

        # ── stress penalty ─────────────────────────────────────────────────────
        if current_net_worth < self.stress_threshold:
            danger_factor  = (self.stress_threshold - current_net_worth) / (
                self.stress_threshold - self.death_threshold + 1e-8
            )
            danger_factor  = float(np.clip(danger_factor, 0.0, 1.0))
            stress_penalty = danger_factor * R_STRESS_SCALE
            remaining_cap  = max(0.0, self.stress_penalty_cap - self.stress_penalty_accum)
            stress_penalty = min(stress_penalty, remaining_cap)
            reward        -= stress_penalty
            self.stress_penalty_accum += stress_penalty
            danger_factor  = stress_penalty / (R_STRESS_SCALE + 1e-8)

        # ── survival / done ────────────────────────────────────────────────────
        survival_done = (
            (current_net_worth <= self.death_threshold) or
            (self.risk_manager.current_drawdown >= self.risk_manager.max_drawdown_limit)
        )
        if survival_done:
            reward -= R_BANKRUPT
            _log(
                f"[{self.symbol}] *** BANKRUPT / MAX DRAWDOWN *** | "
                f"NW: ${current_net_worth:,.0f} | "
                f"Drawdown: {self.risk_manager.current_drawdown:.2%}"
            )

        episode_end = self.current_step >= self.total_steps
        done        = episode_end or survival_done

        if not np.isfinite(reward):
            reward = 0.0
        reward = float(np.clip(reward, -R_CLIP, R_CLIP))

        # ── milestones ─────────────────────────────────────────────────────────
        milestone_reward = 0.0
        for milestone, bonus in zip(MILESTONES, MILESTONE_REWARDS):
            if milestone not in self.milestones_crossed and current_net_worth >= milestone:
                milestone_reward += bonus
                self.milestones_crossed.add(milestone)
                _log(f"[{self.symbol}] ★ MILESTONE ${milestone:,} reached! Bonus: +{bonus:.1f}")

        # ── terminal reward: actual alpha vs true B&H ──────────────────────────
        terminal_reward = 0.0
        if episode_end and not survival_done:
            growth = (current_net_worth / self.initial_balance) - 1.0

            p_start    = float(self.prices[0])
            p_end      = float(self.prices[min(self.current_step, self.total_steps - 1)])
            bh_growth  = (p_end - p_start) / (p_start + 1e-8)
            bh_growth  = float(np.clip(bh_growth, -0.5, 2.0))
            alpha      = growth - bh_growth
            terminal_reward = R_GROWTH_SCALE * alpha

            _log(
                f"[{self.symbol}] EPISODE END | "
                f"NW: ${current_net_worth:,.0f} | "
                f"Growth: {growth:+.2%} | "
                f"B&H: {bh_growth:+.2%} | "
                f"Alpha: {alpha:+.2%} | "
                f"Terminal bonus: {terminal_reward:+.2f} | "
                f"Trades: {self.n_trades_this_episode}"
            )

        reward += milestone_reward + terminal_reward

        if not np.isfinite(reward):
            reward = 0.0
        reward = float(np.clip(reward, -100.0, 100.0))

        info = {
            'is_bankrupt':     survival_done,
            'net_worth':       current_net_worth,
            'balance':         self.balance,
            'position':        self.position,
            'price':           current_price,
            'adjusted_action': np.array([direction, size], dtype=np.float64),
        }

        self.balance = max(0.0, self.balance)
        next_state   = self._get_state() if not done else None

        self.last_reward_breakdown = {
            'trade':           0.0,
            'step':            round(float(base_reward),        4),
            'hold_loser':      round(-hold_loser_penalty,       4),
            'stress':          round(-danger_factor * R_STRESS_SCALE, 4) if current_net_worth < self.stress_threshold else 0.0,
            'premature_close': round(-premature_penalty,        4),
            'milestone':       round(milestone_reward,          4),
            'terminal':        round(terminal_reward,           4),
            'total':           round(reward,                    4),
        }
        return next_state, reward, done, info

    # ── state construction ─────────────────────────────────────────────────────

    def _get_state(self):
        idx         = min(self.current_step, self.total_steps - 1)
        sample_data = self.X[idx].astype(np.float64)

        if not np.isfinite(sample_data).all():
            sample_data = np.nan_to_num(sample_data, nan=0.0, posinf=0.0, neginf=0.0)

        sample = Tensor(sample_data)

        hidden_states, (h_out, c_out) = self.lstm(
            sample,
            h_prev=self._lstm_h,
            c_prev=self._lstm_c
        )
        self._lstm_h = [Tensor(h.data.copy()) for h in h_out]
        self._lstm_c = [Tensor(c.data.copy()) for c in c_out]

        lstm_out         = self.attention(hidden_states)

        window_size, num_features = sample_data.shape
        cnn_input   = Tensor(sample_data.reshape(1, window_size, num_features))
        cnn_out     = self.flatten(self.cnn(cnn_input).tanh())

        nlp_out = (
            self.precomputed_nlp if self.precomputed_nlp is not None
            else self.nlp(self.current_text)
        )

        regime_out = self.regime(sample)

        if hasattr(self, 'telemetry') and self.telemetry is not None:
            self.telemetry.log_regime(regime_out.data.flatten())

        risk_features = self.risk_manager.get_risk_features()
        risk_tensor   = Tensor(risk_features.reshape(1, -1))

        l_f = lstm_out[-1].reshape(1, -1)
        c_f = cnn_out.reshape(1, -1)
        n_f = Tensor(nlp_out.data.reshape(1, -1))
        r_f = regime_out.reshape(1, -1)

        fused  = self.fusion(l_f, c_f, n_f, r_f, risk_tensor)
        f_flat = fused.reshape(64)

        current_price  = self.prices[idx]
        unrealised_pnl = 0.0
        if self.position != 0 and self.entry_price != 0:
            side           = np.sign(self.position)
            unrealised_pnl = float(
                side * (current_price - self.entry_price) / (self.entry_price + 1e-8)
            )

        current_nw = self.net_worth + 1e-6
        portfolio  = Tensor(np.array([
            self.position / current_nw,
            self.balance  / current_nw,
            unrealised_pnl,
        ], dtype=np.float64))

        # Returns a dynamic state containing the fully connected computational graph 
        # back to all feature extractors (LSTM, CNN, Attention, Regime, Fusion)
        return f_flat.concat(portfolio).concat(Tensor(risk_features))

    def get_raw_state(self):
        """
        Returns the raw input sample and detached LSTM hidden state at the
        current step. Used by the training loop to build the TBPTT buffer.
        Only the data is returned — no graph connections.
        """
        idx         = min(self.current_step, self.total_steps - 1)
        raw_sample  = Tensor(self.X[idx].astype(np.float64))   # [window, features]

        h_snap = (
            [Tensor(h.data.copy()) for h in self._lstm_h]
            if self._lstm_h is not None
            else None
        )
        c_snap = (
            [Tensor(c.data.copy()) for c in self._lstm_c]
            if self._lstm_c is not None
            else None
        )
        return raw_sample, h_snap, c_snap