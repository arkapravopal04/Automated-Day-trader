"""
Trading environment — wraps market data, neural feature extractors,
and a simple portfolio simulator into an RL-compatible step/reset API.
"""

import numpy as np
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Linear, Attention, FusionLayers, RegimeDetector
from nlp import NLPEncoder

TRADE_THRESHOLD = 0.505   # direction must exceed this to open / flip
NEUTRAL_ZONE    = 0.10   # direction below this triggers a close
R_TRADE_SCALE   = 8.0   #tuned to be in the same range as typical net worth changes per step, so it can influence the policy without overwhelming the signal from actual profits/losses.  At 10.0 (10× original) the agent learned to make profitable trades but was less consistent and more prone to large drawdowns, likely because the strong reward signal encouraged riskier behavior.  5.0 seems to strike a better balance between rewarding good trades and encouraging more cautious risk management.
R_STEP_SCALE    = 3.0  # encourages the agent to manage open positions effectively, tuned to be in the same range as typical net worth changes per step, so it can influence the policy without overwhelming the signal from actual profits/losses.
R_IDLE_PENALTY  = 0.005 # small penalty for taking no action, encourages the agent to trade and learn from the environment rather than sitting idle.  Tuned to be in the same range as typical net worth changes per step, so it can influence the policy without overwhelming the signal from actual profits/losses.
R_STRESS_SCALE  =2.0   # net worth stress penalty scale — increases as net worth approaches the death threshold, encouraging the agent to take more risk to escape drawdown traps but not so strong that it causes reckless behavior.  At 5.0 (5× original) the agent was more likely to take large risky trades in an attempt to escape drawdowns, which sometimes paid off but often led to faster bankruptcies.  2.0 seems to strike a better balance between encouraging recovery attempts and avoiding reckless gambles.
R_BANKRUPT      = 12.0  # bankruptcy penalty large but not very big,  to allow recovery
R_CLIP          = 10.0  # prevents extreme outliers that could destabilise training


class TradingEnvironment:
    def __init__(self, X, y, lstm, attention, cnn, flatten, regime, fusion, nlp,
                 prices, initial_balance=10000):
        self.X = X
        self.y = y
        self.lstm = lstm
        self.attention = attention
        self.cnn = cnn
        self.flatten = flatten
        self.regime = regime
        self.fusion = fusion
        self.nlp = nlp
        self.prices = prices.copy().astype(np.float64)
        self.initial_balance = float(initial_balance)
        self.fee = 0.001
        self.total_steps = len(X)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.cooldown = 0

        self.last_trade_pnl = None

        self.precomputed_nlp = None
        self.current_text = "market news headline"

        self.stress_threshold = 0.85 * self.initial_balance   
        self.death_threshold  = 0.65 * self.initial_balance   

    @property
    def net_worth(self):
        idx = min(self.current_step, self.total_steps - 1)
        current_price = self.prices[idx]

        if self.position == 0 or self.entry_price == 0:
            return self.balance

        if self.position > 0:
            pos_value = self.position * (current_price / self.entry_price)
        else:
            price_ratio = (current_price - self.entry_price) / self.entry_price
            pos_value = abs(self.position) * (1.0 - price_ratio)

        return self.balance + pos_value

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.cooldown = 0
        self.last_trade_pnl = None

        #Dataleak fix 
        first_price = self.prices[0]
        if first_price > 0:
            self.prices = self.prices / first_price * 100.0

        return self._get_state()
    
    def step(self, action):
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))

        idx = min(self.current_step, self.total_steps - 1)
        current_price = self.prices[idx]

        reward = 0.0
        trade_occurred = False
        self.last_trade_pnl = None

        should_close = (
            (abs(direction) < NEUTRAL_ZONE) or
            (direction < -TRADE_THRESHOLD and self.position > 0) or
            (direction > TRADE_THRESHOLD and self.position < 0)
        )

        if should_close and self.position != 0 and self.entry_price != 0:
            if self.position > 0:
                close_value = self.position * (current_price / self.entry_price)
                trade_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                close_value = abs(self.position) * (
                    1.0 - (current_price - self.entry_price) / self.entry_price
                )
                close_value = max(0.0, close_value)
                trade_pnl = (self.entry_price - current_price) / self.entry_price

            self.balance += close_value * (1.0 - self.fee)
            self.last_trade_pnl = trade_pnl

            reward += trade_pnl * R_TRADE_SCALE

            self.position = 0.0
            self.entry_price = 0.0
            self.cooldown = 8
            trade_occurred = True

        if self.cooldown > 0:
            self.cooldown -= 1
        elif self.position == 0 and abs(direction) >= TRADE_THRESHOLD and size > 0.01:
            investment = self.balance * size
            if investment > 10.0:
                effective_investment = investment * (1.0 - self.fee)
                self.entry_price = current_price
                self.position = (
                    effective_investment if direction > 0 else -effective_investment
                )
                self.balance -= investment
                trade_occurred = True

        if self.position != 0 and self.entry_price != 0:
            prev_price = self.prices[max(0, self.current_step - 1)]
            step_return = np.log(current_price / (prev_price + 1e-8))
            reward += step_return * R_STEP_SCALE * np.sign(self.position)
        elif not trade_occurred:
            reward -= R_IDLE_PENALTY

        # DATA LEAK FIX: compute net worth using current_price BEFORE incrementing step
        if self.position == 0 or self.entry_price == 0:
            current_net_worth = self.balance
        elif self.position > 0:
            current_net_worth = self.balance + self.position * (current_price / self.entry_price)
        else:
            price_ratio = (current_price - self.entry_price) / self.entry_price
            current_net_worth = self.balance + abs(self.position) * (1.0 - price_ratio)

        self.current_step += 1  # increment AFTER net worth is computed

        if current_net_worth < self.stress_threshold:
            danger_factor = (self.stress_threshold - current_net_worth) / (
                self.stress_threshold - self.death_threshold + 1e-8
            )
            danger_factor = float(np.clip(danger_factor, 0.0, 1.0))
            reward -= danger_factor * R_STRESS_SCALE

        survival_done = current_net_worth <= self.death_threshold

        if survival_done:
            reward -= R_BANKRUPT

        done = self.current_step >= self.total_steps or survival_done

        info = {
            'is_bankrupt': survival_done,
            'net_worth':   current_net_worth,
        }

        if not np.isfinite(reward):
            reward = 0.0

        reward = float(np.clip(reward, -R_CLIP, R_CLIP))

        self.balance = max(0.0, self.balance)
        next_state = self._get_state() if not done else None

        return next_state, reward, done, info
    def _get_state(self):
        idx = min(self.current_step, self.total_steps - 1)
        sample_data = self.X[idx].astype(np.float64)

        sample = Tensor(sample_data)

        # LSTM + attention
        hidden_states, _ = self.lstm(sample)
        lstm_out = self.attention(hidden_states)

        # CNN + flatten
        window_size, num_features = sample_data.shape
        cnn_input = Tensor(sample_data.reshape(1, window_size, num_features))
        cnn_raw = self.cnn(cnn_input)
        cnn_out = self.flatten(cnn_raw)

        # NLP
        nlp_out = (
            self.precomputed_nlp if self.precomputed_nlp is not None
            else self.nlp(self.current_text)
        )

        # Regime
        regime_out = self.regime(sample)

        l_f = lstm_out[-1].reshape(1, -1)
        c_f = cnn_out.reshape(1, -1)
        n_f = Tensor(nlp_out.data.reshape(1, -1))
        r_f = regime_out.reshape(1, -1)

        fused = self.fusion(l_f, c_f, n_f, r_f)
        # Stay in-graph: reshape (1,64) → (64,) without wrapping .data
        f_flat = fused.reshape(64)

        # Portfolio features
        current_price = self.prices[idx]
        unrealised_pnl = 0.0
        if self.position != 0 and self.entry_price != 0:
            side = np.sign(self.position)
            unrealised_pnl = float(
                side * (current_price - self.entry_price) / (self.entry_price + 1e-8)
            )

        portfolio = Tensor(np.array([
            self.position / (self.initial_balance + 1e-6),
            self.balance  / (self.initial_balance + 1e-6),
            unrealised_pnl,
        ], dtype=np.float64))

        return f_flat.concat(portfolio)