"""
Trading environment — wraps market data, neural feature extractors,
and a simple portfolio simulator into an RL-compatible step/reset API.
"""

import numpy as np
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Linear, Attention, FusionLayers, RegimeDetector
from nlp import NLPEncoder


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
        self.prices = prices
        self.initial_balance = float(initial_balance)
        self.fee = 0.001
        self.total_steps = len(X)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0     
        self.entry_price = 0.0
        self.cooldown = 0

        self.precomputed_nlp = None
        self.current_text = "market news headline"

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
        return self._get_state()



    def step(self, action):
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))

        idx = min(self.current_step, self.total_steps - 1)
        current_price = self.prices[idx]
        
        reward = 0.0
        trade_occurred = False
        self.last_trade_pnl = None
        
        TRADE_THRESHOLD = 0.7
        NEUTRAL_ZONE = 0.5
        
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
                close_value = abs(self.position) * (1.0 - (current_price - self.entry_price) / self.entry_price)
                close_value = max(0.0, close_value)  
                trade_pnl = (self.entry_price - current_price) / self.entry_price

            self.balance += close_value * (1.0 - self.fee)
            self.last_trade_pnl = trade_pnl 
            reward += trade_pnl * 100.0     
            self.position = 0.0
            self.entry_price = 0.0
            self.cooldown = 20
            trade_occurred = True

        if self.cooldown > 0:
            self.cooldown -= 1
        elif self.position == 0 and abs(direction) >= TRADE_THRESHOLD and size > 0.01:
            investment = self.balance * size
            if investment > 10.0:
                effective_investment = investment * (1.0 - self.fee)
                self.entry_price = current_price
                self.position = effective_investment if direction > 0 else -effective_investment
                self.balance -= investment
                trade_occurred = True

        # Unrealised P&L for open positions
        if self.position != 0 and self.entry_price != 0:
            prev_price = self.prices[max(0, self.current_step - 1)]
            step_return = np.log(current_price / (prev_price + 1e-8))
            reward += step_return * 100.0 * np.sign(self.position)
        elif not trade_occurred:
            reward -= 0.001 

        self.current_step += 1
        
        # Only terminate early on near-total loss
        done = (self.current_step >= self.total_steps or 
                self.net_worth <= self.initial_balance * 0.1)

        if not np.isfinite(reward):
            reward = 0.0
            
        self.balance = max(0.0, self.balance)
        next_state = self._get_state() if not done else None

        return next_state, reward, done


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
        f_flat = Tensor(fused.data.flatten())

        # Portfolio features
        current_price = self.prices[idx]
        unrealised_pnl = 0.0
        if self.position != 0 and self.entry_price != 0:
            side = np.sign(self.position)
            unrealised_pnl = float(side * (current_price - self.entry_price) / (self.entry_price + 1e-8))

        portfolio = Tensor(np.array([
            self.position / (self.initial_balance + 1e-6),
            self.balance / (self.initial_balance + 1e-6),
            unrealised_pnl
        ], dtype=np.float64))

        return f_flat.concat(portfolio)