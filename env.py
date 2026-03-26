'''
this is the palce where we store the actual environment of the project
'''

import numpy as np
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Linear, Attention, FusionLayers, RegimeDetector
from nlp import NLPEncoder, get_sentiment_vector, fetch_news


class TradingEnvironment:
    def __init__(self, X, y, lstm, attention, cnn, flatten, regime, fusion, nlp, prices, initial_balance=10000):
        self.X = X
        self.y = y
        self.lstm = lstm
        self.attention = attention
        self.cnn = cnn
        self.flatten = flatten
        self.regime = regime
        self.fusion = fusion
        self.nlp = nlp
        self.initial_balance = float(initial_balance)
        self.fee = 0.001  
        self.current_step = 0        
        self.balance = self.initial_balance  
        self.position = 0.0       
        self.entry_price = 0.0       
        self.total_steps = len(X)   
        self.current_text = "market news headline"
        self.prices = prices
        self.cooldown = 0

    @property
    def net_worth(self):
        idx = min(self.current_step, self.total_steps - 1)
        current_price = self.prices[idx]
        
        if self.position == 0 or self.entry_price == 0:
            return self.balance
        
        if self.position > 0:
            pos_value = self.position * (current_price / self.entry_price)
        else:
            pos_value = abs(self.position) * (1.0 - (current_price - self.entry_price) / self.entry_price)
            
        return self.balance + pos_value

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.cooldown = 0 
        return self._get_state()

    def step(self, action):
        direction = action[0]
        size = np.clip(action[1], 0.1, 1.0)
        idx = min(self.current_step, self.total_steps - 1)
        current_price = self.prices[idx]
        reward = 0.0
        trade_occurred = False
        
        trade_threshold = 0.5
        neutral_zone = 0.3
        should_close = (abs(direction) < neutral_zone) or \
                       (direction < -trade_threshold and self.position > 0) or \
                       (direction > trade_threshold and self.position < 0)

        if should_close and self.position != 0:
            if self.position > 0:
                trade_reward = (current_price - self.entry_price) / self.entry_price
                final_value = self.position * (current_price / self.entry_price)
            else:
                trade_reward = (self.entry_price - current_price) / self.entry_price
                final_value = abs(self.position) * (1.0 - (current_price - self.entry_price) / self.entry_price)
            
            reward += trade_reward * 100.0 
            self.balance += final_value * (1.0 - self.fee)
            self.position = 0.0
            self.entry_price = 0.0
            
            self.cooldown = 20 
            trade_occurred = True
        if self.cooldown > 0:
            self.cooldown -= 1
        elif self.position == 0 and abs(direction) >= trade_threshold:
            investment = self.balance * size
            if investment > 50.0:
                effective_investment = investment * (1.0 - self.fee)
                self.entry_price = current_price
                if direction > trade_threshold:
                    self.position = effective_investment
                else:
                    self.position = -effective_investment
                self.balance -= investment
                trade_occurred = True

        self.current_step += 1
        done = self.current_step >= self.total_steps
        
        if not done and not trade_occurred and self.position != 0:
            prev_price = self.prices[self.current_step - 1]
            step_return = (self.prices[self.current_step] - prev_price) / prev_price
            
            if self.position > 0:
                reward += step_return * 100.0
            else:
                reward -= step_return * 100.0
        
        if self.position == 0:
            reward -= 0.001

        next_state = self._get_state() if not done else None
        
        self.balance = max(0.0, self.balance)
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
            
        return next_state, reward, done

    def _get_state(self):
        idx = min(self.current_step, self.total_steps - 1)
        sample_data = self.X[idx]
        sample = Tensor(sample_data)
        
        hidden_states, _ = self.lstm(sample)
        lstm_out = self.attention(hidden_states)
        cnn_out = self.flatten(self.cnn(Tensor(sample_data.reshape(1, 10, 5))))
        
        if hasattr(self, 'precomputed_nlp'):
            nlp_out = self.precomputed_nlp
        else:
            nlp_out = self.nlp(self.current_text)
            
        regime_out = self.regime(sample)
        
        l_f = Tensor(lstm_out.data.flatten())
        c_f = Tensor(cnn_out.data.flatten())
        n_f = Tensor(nlp_out.data.flatten())
        r_f = Tensor(regime_out.data.flatten())
        
        fused = self.fusion(l_f, c_f, n_f, r_f)
        f_flat = Tensor(fused.data.flatten())
        
        current_price = self.prices[idx]
        unrealised_pnl = 0.0
        if self.position != 0 and self.entry_price != 0:
            side = 1 if self.position > 0 else -1
            unrealised_pnl = side * (current_price - self.entry_price) / self.entry_price
            
        portfolio = Tensor(np.array([
            self.position / (self.initial_balance + 1e-6), 
            self.balance / (self.initial_balance + 1e-6), 
            unrealised_pnl
        ])) 
        
        return f_flat.concat(portfolio)