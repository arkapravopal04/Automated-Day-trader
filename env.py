'''
this is the palce where we store the actual environment of the project
'''

import numpy as np
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Linear, Attention, FusionLayers, RegimeDetector
from nlp import NLPEncoder, get_sentiment_vector, fetch_news


class TradingEnvironment:
    def __init__(self, X, y, lstm , attention, cnn , flatten , regime, fusion, nlp,prices, initial_balance = 10000):
        self.X = X
        self.y = y
        self.lstm = lstm
        self.attention = attention
        self.cnn = cnn
        self.flatten = flatten
        self.regime = regime
        self.fusion = fusion
        self.nlp = nlp
        self.initial_balance = initial_balance
        self.current_step = 0        
        self.balance = initial_balance  
        self.position = 0.0       
        self.entry_price = 0       
        self.total_steps = len(X)   
        self.current_text = "market news headline"
        self.prices = prices

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._get_state()
    

# weighted average entry price missing
    def step(self, action):
        
        direction, size = action
        current_price = self.prices[self.current_step] 

        if self.current_step % 500 == 0:
            print(f"  Step {self.current_step}: balance={self.balance:.2f} position={self.position:.2f} entry={self.entry_price:.4f} price={current_price:.4f}")
        threshold = 0.3
        reward = 0
        if direction > threshold:
            if self.position < 0 and self.entry_price != 0:
                pnl = abs(self.position) * (self.entry_price - current_price) / self.entry_price
                self.balance += abs(self.position) + pnl
                self.position = 0
                reward = pnl / self.initial_balance
                
            if self.position == 0:
                amount = self.balance * min(size, 0.5)
                self.position = amount
                self.balance -= amount
                self.entry_price = current_price

        elif direction < -threshold:
            if self.position > 0 and self.entry_price != 0:
                pnl = self.position * (current_price - self.entry_price) / self.entry_price
                self.balance += self.position + pnl
                self.position = 0
                reward = pnl / self.initial_balance
                
            if self.position == 0:
                amount = self.balance * min(size, 0.5)
                self.position = -amount 
                self.balance -= amount
                self.entry_price = current_price

        else:
            if self.position != 0 and self.entry_price != 0:
                unrealised_pnl = (current_price - self.entry_price) / self.entry_price * self.position
                reward = unrealised_pnl * 0.1
            else:
                reward = 0

        # can have time penalty 
        self.current_step += 1
        done = self.current_step >= self.total_steps
        next_state = self._get_state() if not done else None

        self.balance = max(0.0, self.balance)
        if np.isnan(reward) or np.isinf(reward):
          print(f"BAD REWARD at step {self.current_step}: reward={reward}, position={self.position}, entry={self.entry_price}, price={current_price}")
          reward = -0.01   
        
        return next_state, reward, done

    def _get_state(self):
        sample = Tensor(self.X[self.current_step])
        hidden_states, (h, c) = self.lstm(sample)
        lstm_out = self.attention(hidden_states)
        cnn_out = self.flatten(self.cnn(Tensor(self.X[self.current_step].reshape(1, 10, 5))))
        if hasattr(self, 'precomputed_nlp'):
            nlp_out = self.precomputed_nlp
        else:
            nlp_out = self.nlp(self.current_text)
        regime_out = self.regime(sample)
        fused = self.fusion(lstm_out, cnn_out, nlp_out, regime_out)
        current_price = self.prices[self.current_step]
        if self.position > 0 and self.entry_price != 0:
            unrealised_pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position < 0 and self.entry_price != 0:
            unrealised_pnl = (self.entry_price - current_price) / self.entry_price
        else:
            unrealised_pnl = 0.0
        portfolio = Tensor(np.array([self.position / self.initial_balance, self.balance / self.initial_balance, unrealised_pnl]))       
        state = fused.concat(portfolio)
        return state