"""
Trading environment — same structure as before, _get_state now returns
a torch.Tensor so PPOAgent can pass it directly to the network.
"""

import numpy as np
import torch
from Torch_Neural_Nets import DEVICE, to_tensor

TRADE_THRESHOLD = 0.6
NEUTRAL_ZONE = 0.4
R_TRADE_SCALE = 10.0
R_STEP_SCALE = 3.0
R_IDLE_PENALTY= 0.005
R_STRESS_SCALE= 2.5
R_BANKRUPT= 10.0
R_CLIP = 10.0


class TradingEnvironment:
    def __init__(self, X, y, lstm, attention, cnn, flatten, regime, fusion, nlp,
                 prices, initial_balance=10000):
        self.X = X
        self.y = y
        self.lstm= lstm
        self.attention = attention
        self.cnn= cnn
        self.flatten = flatten
        self.regime= regime
        self.fusion= fusion
        self.nlp = nlp
        self.prices= prices.copy().astype(np.float64)
        self.initial_balance = float(initial_balance)
        self.fee= 0.001
        self.total_steps = len(X)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position= 0.0
        self.entry_price = 0.0
        self.cooldown= 0
        self.last_trade_pnl = None

        self.precomputed_nlp = None
        self.current_text= "market news headline"

        self.stress_threshold = 0.9 * self.initial_balance
        self.death_threshold = 0.7 * self.initial_balance


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
            pos_value= abs(self.position) * (1.0 - price_ratio)
        return self.balance + pos_value

    def reset(self):
        self.current_step  = 0
        self.balance = self.initial_balance
        self.position  = 0.0
        self.entry_price   = 0.0
        self.cooldown= 0
        self.last_trade_pnl = None

        first_price = self.prices[0]
        if first_price > 0:
            self.prices = self.prices / first_price * 100.0

        return self._get_state()


    def step(self, action):
        direction = float(action[0])
        size = float(np.clip(action[1], 0.0, 1.0))

        idx  = min(self.current_step, self.total_steps - 1)
        current_price = self.prices[idx]

        reward = 0.0
        trade_occurred = False
        self.last_trade_pnl = None

        should_close = (
            (abs(direction) < NEUTRAL_ZONE) or
            (direction < -TRADE_THRESHOLD and self.position > 0) or
            (direction > TRADE_THRESHOLD  and self.position < 0)
        )

        if should_close and self.position != 0 and self.entry_price != 0:
            if self.position > 0:
                close_value = self.position * (current_price / self.entry_price)
                trade_pnl   = (current_price - self.entry_price) / self.entry_price
            else:
                close_value = abs(self.position) * (
                    1.0 - (current_price - self.entry_price) / self.entry_price)
                close_value = max(0.0, close_value)
                trade_pnl   = (self.entry_price - current_price) / self.entry_price

            self.balance+= close_value * (1.0 - self.fee)
            self.last_trade_pnl = trade_pnl
            reward += trade_pnl * R_TRADE_SCALE
            self.position  = 0.0
            self.entry_price  = 0.0
            self.cooldown= 15
            trade_occurred = True

        if self.cooldown > 0:
            self.cooldown -= 1
        elif self.position == 0 and abs(direction) >= TRADE_THRESHOLD and size > 0.01:
            investment = self.balance * size
            if investment > 10.0:
                effective_investment = investment * (1.0 - self.fee)
                self.entry_price = current_price
                self.position= (effective_investment if direction > 0
                                    else -effective_investment)
                self.balance -= investment
                trade_occurred= True

        if self.position != 0 and self.entry_price != 0:
            prev_price  = self.prices[max(0, self.current_step - 1)]
            step_return = np.log(current_price / (prev_price + 1e-8))
            reward     += step_return * R_STEP_SCALE * np.sign(self.position)
        elif not trade_occurred:
            reward -= R_IDLE_PENALTY

        self.current_step += 1
        current_net_worth = self.net_worth

        if current_net_worth < self.stress_threshold:
            danger_factor = (self.stress_threshold - current_net_worth) / (
                self.stress_threshold - self.death_threshold + 1e-8)
            danger_factor = float(np.clip(danger_factor, 0.0, 1.0))
            reward-= danger_factor * R_STRESS_SCALE

        survival_done = current_net_worth <= self.death_threshold
        if survival_done:
            reward -= R_BANKRUPT

        done = self.current_step >= self.total_steps or survival_done

        info = {'is_bankrupt': survival_done, 'net_worth': current_net_worth}

        if not np.isfinite(reward):
            reward = 0.0
        reward = float(np.clip(reward, -R_CLIP, R_CLIP))

        self.balance = max(0.0, self.balance)
        next_state = self._get_state() if not done else None

        return next_state, reward, done, info


    def compute_features(self, idx: int) -> torch.Tensor:
    
        idx = min(idx, self.total_steps - 1)
        sample_data = self.X[idx].astype(np.float32)
        sample = torch.tensor(sample_data, dtype=torch.float32, device=DEVICE)

        hidden_states, _ = self.lstm(sample)
        lstm_out = self.attention(hidden_states)
        l_f = lstm_out[-1].unsqueeze(0)

        window_size, num_features = sample_data.shape
        cnn_input = sample.reshape(1, window_size, num_features)
        cnn_raw = self.cnn(cnn_input)
        cnn_flat  = self.flatten(cnn_raw)
        c_f = cnn_flat.unsqueeze(0)

        if self.precomputed_nlp is not None:
            nlp_np = self.precomputed_nlp
            if isinstance(nlp_np, torch.Tensor):
                n_f = nlp_np.to(DEVICE).reshape(1, -1).detach()
            else:
                n_f = torch.tensor(nlp_np, dtype=torch.float32, device=DEVICE).reshape(1, -1)
        else:
            with torch.no_grad():
                nlp_out = self.nlp(self.current_text)
            n_f = torch.tensor(nlp_out, dtype=torch.float32, device=DEVICE).reshape(1, -1)

        regime_out = self.regime(sample)
        r_f = regime_out

        fused = self.fusion(l_f, c_f, n_f, r_f)
        return fused.squeeze(0)  

    def _get_state(self):
        idx = min(self.current_step, self.total_steps - 1)
        sample_data = self.X[idx].astype(np.float32)

        sample = torch.tensor(sample_data, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            hidden_states, _ = self.lstm(sample)
            lstm_out = self.attention(hidden_states)
            l_f= lstm_out[-1].unsqueeze(0)

            window_size, num_features = sample_data.shape
            cnn_input = sample.reshape(1, window_size, num_features)
            cnn_raw = self.cnn(cnn_input)
            cnn_flat  = self.flatten(cnn_raw)
            c_f  = cnn_flat.unsqueeze(0)

            if self.precomputed_nlp is not None:
                nlp_np = self.precomputed_nlp
                if isinstance(nlp_np, torch.Tensor):
                    n_f = nlp_np.to(DEVICE).reshape(1, -1)
                else:
                    n_f = torch.tensor(nlp_np, dtype=torch.float32, device=DEVICE).reshape(1, -1)
            else:
                nlp_out = self.nlp(self.current_text)
                n_f = torch.tensor(nlp_out, dtype=torch.float32, device=DEVICE).reshape(1, -1)

            regime_out = self.regime(sample)
            r_f = regime_out

            fused  = self.fusion(l_f, c_f, n_f, r_f)
            f_flat = fused.squeeze(0)

        current_price = self.prices[idx]
        unrealised_pnl = 0.0
        if self.position != 0 and self.entry_price != 0:
            side = np.sign(self.position)
            unrealised_pnl = float(
                side * (current_price - self.entry_price) / (self.entry_price + 1e-8))

        portfolio = torch.tensor([
            self.position / (self.initial_balance + 1e-6),
            self.balance  / (self.initial_balance + 1e-6),
            unrealised_pnl,
        ], dtype=torch.float32, device=DEVICE)

        return torch.cat([f_flat, portfolio], dim=0)