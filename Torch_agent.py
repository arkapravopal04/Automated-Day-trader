"""
PPO agent — actor-critic with clipped surrogate objective.

Migrated to PyTorch. Architecture, hyperparameters and call-sites
are identical to the custom-engine version.
"""

import numpy as np
import torch
import torch.nn as nn
from Torch_Neural_Nets import Linear, LayerNorm, Dropout, DEVICE


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, min_samples: int = 200):
        self.mean = 0.0
        self.var = 1.0
        self.count= epsilon
        self.min_samples = min_samples

    def update(self, x: np.ndarray) -> None:
        batch_mean  = float(np.mean(x))
        batch_var   = float(np.var(x))
        batch_count = len(x)
        total_count = self.count + batch_count
        delta  = batch_mean - self.mean
        self.mean  += delta * batch_count / total_count
        m_a= self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count  = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < self.min_samples:
            return x
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            Linear(state_size, 64), LayerNorm(64), nn.ReLU(), Dropout(0.1),
            Linear(64, 32),LayerNorm(32), nn.ReLU(), Dropout(0.1),
            Linear(32, action_size),
        )

    def forward(self, x):
        return self.net(x)


class CriticNet(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.net = nn.Sequential(
            Linear(state_size, 64), LayerNorm(64), nn.ReLU(), Dropout(0.1),
            Linear(64, 32), LayerNorm(32), nn.ReLU(), Dropout(0.1),
            Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

class PPOAgent:
    def __init__(self, state_size=67, action_size=2,
                 lstm=None, attention=None, cnn=None, flatten=None,
                 regime=None, fusion=None):

        self.gamma = 0.98
        self.epsilon= 0.18
        self.epochs= 5
        self.std= 0.5
        self.std_min = 0.02
        self.std_decay= 0.997
        self.entropy_coef = 0.01
        self.value_clip= 1.0

        # Rollout buffers
        self.states= []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values= []

        self.return_rms = RunningMeanStd()

        # Feature extractors (trained end-to-end)
        self.lstm = lstm
        self.attention = attention
        self.cnn= cnn
        self.flatten= flatten
        self.regime= regime
        self.fusion= fusion

        # Networks
        self.actor= ActorNet(state_size, action_size).to(DEVICE)
        self.critic = CriticNet(state_size).to(DEVICE)

        # Optimisers
        head_params = (list(self.actor.parameters()) +
                       list(self.critic.parameters()))
        extractor_params = self._extractor_parameters()

        self.optimizer = torch.optim.Adam(head_params, lr=3e-4)
        self.extractor_optimizer = (
            torch.optim.Adam(extractor_params, lr=5e-5)
            if extractor_params else None
        )

    def _extractor_parameters(self):
        params = []
        for module in (self.lstm, self.attention, self.cnn,
                       self.flatten, self.regime, self.fusion):
            if module is not None and hasattr(module, 'parameters'):
                params.extend(list(module.parameters()))
        return params

    def _set_train_mode(self, mode: bool):
        self.actor.train(mode)
        self.critic.train(mode)

    def _log_prob(self, action_val, mean_val):
        return -0.5 * ((action_val - mean_val) / (self.std + 1e-8)) ** 2

    def select_action(self, state):
        """state: torch.Tensor (state_size,)"""
        self._set_train_mode(False)
        with torch.no_grad():
            out  = self.actor(state.unsqueeze(0))   # (1, action_size)
            direction_mean = float(torch.tanh(out[0, 0]).item())
            size_mean = float(torch.sigmoid(out[0, 1]).item())
            value= float(self.critic(state.unsqueeze(0)).item())

        direction = float(np.clip(direction_mean + np.random.normal(0, self.std), -1, 1))
        size= float(np.clip(size_mean      + np.random.normal(0, self.std),  0, 1))
        log_prob  = self._log_prob(direction, direction_mean) + \
                    self._log_prob(size, size_mean)

        self.states.append(state)
        self.actions.append(np.array([direction, size]))
        self.log_probs.append(log_prob)
        self.values.append(value)

        return np.array([direction, size])

    def compute_returns(self, next_value=0.0):
        R = float(next_value)
        returns = []
        for r in reversed(self.rewards):
            R = float(r) + self.gamma * R
            returns.insert(0, R)
        return np.array(returns, dtype=np.float64)

    def update(self):
        if not self.states:
            return

        states        = self.states
        actions       = self.actions
        old_log_probs = np.array(self.log_probs, dtype=np.float64)
        old_values    = np.array(self.values,    dtype=np.float64)

        returns = self.compute_returns(next_value=0.0)
        if len(returns) > 1:
            self.return_rms.update(returns)
            returns = self.return_rms.normalize(returns)

        advantages = returns - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._set_train_mode(True)

        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))
            for i in indices:
                state = states[i].unsqueeze(0)
                action= actions[i]
                old_log_p = float(old_log_probs[i])
                adv = float(advantages[i])
                ret= float(returns[i])
                old_val   = float(old_values[i])

                out= self.actor(state)          
                direction_mean = torch.tanh(out[0, 0])
                size_mean = torch.sigmoid(out[0, 1])

                diff_dir = torch.tensor([action[0]], device=DEVICE) - direction_mean
                diff_size = torch.tensor([action[1]], device=DEVICE) - size_mean
                inv_std2 = 1.0 / (self.std + 1e-8) ** 2

                new_log_p_tensor = (
                    -0.5 * diff_dir  * diff_dir  * inv_std2 +
                    -0.5 * diff_size * diff_size * inv_std2
                )
                new_log_p = float(new_log_p_tensor.item())

                ratio = np.exp(new_log_p - old_log_p)
                clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
                surr  = min(ratio * adv, clipped_ratio * adv)
                actor_loss = (torch.tensor([-surr], device=DEVICE) -
                              self.entropy_coef * new_log_p_tensor)

                new_value   = self.critic(state)            
                new_value_f = float(new_value.item())
                ret_t = torch.tensor([[ret]], dtype=torch.float32, device=DEVICE)

                unclipped_loss = (ret - new_value_f) ** 2
                clipped_val    = np.clip(new_value_f,
                                         old_val - self.value_clip,
                                         old_val + self.value_clip)
                clipped_loss = (ret - clipped_val) ** 2

                if clipped_loss >= unclipped_loss:
                    critic_loss = torch.tensor([clipped_loss], device=DEVICE)
                else:
                    critic_loss = (ret_t - new_value) ** 2

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                if self.extractor_optimizer:
                    self.extractor_optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.optimizer.step()

                if self.extractor_optimizer:
                    for module in (self.lstm, self.attention, self.cnn,
                                   self.flatten, self.regime, self.fusion):
                        if module is not None:
                            torch.nn.utils.clip_grad_norm_(
                                module.parameters(), 0.5)
                    self.extractor_optimizer.step()

        self.std = max(self.std_min, self.std * self.std_decay)
        self.states, self.actions, self.rewards, self.log_probs, self.values = \
            [], [], [], [], []