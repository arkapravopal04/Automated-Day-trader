"""
PPO agent — actor-critic with clipped surrogate objective.

End-to-end training: during update(), feature extractors (LSTM, CNN, attention,
RegimeDetector, FusionLayers) are re-run WITH gradients so the extractor
optimizer trains them jointly with the actor-critic heads.
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
        batch_var= float(np.var(x))
        batch_count = len(x)
        total_count = self.count + batch_count
        delta= batch_mean - self.mean
        self.mean  += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var   = M2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < self.min_samples:
            return x
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            Linear(state_size, 64), LayerNorm(64), nn.ReLU(), Dropout(0.1),
            Linear(64, 32), LayerNorm(32), nn.ReLU(), Dropout(0.1),
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

        self.gamma  = 0.98
        self.epsilon  = 0.18
        self.epochs = 5
        self.std = 0.5
        self.std_min = 0.02
        self.std_decay= 0.997
        self.entropy_coef = 0.01
        self.value_clip = 1.0

        # Rollout buffers
        self.states= [] 
        self.actions = []
        self.rewards  = []
        self.log_probs  = []
        self.values= []
        self.raw_indices= []
        self.portfolio_states = []   

        self.return_rms = RunningMeanStd()

        # Feature extractors — trained end-to-end via extractor_optimizer
        self.lstm = lstm
        self.attention = attention
        self.cnn  = cnn
        self.flatten = flatten
        self.regime= regime
        self.fusion = fusion

        # Networks
        self.actor= ActorNet(state_size, action_size).to(DEVICE)
        self.critic = CriticNet(state_size).to(DEVICE)
        head_params      = (list(self.actor.parameters()) +
                            list(self.critic.parameters()))
        extractor_params = self._extractor_parameters()

        self.optimizer           = torch.optim.Adam(head_params,       lr=3e-4)
        self.extractor_optimizer = (torch.optim.Adam(extractor_params, lr=1e-4)
                                    if extractor_params else None)


    def _extractor_parameters(self):
        """Return all trainable extractor parameters (Flatten skipped — no params)."""
        params = []
        for module in (self.lstm, self.attention, self.cnn,
                       self.regime, self.fusion):
            if module is not None and hasattr(module, 'parameters'):
                params.extend(list(module.parameters()))
        return params

    def _set_train_mode(self, mode: bool):
        """Put ALL modules (heads + extractors) into train or eval mode."""
        self.actor.train(mode)
        self.critic.train(mode)
        for module in (self.lstm, self.attention, self.cnn,
                       self.flatten, self.regime, self.fusion):
            if module is not None:
                module.train(mode)

    def _log_prob(self, action_val, mean_val):
        return -0.5 * ((action_val - mean_val) / (self.std + 1e-8)) ** 2


    def select_action(self, state, idx: int = None):
        """state: torch.Tensor (state_size,)  |  idx: raw data index for this step.

        idx is stored so update() can recompute extractor features with
        gradients rather than relying on the stored (detached) state tensor.
        """
        self._set_train_mode(False)
        with torch.no_grad():
            out = self.actor(state.unsqueeze(0))
            direction_mean = float(torch.tanh(out[0, 0]).item())
            size_mean = float(torch.sigmoid(out[0, 1]).item())
            value= float(self.critic(state.unsqueeze(0)).item())

        direction = float(np.clip(direction_mean + np.random.normal(0, self.std), -1, 1))
        size= float(np.clip(size_mean      + np.random.normal(0, self.std),  0, 1))
        log_prob= (self._log_prob(direction, direction_mean) +
                     self._log_prob(size, size_mean))

        self.states.append(state)
        self.actions.append(np.array([direction, size]))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.raw_indices.append(idx)
        # Portfolio slice = last 3 dims of state (position, balance, unrealised_pnl)
        self.portfolio_states.append(state[-3:].detach().clone())

        return np.array([direction, size])

    def compute_returns(self, next_value=0.0):
        R = float(next_value)
        returns = []
        for r in reversed(self.rewards):
            R = float(r) + self.gamma * R
            returns.insert(0, R)
        return np.array(returns, dtype=np.float64)

    def update(self, feature_fn=None):
        """Run PPO epochs over the collected rollout.

        feature_fn: callable(idx) -> (64,) tensor WITH gradients.
                    When provided (and all indices are valid), the extractor
                    stack is re-run fresh each sample so LSTM, CNN, attention,
                    RegimeDetector and FusionLayers all receive gradient updates
                    alongside the actor-critic heads.
                    Pass env.compute_features from the training loop.
        """
        if not self.states:
            return

        # End-to-end mode requires a valid feature_fn and a stored data index
        # for every collected step.
        end_to_end = (
            feature_fn is not None and
            len(self.raw_indices) == len(self.states) and
            all(i is not None for i in self.raw_indices)
        )

        actions = self.actions
        portfolio_states = self.portfolio_states
        old_log_probs= np.array(self.log_probs, dtype=np.float64)
        old_values  = np.array(self.values,    dtype=np.float64)

        returns = self.compute_returns(next_value=0.0)
        if len(returns) > 1:
            self.return_rms.update(returns)
            returns = self.return_rms.normalize(returns)

        advantages = returns - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._set_train_mode(True)

        n = len(actions)
        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for i in indices:
                if end_to_end:
                    features  = feature_fn(self.raw_indices[i])         
                    portfolio = portfolio_states[i].to(DEVICE)           
                    state     = torch.cat([features, portfolio], dim=0).unsqueeze(0)  
                else:
                    state = self.states[i].unsqueeze(0)

                action    = actions[i]
                old_log_p = float(old_log_probs[i])
                adv       = float(advantages[i])
                ret       = float(returns[i])
                old_val   = float(old_values[i])

                out            = self.actor(state)
                direction_mean = torch.tanh(out[0, 0])
                size_mean      = torch.sigmoid(out[0, 1])

                diff_dir  = torch.tensor([action[0]], dtype=torch.float32, device=DEVICE) - direction_mean
                diff_size = torch.tensor([action[1]], dtype=torch.float32, device=DEVICE) - size_mean
                inv_std2  = 1.0 / (self.std + 1e-8) ** 2

                new_log_p_tensor = (
                    -0.5 * diff_dir  * diff_dir  * inv_std2 +
                    -0.5 * diff_size * diff_size * inv_std2
                )

                old_log_p_t = torch.tensor([old_log_p], dtype=torch.float32, device=DEVICE)
                adv_t       = torch.tensor([adv],       dtype=torch.float32, device=DEVICE)

                ratio         = torch.exp(new_log_p_tensor - old_log_p_t)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                surr          = torch.min(ratio * adv_t, clipped_ratio * adv_t)
                actor_loss    = -surr - self.entropy_coef * new_log_p_tensor

                new_value = self.critic(state)
                ret_t     = torch.tensor([[ret]],     dtype=torch.float32, device=DEVICE)
                old_val_t = torch.tensor([[old_val]], dtype=torch.float32, device=DEVICE)

                clipped_value  = torch.clamp(new_value,
                                             old_val_t - self.value_clip,
                                             old_val_t + self.value_clip)
                unclipped_loss = (ret_t - new_value) ** 2
                clipped_loss   = (ret_t - clipped_value) ** 2
                critic_loss    = torch.max(unclipped_loss, clipped_loss)

                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                if end_to_end and self.extractor_optimizer:
                    self.extractor_optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.optimizer.step()

                if end_to_end and self.extractor_optimizer:
                    # Conservative clipping for extractors — deeper stack,
                    # indirect gradients, so tighter norm budget.
                    for module in (self.lstm, self.attention, self.cnn,
                                   self.regime, self.fusion):
                        if module is not None:
                            torch.nn.utils.clip_grad_norm_(module.parameters(), 0.5)
                    self.extractor_optimizer.step()

        self.std = max(self.std_min, self.std * self.std_decay)

        # Clear all rollout buffers
        self.states, self.actions, self.rewards, self.log_probs, self.values = \
            [], [], [], [], []
        self.raw_indices, self.portfolio_states = [], []