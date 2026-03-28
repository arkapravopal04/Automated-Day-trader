"""
PPO agent — actor-critic with clipped surrogate objective.
"""

import numpy as np
from engine import Tensor
from Neural_Nets import LayerNorm, Dropout, Linear, Adam_Optimiser


class PPOAgent:
    def __init__(self, state_size=67, action_size=2):
        self.gamma = 0.99
        self.epsilon = 0.2     
        self.epochs = 5
        self.std = 0.3 
        self.std_min = 0.02
        self.std_decay = 0.995
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.actor_l1 = Linear(state_size, 64)
        self.actor_norm1 = LayerNorm(64)
        self.actor_drop1 = Dropout(0.1)
        self.actor_l2 = Linear(64, 32)
        self.actor_norm2 = LayerNorm(32)
        self.actor_drop2 = Dropout(0.1)
        self.actor_out = Linear(32, action_size)
        self.critic_l1 = Linear(state_size, 64)
        self.critic_norm1 = LayerNorm(64)
        self.critic_drop1 = Dropout(0.1)
        self.critic_l2 = Linear(64, 32)
        self.critic_norm2 = LayerNorm(32)
        self.critic_drop2 = Dropout(0.1)
        self.critic_out = Linear(32, 1)
        all_params = (
            self.actor_l1.parameters() + self.actor_norm1.parameters() +
            self.actor_l2.parameters() + self.actor_norm2.parameters() +
            self.actor_out.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )
        self.optimizer = Adam_Optimiser(all_params, lr=3e-4)

    def _set_train_mode(self, mode: bool):
        for drop in (self.actor_drop1, self.actor_drop2,
                     self.critic_drop1, self.critic_drop2):
            drop.training = mode

    def _actor_forward(self, state):
        x = self.actor_drop1(self.actor_norm1(self.actor_l1(state).relu()))
        x = self.actor_drop2(self.actor_norm2(self.actor_l2(x).relu()))
        return self.actor_out(x)

    def _critic_forward(self, state):
        v = self.critic_drop1(self.critic_norm1(self.critic_l1(state).relu()))
        v = self.critic_drop2(self.critic_norm2(self.critic_l2(v).relu()))
        return self.critic_out(v)

    def _log_prob(self, action_val, mean_val):
        """Gaussian log-probability (scalar, detached from graph)."""
        return -0.5 * ((action_val - mean_val) / (self.std + 1e-8)) ** 2

    def select_action(self, state):
        self._set_train_mode(False)

        out = self._actor_forward(state)
        direction_mean = float(out[0].tanh().data)
        size_mean = float(out[1].sigmoid().data)

        direction = np.clip(direction_mean + np.random.normal(0, self.std), -1, 1)
        size = np.clip(size_mean + np.random.normal(0, self.std), 0, 1)

        log_prob = self._log_prob(direction, direction_mean) + \
                   self._log_prob(size, size_mean)

        value = float(self._critic_forward(state).data.flat[0])

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
        returns = np.array(returns, dtype=np.float64)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self):
        if not self.states:
            return

        states = self.states
        actions = self.actions
        old_log_probs = np.array(self.log_probs, dtype=np.float64)

        returns = self.compute_returns(next_value=0.0)
        advantages = returns - np.array(self.values, dtype=np.float64)

        # Normalise advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self._set_train_mode(True)

        for _ in range(self.epochs):
            # Shuffle mini-batch order each epoch
            indices = np.random.permutation(len(states))
            for i in indices:
                state = states[i]
                action = actions[i]
                old_log_p = float(old_log_probs[i])
                adv = float(advantages[i])
                ret = float(returns[i])

                out = self._actor_forward(state)
                direction_mean = out[0].tanh()
                size_mean = out[1].sigmoid()

                new_log_p = (self._log_prob(action[0], float(direction_mean.data)) +
                             self._log_prob(action[1], float(size_mean.data)))
                ratio = np.exp(float(new_log_p) - old_log_p)
                clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)

                surr = min(ratio * adv, clipped_ratio * adv)
                actor_loss = Tensor(np.array([-surr]))

                new_value = self._critic_forward(state)
                ret_tensor = Tensor(np.array([ret]))
                critic_loss = (ret_tensor - new_value) ** 2

                loss = (actor_loss + Tensor(np.array([0.5])) * critic_loss).sum()

                self.optimizer.zero_grad()
                loss.backward()
                for p in self.optimizer.parameters:
                    np.clip(p.grad, -1.0, 1.0, out=p.grad)
                self.optimizer.step()

        self.std = max(self.std_min, self.std * self.std_decay)

        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []