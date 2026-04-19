"""
PPO agent — actor-critic with clipped surrogate objective.

The feature extractors (LSTM, Attention, CNN, RegimeDetector, FusionLayers)
are registered here and included in the optimiser so their weights are trained
end-to-end alongside the PPO head. Without this, the state representation
is a frozen random projection and the agent cannot learn meaningful patterns.
"""

import numpy as np
from engine import Tensor
from Neural_Nets import LayerNorm, Dropout, Linear, Adam_Optimiser

#normalizes reward
class RunningMeanStd:
    """Welford online algorithm for running mean and variance.

    FIX (issue 4): a `min_samples` threshold prevents the normalizer from
    being applied during the first few episodes when the variance estimate is
    essentially noise.  Before that threshold is reached, `normalize` returns
    the input unchanged so early gradients are not scaled by an unreliable
    estimate.
    """

    def __init__(self, epsilon: float = 1e-4, min_samples: int = 200):
        self.mean        = 0.0
        self.var         = 1.0
        self.count       = epsilon# small seed so variance is never exactly 0
        self.min_samples = min_samples

    def update(self, x: np.ndarray) -> None:
        batch_mean  = float(np.mean(x))
        batch_var   = float(np.var(x))
        batch_count = len(x)

        total_count = self.count + batch_count
        delta       = batch_mean - self.mean

        self.mean  += delta * batch_count / total_count
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        M2          = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var    = M2 / total_count
        self.count  = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        #we skip normalization until we have enough samples for a stable estimate
        if self.count < self.min_samples:
            return x
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class PPOAgent:
    def __init__(self, state_size=67, action_size=2,
                 lstm=None, attention=None, cnn=None, flatten=None,
                 regime=None, fusion=None):

        self.gamma      = 0.995
        self.epsilon    = 0.2    # clip ratio — increases stability, can slow learning
        self.epochs     = 5      # update epochs per rollout
        self.std        = 0.30  # initial exploration noise
        self.std_min    = 0.283
        self.std_decay  = 0.0

        # rewards for exploration and risk management  tuned to be in the same range as typical net worth changes per step, so they can influence the policy without overwhelming the signal from actual profits/losses.
        self.entropy_coef = 0.01

        self.value_clip = 0.2 # prevents excessive value function updates that can destabilize training, tuned to be in the same range as typical net worth changes per step, so it can influence the policy without overwhelming the signal from actual profits/losses.

        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []

        self.return_rms = RunningMeanStd()

        self.lstm      = lstm
        self.attention = attention
        self.cnn       = cnn
        self.flatten   = flatten
        self.regime    = regime
        self.fusion    = fusion

        self.actor_l1    = Linear(state_size, 64)
        self.actor_norm1 = LayerNorm(64)
        self.actor_drop1 = Dropout(0.1)
        self.actor_l2    = Linear(64, 32)
        self.actor_norm2 = LayerNorm(32)
        self.actor_drop2 = Dropout(0.1)
        self.actor_out   = Linear(32, action_size)

        self.critic_l1    = Linear(state_size, 64)
        self.critic_norm1 = LayerNorm(64)
        self.critic_drop1 = Dropout(0.1)
        self.critic_l2    = Linear(64, 32)
        self.critic_norm2 = LayerNorm(32)
        self.critic_drop2 = Dropout(0.1)
        self.critic_out   = Linear(32, 1)

        head_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_out.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )
        extractor_params = self._extractor_parameters()

        self.optimizer = Adam_Optimiser(head_params, lr=3e-4)
        self.extractor_optimizer = (
            Adam_Optimiser(extractor_params, lr=1e-4) if extractor_params else None
        )


    def _set_train_mode(self, mode: bool):
        for drop in (self.actor_drop1, self.actor_drop2,
                     self.critic_drop1, self.critic_drop2):
            drop.training = mode

    def _extractor_parameters(self):
        params = []
        for module in (self.lstm, self.attention, self.cnn,
                       self.flatten, self.regime, self.fusion):
            if module is not None and hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params

    def _actor_forward(self, state):
        x = self.actor_drop1(self.actor_norm1(self.actor_l1(state).relu()))
        x = self.actor_drop2(self.actor_norm2(self.actor_l2(x).relu()))
        return self.actor_out(x)

    def _critic_forward(self, state):
        v = self.critic_drop1(self.critic_norm1(self.critic_l1(state).relu()))
        v = self.critic_drop2(self.critic_norm2(self.critic_l2(v).relu()))
        return self.critic_out(v)

    def _log_prob(self, action_val, mean_val):
        """Gaussian log-probability (constant terms omitted — they cancel in ratio)."""
        return -0.5 * ((action_val - mean_val) / (self.std + 1e-8)) ** 2


    def select_action(self, state):
        self._set_train_mode(False)

        out = self._actor_forward(state)
        direction_mean = float(out[0].tanh().data)
        size_mean      = float(out[1].sigmoid().data)

        direction = np.clip(direction_mean + np.random.normal(0, self.std), -1, 1)
        size      = np.clip(size_mean      + np.random.normal(0, self.std),  0, 1)

        log_prob = (self._log_prob(direction, direction_mean) +
                    self._log_prob(size,      size_mean))

        value = float(self._critic_forward(state).data.flat[0])

        self.states.append(state)
        self.actions.append(np.array([direction, size]))
        self.log_probs.append(log_prob)
        self.values.append(value)

        return np.array([direction, size])


    def compute_returns(self, next_value=0.0):
        """Discount rewards into returns.  No normalization here — that is
        done externally in update() AFTER discounting so the temporal
        structure is preserved (fix for issue 3)."""
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
                state     = states[i]
                action    = actions[i]
                old_log_p = float(old_log_probs[i])
                adv       = float(advantages[i])
                ret       = float(returns[i])

                out            = self._actor_forward(state)
                direction_mean = out[0].tanh()
                size_mean      = out[1].sigmoid()

                diff_dir  = Tensor(np.array([action[0]])) - direction_mean
                diff_size = Tensor(np.array([action[1]])) - size_mean
                inv_std2  = Tensor(np.array([1.0 / (self.std + 1e-8) ** 2]))
                new_log_p_tensor = (
                    Tensor(np.array([-0.5])) * diff_dir  * diff_dir  * inv_std2 +
                    Tensor(np.array([-0.5])) * diff_size * diff_size * inv_std2
                )
                new_log_p = float(new_log_p_tensor.data.flat[0])

                ratio         = np.exp(new_log_p - old_log_p)
                clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
                surr          = min(ratio * adv, clipped_ratio * adv)

                actor_loss = (
                    Tensor(np.array([-surr])) -
                    Tensor(np.array([self.entropy_coef])) * new_log_p_tensor
                )

                new_value   = self._critic_forward(state)
                new_value_f = float(new_value.data.flat[0])
                old_value_f = float(old_values[i])
                ret_tensor  = Tensor(np.array([ret]))

                critic_loss_unclipped_f = (ret - new_value_f) ** 2
                value_clipped_f = np.clip(
                    new_value_f,
                    old_value_f - self.value_clip,
                    old_value_f + self.value_clip,
                )
                critic_loss_clipped_f = (ret - value_clipped_f) ** 2

                if critic_loss_clipped_f >= critic_loss_unclipped_f:
                    # Clipped loss is worse  use it to prevent over-optimisation
                    critic_loss = Tensor(np.array([critic_loss_clipped_f]))
                else:
                    # Normal case: gradient flows through new_value
                    critic_loss = (ret_tensor - new_value) ** 2

                loss = (actor_loss + Tensor(np.array([0.5])) * critic_loss).sum()

                if not np.isfinite(loss.data).all():
                    self.optimizer.zero_grad()
                    if self.extractor_optimizer:
                        self.extractor_optimizer.zero_grad()
                    continue  # skip this sample, don't corrupt weights

                self.optimizer.zero_grad()
                if self.extractor_optimizer:
                    self.extractor_optimizer.zero_grad()

                loss.backward()


                for p in self.optimizer.parameters:
                    np.clip(p.grad, -1.0, 1.0, out=p.grad)
                self.optimizer.step()

                if self.extractor_optimizer:
                    for p in self.extractor_optimizer.parameters:
                        np.clip(p.grad, -0.5, 0.5, out=p.grad)
                    self.extractor_optimizer.step()

        # Decay exploration noise
        self.std = max(self.std_min, self.std * self.std_decay)
        #added a small constant to prevent std from decaying too quickly at the start, which can lead to premature convergence on suboptimal policies due to insufficient exploration in the early stages of training.
        # Clear rollout buffers
        self.states, self.actions, self.rewards, self.log_probs, self.values = \
            [], [], [], [], []
        
    def reset_critic(self):
        """Reset critic weights only — actor and extractor weights are preserved."""
        self.critic_l1    = Linear(self.critic_l1.W.data.shape[0], self.critic_l1.W.data.shape[1])
        self.critic_norm1 = LayerNorm(self.critic_norm1.num_features)
        self.critic_l2    = Linear(self.critic_l2.W.data.shape[0], self.critic_l2.W.data.shape[1])
        self.critic_norm2 = LayerNorm(self.critic_norm2.num_features)
        self.critic_out   = Linear(self.critic_out.W.data.shape[0], self.critic_out.W.data.shape[1])

        # rebuild optimizer with fresh critic params
        head_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_out.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )
        self.optimizer = Adam_Optimiser(head_params, lr=3e-4)
        print("Critic weights reset — actor and extractors preserved.")
        