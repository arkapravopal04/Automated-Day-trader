'''
just the agent bot at its final stage, uses all the different things i learnt, 
'''

import sys
import numpy as np
from engine import Tensor
from Neural_Nets import LayerNorm, Dropout, Linear, Adam_Optimiser
from risk import RISK_FEATURE_SIZE


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""

    def __init__(self, epsilon: float = 1e-4, min_samples: int = 200):
        self.mean        = 0.0
        self.var         = 1.0
        self.count       = epsilon
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
        if self.count < self.min_samples:
            return x
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class PPOAgent:
    def __init__(self, state_size=75, action_size=2,
                 lstm=None, attention=None, cnn=None, flatten=None,
                 regime=None, fusion=None, num_envs=1):

        self.gamma   = 0.98
        self.epsilon = 0.25
        self.epochs  = 4
        self.lam     = 0.97

        self.std           = 0.45
        self.std_min       = 0.2
        self.std_decay     = 0.998
        self.std_threshold = 0.25

        self.entropy_coef_start = 0.02
        self.entropy_coef_min   = 0.001
        self.entropy_decay      = 0.992
        self.entropy_coef       = self.entropy_coef_start

        self.episode_count = 0
        self.value_clip    = 0.20

        self.head_lr      = 1e-4
        self.extractor_lr = 2e-05
        self.reset_lr     = 3e-4

        self.return_rms = RunningMeanStd()

        self.lstm      = lstm
        self.attention = attention
        self.cnn       = cnn
        self.flatten   = flatten
        self.regime    = regime
        self.fusion    = fusion

        self.actor_l1    = Linear(state_size, 64)
        self.actor_norm1 = LayerNorm(64)
        self.actor_drop1 = Dropout(0.0)
        self.actor_l2    = Linear(64, 32)
        self.actor_norm2 = LayerNorm(32)
        self.actor_drop2 = Dropout(0.0)
        self.actor_dir   = Linear(32, 3)
        self.actor_size  = Linear(32, 1)

        self.critic_l1    = Linear(state_size, 64)
        self.critic_norm1 = LayerNorm(64)
        self.critic_drop1 = Dropout(0.0)
        self.critic_l2    = Linear(64, 32)
        self.critic_norm2 = LayerNorm(32)
        self.critic_drop2 = Dropout(0.0)
        self.critic_out   = Linear(32, 1)

        head_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_dir.parameters() + self.actor_size.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )
        extractor_params      = self._extractor_parameters()
        fusion_params         = self.fusion.parameters() if self.fusion is not None else []
        extractor_only_params = [p for p in extractor_params
                                 if id(p) not in {id(fp) for fp in fusion_params}]

        self.optimizer           = Adam_Optimiser(head_params,           lr=self.head_lr)
        self.extractor_optimizer = (
            Adam_Optimiser(extractor_only_params, lr=self.extractor_lr)
            if extractor_only_params else None
        )
        self.fusion_optimizer    = (
            Adam_Optimiser(fusion_params, lr=self.extractor_lr)
            if fusion_params else None
        )

        self.param_ids = {
            id(p) for p in (
                head_params + extractor_params +
                (self.fusion.parameters() if self.fusion is not None else [])
            )
        }

        # Initialize vectorized buffers
        self.init_buffers(num_envs)

    def init_buffers(self, num_envs):
        self.num_envs      = num_envs
        self.env_states    = [[] for _ in range(num_envs)]
        self.env_actions   = [[] for _ in range(num_envs)]
        self.env_rewards   = [[] for _ in range(num_envs)]
        self.env_log_probs = [[] for _ in range(num_envs)]
        self.env_values    = [[] for _ in range(num_envs)]
        self.env_dones     = [[] for _ in range(num_envs)]

        # TBPTT buffers: raw inputs and hidden state snapshots at chunk boundaries
        self.env_raw_inputs = [[] for _ in range(num_envs)]   # raw X[idx] per step
        self.env_lstm_h     = [[] for _ in range(num_envs)]   # h snapshot per step (None except at boundaries)
        self.env_lstm_c     = [[] for _ in range(num_envs)]   # c snapshot per step (None except at boundaries)

        self._last_states      = None
        self._last_action_idxs = None
        self._last_dir_probs   = None
        self._last_raw_sizes   = None
        self._last_size_means  = None
        self._last_values      = None

    # ── helpers ────────────────────────────────────────────────────────────────

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
        return self.actor_dir(x), self.actor_size(x)

    def _critic_forward(self, state):
        v = self.critic_drop1(self.critic_norm1(self.critic_l1(state).relu()))
        v = self.critic_drop2(self.critic_norm2(self.critic_l2(v).relu()))
        return self.critic_out(v)

    def _zero_state_grads(self, state_tensor):
        visited = set()
        stack   = [state_tensor]
        while stack:
            t = stack.pop()
            if id(t) in visited:
                continue
            visited.add(id(t))
            if id(t) not in self.param_ids:
                if hasattr(t, 'grad') and t.grad is not None:
                    t.grad = np.zeros_like(t.grad)
            if hasattr(t, '_prev'):
                stack.extend(t._prev)

    # ── vectorized action selection ────────────────────────────────────────────

    def select_vectorized_action(self, states_list):
        self._set_train_mode(False)

        # Stack into [num_envs, state_size] parallel batch
        batch_data = np.vstack([s.data.flatten() for s in states_list])
        batch_tensor = Tensor(batch_data)

        # ONE forward pass for the entire batch
        dir_logits, size_out = self._actor_forward(batch_tensor)

        # Row-wise Softmax
        logits_data = dir_logits.data
        max_logits  = np.max(logits_data, axis=1, keepdims=True)
        exp_logits  = np.exp(logits_data - max_logits)
        probs       = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)

        # Sample independent actions per environment
        action_indices = [np.random.choice([0, 1, 2], p=p) for p in probs]
        directions     = [-1.0 if a == 0 else 0.0 if a == 1 else 1.0 for a in action_indices]

        # Calculate continuous sizes per environment
        size_means = size_out.sigmoid().data.flatten()
        raw_sizes  = size_means + np.random.normal(0, self.std, size=len(size_means))
        sizes      = np.clip(raw_sizes, 0.0, 1.0)

        values = self._critic_forward(batch_tensor).data.flatten()

        # Cache batched results to store in memory after env.step()
        self._last_states      = states_list 
        self._last_action_idxs = action_indices
        self._last_dir_probs   = probs
        self._last_raw_sizes   = raw_sizes
        self._last_size_means  = size_means
        self._last_values      = values

        return np.column_stack((directions, sizes))

    def get_vectorized_values(self, states_list):
        """Used purely to bootstrap terminal values at end of rollout."""
        self._set_train_mode(False)
        batch_data = np.vstack([
            s.data.flatten() if s is not None else np.zeros(75) 
            for s in states_list
        ])
        batch_tensor = Tensor(batch_data)
        values = self._critic_forward(batch_tensor).data.flatten()
        return [float(v) for v in values]

    def store_vectorized_transition(self, rewards, dones, adjusted_actions=None,
                                        raw_inputs=None, lstm_h_snaps=None, lstm_c_snaps=None):
        if self._last_states is None:
            return

        for i in range(self.num_envs):
            action_idx = self._last_action_idxs[i]
            probs      = self._last_dir_probs[i]
            size_mean  = self._last_size_means[i]
            value      = self._last_values[i]

            # Use adjusted size if available, else fall back to raw
            if adjusted_actions is not None:
                stored_size = float(adjusted_actions[i][1])
            else:
                stored_size = float(self._last_raw_sizes[i])

            log_prob_dir  = np.log(probs[action_idx] + 1e-8)
            log_prob_size = -0.5 * ((stored_size - size_mean) / (self.std + 1e-8)) ** 2
            log_prob      = log_prob_dir + log_prob_size

            direction = float([-1.0, 0.0, 1.0][action_idx])
            action    = np.array([direction, stored_size])

            # Store detached fused state (for fallback / value retrieval only)
            self.env_states[i].append(self._last_states[i].detach())
            self.env_actions[i].append(action)
            self.env_rewards[i].append(float(rewards[i]))
            self.env_log_probs[i].append(log_prob)
            self.env_values[i].append(value)
            self.env_dones[i].append(dones[i])

            # Store raw input for TBPTT replay
            if raw_inputs is not None:
                self.env_raw_inputs[i].append(raw_inputs[i].detach())
            # Store hidden state snapshot (None at non-boundary steps)
            self.env_lstm_h[i].append(lstm_h_snaps[i] if lstm_h_snaps is not None else None)
            self.env_lstm_c[i].append(lstm_c_snaps[i] if lstm_c_snaps is not None else None)

    # ── GAE / returns per env ──────────────────────────────────────────────────

    def compute_returns_for_env(self, env_idx, next_value=0.0):
        rewards = self.env_rewards[env_idx]
        values  = self.env_values[env_idx]
        dones   = self.env_dones[env_idx]
        n_steps = len(rewards)

        gae = 0.0
        gae_advantages = np.zeros(n_steps, dtype=np.float64)
        td_returns     = np.zeros(n_steps, dtype=np.float64)
        values_ext     = values + [float(next_value)]

        for t in reversed(range(n_steps)):
            # If done flag triggered, next state value is 0
            next_val = 0.0 if dones[t] else values_ext[t + 1]
            delta    = rewards[t] + self.gamma * next_val - values_ext[t]

            # Mask GAE accumulation if boundary reached
            gae = delta + self.gamma * self.lam * gae * (1.0 - float(dones[t]))
            gae_advantages[t] = gae
            td_returns[t]     = gae + values_ext[t]

        td_returns = np.clip(td_returns, -30.0, 30.0)
        return td_returns, gae_advantages

    # ── full forward pass through all extractors ───────────────────────────────

    def _full_forward(self, raw_sample, h_prev, c_prev):
        """
        Replays the full pipeline: LSTM → Attention → CNN → Regime → Fusion → heads.
        raw_sample: Tensor [window, features]
        h_prev / c_prev: LSTM hidden state to seed from (detached boundary snapshot)
        Returns: (dir_logits, size_out, new_value, h_out, c_out)
        """
        # LSTM — carries hidden state through the chunk via internal TBPTT
        hidden_states, (h_out, c_out) = self.lstm(
            raw_sample, h_prev=h_prev, c_prev=c_prev
        )

        lstm_out = self.attention(hidden_states)

        window_size, num_features = raw_sample.data.shape
        cnn_input = Tensor(raw_sample.data.reshape(1, window_size, num_features))
        cnn_out   = self.flatten(self.cnn(cnn_input).tanh())

        regime_out = self.regime(raw_sample)

        # Risk features not available during replay — use zeros (same as NLP placeholder)
        risk_tensor = Tensor(np.zeros((1, RISK_FEATURE_SIZE), dtype=np.float64))

        l_f = lstm_out[-1].reshape(1, -1)
        c_f = cnn_out.reshape(1, -1)
        n_f = Tensor(np.zeros((1, 64), dtype=np.float64))   # NLP placeholder
        r_f = regime_out.reshape(1, -1)

        fused   = self.fusion(l_f, c_f, n_f, r_f, risk_tensor)
        f_flat  = fused.reshape(64)

        # Minimal portfolio features (zeros during replay — heads learn from fused signal)
        portfolio = Tensor(np.zeros(3, dtype=np.float64))
        risk_feat = Tensor(np.zeros(RISK_FEATURE_SIZE, dtype=np.float64))
        state     = f_flat.concat(portfolio).concat(risk_feat)

        dir_logits, size_out = self._actor_forward(state)
        new_value            = self._critic_forward(state)

        return dir_logits, size_out, new_value, h_out, c_out

    # ── PPO update ─────────────────────────────────────────────────────────────

    def update(self, next_values: list):
        self.episode_count += 1
        CHUNK_SIZE = 32   # TBPTT chunk — safe for 16GB, 6 cores

        # ── Step 1: GAE over stored scalar rewards/values (no graph needed) ───
        all_actions    = []
        all_log_probs  = []
        all_values     = []
        all_returns    = []
        all_advantages = []
        all_raw        = []    # raw inputs per step, per env, in order
        all_h_snaps    = []    # h snapshot at chunk boundary (or None)
        all_c_snaps    = []    # c snapshot at chunk boundary (or None)
        all_env_ids    = []    # which env each step belongs to

        for i in range(self.num_envs):
            if not self.env_raw_inputs[i]:
                continue
            ret, adv = self.compute_returns_for_env(i, next_values[i])

            n = len(self.env_raw_inputs[i])
            all_actions.extend(self.env_actions[i])
            all_log_probs.extend(self.env_log_probs[i])
            all_values.extend(self.env_values[i])
            all_returns.extend(ret)
            all_advantages.extend(adv)
            all_raw.extend(self.env_raw_inputs[i])
            all_h_snaps.extend(self.env_lstm_h[i])
            all_c_snaps.extend(self.env_lstm_c[i])
            all_env_ids.extend([i] * n)

        n_samples = len(all_raw)
        if n_samples == 0:
            return 0.0, 0.0, 0.0

        actions       = all_actions
        old_log_probs = np.array(all_log_probs,  dtype=np.float64)
        old_values    = np.array(all_values,     dtype=np.float64)
        returns       = np.array(all_returns,    dtype=np.float64)
        advantages    = np.array(all_advantages, dtype=np.float64)

        if len(advantages) > 1:
            adv_mean   = advantages.mean()
            adv_std    = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            adv_mean = adv_std = 0.0

        _log(
            f"\n[PPO] Update ep={self.episode_count} | "
            f"Steps={n_samples} (TBPTT chunks={CHUNK_SIZE}) | "
            f"RetMean={float(np.mean(returns)):+.3f} RetStd={float(np.std(returns)):.3f} | "
            f"AdvMean={adv_mean:+.4f} AdvStd={adv_std:.4f} | "
            f"Std={self.std:.3f} EntropyCoef={self.entropy_coef:.4f}"
        )

        self._set_train_mode(True)

        head_norm_accum = []
        ext_norm_accum  = []
        fus_norm_accum  = []

        def _grad_norm(params):
            return float(np.sqrt(sum(
                np.sum(p.grad ** 2)
                for p in params
                if p.grad is not None and np.any(p.grad != 0)
            )))

        for epoch in range(self.epochs):
            epoch_ratios   = []
            epoch_clipped  = 0
            epoch_valid    = 0
            epoch_val_errs = []

            # ── Process each env's steps as sequential chunks ─────────────────
            # Group indices by env to preserve temporal order within each env
            env_step_indices = {}
            for flat_idx, env_id in enumerate(all_env_ids):
                env_step_indices.setdefault(env_id, []).append(flat_idx)

            for env_id, step_indices in env_step_indices.items():
                # Carry hidden state across chunks within this env
                chunk_h = None
                chunk_c = None

                for chunk_start in range(0, len(step_indices), CHUNK_SIZE):
                    chunk_idx = step_indices[chunk_start : chunk_start + CHUNK_SIZE]

                    self.optimizer.zero_grad()
                    if self.extractor_optimizer:
                        self.extractor_optimizer.zero_grad()
                    if self.fusion_optimizer:
                        self.fusion_optimizer.zero_grad()

                    valid_samples = 0
                    batch_scale   = 1.0 / len(chunk_idx)

                    for flat_i in chunk_idx:
                        raw_sample = all_raw[flat_i]
                        action     = actions[flat_i]
                        old_log_p  = float(old_log_probs[flat_i])
                        adv        = float(advantages[flat_i])
                        ret        = float(returns[flat_i])

                        # Use stored boundary snapshot if available,
                        # otherwise carry forward from previous chunk
                        h_snap = all_h_snaps[flat_i]
                        c_snap = all_c_snaps[flat_i]
                        h_init = h_snap if h_snap is not None else chunk_h
                        c_init = c_snap if c_snap is not None else chunk_c

                        dir_logits, size_out, new_value, h_out, c_out = \
                            self._full_forward(raw_sample, h_init, c_init)

                        # Carry hidden state to next step within chunk
                        chunk_h = [Tensor(h.data.copy()) for h in h_out]
                        chunk_c = [Tensor(c.data.copy()) for c in c_out]

                        size_mean = size_out.sigmoid()
                        probs     = dir_logits.softmax(axis=-1)
                        log_probs = probs.log()

                        if action[0] < -0.5:
                            dir_idx = 0
                        elif action[0] > 0.5:
                            dir_idx = 2
                        else:
                            dir_idx = 1

                        new_log_p_dir    = log_probs[dir_idx]
                        diff_size        = Tensor(np.array([action[1]])) - size_mean
                        scaled_diff      = diff_size / (self.std + 1e-8)
                        new_log_p_size   = (scaled_diff * scaled_diff) * Tensor(np.array([-0.5]))
                        new_log_p_tensor = new_log_p_dir + new_log_p_size.reshape(1)
                        new_log_p        = float(new_log_p_tensor.data.flat[0])

                        ratio_val = np.exp(np.clip(new_log_p - old_log_p, -20.0, 2.0))
                        epoch_ratios.append(ratio_val)

                        surr1 = ratio_val * adv
                        surr2 = float(np.clip(ratio_val,
                                              1.0 - self.epsilon,
                                              1.0 + self.epsilon)) * adv

                        min_surr = min(surr1, surr2)
                        if min_surr == surr1:
                            actor_grad_coef = float(np.clip(-adv, -2.0, 2.0))
                            is_clipped      = False
                        else:
                            actor_grad_coef = 0.0
                            is_clipped      = True

                        if is_clipped:
                            epoch_clipped += 1

                        actor_loss = new_log_p_tensor * Tensor(np.array([actor_grad_coef]))
                        reg_dir    = (dir_logits * dir_logits).sum() * Tensor(np.array([0.005]))
                        reg_size   = (size_out * size_out) * Tensor(np.array([0.005]))
                        entropy_t  = -(probs * log_probs).sum()
                        actor_loss = (actor_loss + reg_dir + reg_size.reshape(1)
                                      - entropy_t * Tensor(np.array([self.entropy_coef])))

                        new_value_f = float(new_value.data.flat[0])
                        old_value_f = float(old_values[flat_i])
                        val_err     = ret - new_value_f
                        epoch_val_errs.append(abs(val_err))

                        critic_loss_unclipped_f = val_err ** 2
                        value_clipped_f = np.clip(
                            new_value_f,
                            old_value_f - self.value_clip,
                            old_value_f + self.value_clip,
                        )
                        critic_loss_clipped_f = (ret - value_clipped_f) ** 2

                        if critic_loss_unclipped_f > critic_loss_clipped_f:
                            critic_grad = np.clip(-2.0 * val_err, -3.0, 3.0)
                        else:
                            critic_grad = 0.0

                        critic_loss = new_value * Tensor(np.array([critic_grad]))
                        loss        = (actor_loss + Tensor(np.array([0.5])) * critic_loss).sum()
                        scaled_loss = loss * Tensor(np.array([batch_scale]))

                        if not np.isfinite(scaled_loss.data).all():
                            continue

                        scaled_loss.backward()
                        valid_samples += 1
                        epoch_valid   += 1

                    if valid_samples > 0:
                        hn = _grad_norm(self.optimizer.parameters)
                        head_norm_accum.append(hn)
                        self.optimizer.step()

                        if self.extractor_optimizer:
                            en = _grad_norm(self.extractor_optimizer.parameters)
                            ext_norm_accum.append(en)
                            self.extractor_optimizer.step()

                        if self.fusion_optimizer:
                            fn = _grad_norm(self.fusion_optimizer.parameters)
                            fus_norm_accum.append(fn)
                            self.fusion_optimizer.step()

            clip_frac    = epoch_clipped / max(epoch_valid, 1)
            ratio_arr    = np.array(epoch_ratios) if epoch_ratios else np.array([1.0])
            mean_val_err = float(np.mean(epoch_val_errs)) if epoch_val_errs else 0.0
            _log(
                f"  [Epoch {epoch+1}/{self.epochs}] "
                f"Valid={epoch_valid} | "
                f"ClipFrac={clip_frac:.2%} | "
                f"Ratio min={ratio_arr.min():.3f} mean={ratio_arr.mean():.3f} max={ratio_arr.max():.3f} | "
                f"ValErr={mean_val_err:.4f} | "
                f"HeadNorm={head_norm_accum[-1] if head_norm_accum else 0.0:.4f}"
            )

        # ── exploration decay ──────────────────────────────────────────────────
        if self.std * self.std_decay < self.std_threshold:
            self.std = max(self.std_min, self.std * self.std_decay + 0.1)
        else:
            self.std = max(self.std_min, self.std * self.std_decay)

        self.entropy_coef = max(self.entropy_coef_min,
                                self.entropy_coef * self.entropy_decay)

        head_norm = float(np.mean(head_norm_accum)) if head_norm_accum else 0.0
        ext_norm  = float(np.mean(ext_norm_accum))  if ext_norm_accum  else 0.0
        fus_norm  = float(np.mean(fus_norm_accum))  if fus_norm_accum  else 0.0

        _log(
            f"[PPO] Update done | "
            f"HeadNorm={head_norm:.4f} ExtNorm={ext_norm:.4f} FusNorm={fus_norm:.4f} | "
            f"Std={self.std:.4f} EntropyCoef={self.entropy_coef:.5f}"
        )

        # Clear all buffers explicitly before next episode
        for i in range(self.num_envs):
            del self.env_raw_inputs[i][:]
            del self.env_lstm_h[i][:]
            del self.env_lstm_c[i][:]
        self.init_buffers(self.num_envs)

        return head_norm, ext_norm, fus_norm

    # ── reset helpers ──────────────────────────────────────────────────────────

    def reset_critic(self):
        old_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_dir.parameters() + self.actor_size.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )
        old_m = self.optimizer.m
        old_v = self.optimizer.v
        old_t = self.optimizer.t

        self.critic_l1    = Linear(self.critic_l1.W.data.shape[0], 64)
        self.critic_norm1 = LayerNorm(64)
        self.critic_l2    = Linear(64, 32)
        self.critic_norm2 = LayerNorm(32)
        self.critic_out   = Linear(32, 1)

        new_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_dir.parameters() + self.actor_size.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )

        self.optimizer = Adam_Optimiser(new_params, lr=self.reset_lr)
        for old_p, new_p in zip(old_params, new_params):
            if id(old_p) in old_m:
                self.optimizer.m[id(new_p)] = old_m[id(old_p)]
                self.optimizer.v[id(new_p)] = old_v[id(old_p)]
        self.optimizer.t = old_t

        self.param_ids = {
            id(p) for p in (
                new_params + self._extractor_parameters() +
                (self.fusion.parameters() if self.fusion is not None else [])
            )
        }
        _log("[AGENT] Critic weights reset — actor and extractors preserved.")

    def reset_actor(self):
        old_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_dir.parameters() + self.actor_size.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )
        old_m = self.optimizer.m
        old_v = self.optimizer.v
        old_t = self.optimizer.t

        self.actor_l1    = Linear(self.actor_l1.W.data.shape[0], 64)
        self.actor_norm1 = LayerNorm(64)
        self.actor_l2    = Linear(64, 32)
        self.actor_norm2 = LayerNorm(32)
        self.actor_dir   = Linear(32, 3)
        self.actor_size  = Linear(32, 1)

        new_params = (
            self.actor_l1.parameters()  + self.actor_norm1.parameters() +
            self.actor_l2.parameters()  + self.actor_norm2.parameters() +
            self.actor_dir.parameters() + self.actor_size.parameters() +
            self.critic_l1.parameters() + self.critic_norm1.parameters() +
            self.critic_l2.parameters() + self.critic_norm2.parameters() +
            self.critic_out.parameters()
        )

        self.optimizer = Adam_Optimiser(new_params, lr=self.reset_lr)
        for old_p, new_p in zip(old_params, new_params):
            if id(old_p) in old_m:
                self.optimizer.m[id(new_p)] = old_m[id(old_p)]
                self.optimizer.v[id(new_p)] = old_v[id(old_p)]
        self.optimizer.t = old_t

        self.param_ids = {
            id(p) for p in (
                new_params + self._extractor_parameters() +
                (self.fusion.parameters() if self.fusion is not None else [])
            )
        }
        _log("[AGENT] Actor weights reset — critic and extractors preserved.")