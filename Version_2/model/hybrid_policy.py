"""
model/hybrid_policy.py

Two-stage hybrid action head:
    Stage 1 (discrete):   Categorical over {SHORT, FLAT, LONG}.
    Stage 2 (continuous, conditioned on stage 1's index): size in (0,1) and
        limit_offset in (0,1), both modeled as Beta distributions (bounded
        support -> no tanh-squash entropy/log-prob distortion at the action
        bounds). The continuous head is conditioned on an embedding of the
        discrete action, so "how much" is learned given "which direction."

THIS IS THE HIGHEST-RISK FILE IN THE PROJECT. The architecture is simple;
almost every real bug here will be in distribution/log-prob correctness,
not network shapes. Three things this file is careful about, and that any
future change to it needs to preserve:

1. Beta concentration floor. Raw network outputs are mapped to
   alpha, beta > 1 (via softplus(x) + min_concentration, min_concentration
   >= 1.0 by default) rather than letting either parameter fall below 1.
   A Beta(alpha, beta) with alpha or beta < 1 is U-shaped: density blows up
   toward the boundary it favors, log_prob can spike or go to -inf near 0/1,
   and PPO's ratio/clipping math becomes numerically unstable exactly on the
   actions the policy is most confident about. Keeping both parameters > 1
   guarantees a finite, unimodal density everywhere on (0, 1), including a
   well-defined mode, which _beta_mode() below relies on.

2. Masked continuous log-prob / entropy for FLAT. When the sampled (or
   given) discrete action is FLAT, no order is actually submitted -- size
   and limit_offset are meaningless for that step, not "a size of whatever
   the Beta happened to sample." Their log-prob and entropy contributions
   are masked to 0 rather than included, so:
     - PPO's importance ratio isn't polluted by a nonsense continuous
       log-prob on FLAT steps (which would otherwise inject gradient signal
       into a component of the action that was never actually used).
     - The entropy bonus doesn't reward or punish exploration on a dimension
       that isn't live that step.
   Removing this masking will not raise an error; it will just quietly teach
   the policy a spurious direction/size coupling on FLAT steps and add noise
   to the advantage estimates for no benefit -- exactly the kind of bug that
   doesn't show up until you're staring at a training curve wondering why.

3. Same discrete index at rollout AND update. act() samples direction_idx
   and returns it in the HybridActionSample; evaluate_actions() MUST be
   called with that SAME stored direction_idx (never a freshly re-sampled
   one) to recompute log_prob/entropy under the current parameters for the
   PPO ratio. The continuous heads are conditioned on the discrete
   embedding, so evaluating with a different index than what was actually
   taken at rollout time silently breaks the importance-sampling assumption
   PPO depends on. There is no shape-level error to catch this -- only a
   subtly-wrong-but-runnable training loop.

Everything here operates in NORMALIZED action space: size and limit_offset
are both in (0, 1). Rescaling to actual shares (post risk_manager.py /
kelly_sizing.py) and actual tick offsets happens in the two static
rescale_* helpers at the bottom, kept deliberately separate from the
distribution math above -- this file has no notion of equity, price, or
risk limits, and shouldn't.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical

SHORT_IDX, FLAT_IDX, LONG_IDX = 0, 1, 2
# index -> {-1, 0, 1}, matching execution_sim.py's / vec_trading_env.py's direction convention
IDX_TO_DIRECTION = torch.tensor([-1.0, 0.0, 1.0])

# keeps Beta.log_prob() away from the exact 0/1 boundary -- density is finite
# there given alpha, beta > 1, but log_prob(0.0) / log_prob(1.0) can still
# round to -inf/nan in float32, and rsample() can (rarely) underflow to
# exactly 0.0 or 1.0
_EPS = 1e-6


@dataclass
class HybridActionSample:
    direction: torch.Tensor      # [batch], in {-1., 0., 1.} -- what execution_sim.py / vec_trading_env.py expect
    direction_idx: torch.Tensor  # [batch] long, in {0,1,2} -- MUST be passed back into evaluate_actions() unchanged
    size: torch.Tensor           # [batch], in (0, 1) -- raw Beta-space sample, NOT yet rescaled to shares
    limit_offset: torch.Tensor   # [batch], in (0, 1) -- raw Beta-space sample, NOT yet rescaled to ticks
    log_prob: torch.Tensor       # [batch], joint log-prob of (direction_idx, size, limit_offset) under this policy
    entropy: torch.Tensor        # [batch], joint entropy (discrete + FLAT-masked continuous)


class HybridPolicyHead(nn.Module):
    def __init__(
        self,
        trunk_dim: int,
        direction_embed_dim: int = 16,
        hidden_dim: int = 64,
        min_concentration: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.min_concentration = min_concentration

        self.direction_head = nn.Linear(trunk_dim, 3)
        self.direction_embed = nn.Embedding(3, direction_embed_dim)

        cond_dim = trunk_dim + direction_embed_dim

        def make_continuous_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),  # raw (alpha, beta) pre-transform
            )

        self.size_head = make_continuous_head()
        self.limit_offset_head = make_continuous_head()

    # --- internal helpers ----------------------------------------------

    def _to_concentration(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """raw: [batch, 2] -> (alpha, beta), both > min_concentration. See module docstring point 1."""
        alpha = nn.functional.softplus(raw[..., 0]) + self.min_concentration
        beta = nn.functional.softplus(raw[..., 1]) + self.min_concentration
        return alpha, beta

    def _continuous_dists(self, trunk: torch.Tensor, direction_idx: torch.Tensor) -> Tuple[Beta, Beta]:
        embed = self.direction_embed(direction_idx)
        cond = torch.cat([trunk, embed], dim=-1)
        size_alpha, size_beta = self._to_concentration(self.size_head(cond))
        limit_alpha, limit_beta = self._to_concentration(self.limit_offset_head(cond))
        return Beta(size_alpha, size_beta), Beta(limit_alpha, limit_beta)

    @staticmethod
    def _beta_mode(dist: Beta) -> torch.Tensor:
        """
        Mode of Beta(alpha, beta) = (alpha - 1) / (alpha + beta - 2).
        Well-defined and finite almost everywhere here because both alpha,
        beta >= min_concentration >= 1 always (see module docstring point 1).

        One degenerate case the closed-form formula can't express: if alpha
        AND beta both land on exactly min_concentration (e.g. both softplus
        underflow to 0 under extreme/saturated trunk activations -- verified
        reachable, not just theoretical), the distribution is Beta(1,1), the
        uniform distribution: every point in (0,1) is equally a mode, and the
        formula divides 0/0 -> NaN. Since this is specifically the
        deterministic=True path used for backtest/live decisions, a NaN here
        would silently propagate through rescale_size() into execution_sim.py
        rather than raising anything. Guarded by falling back to 0.5 (the
        natural midpoint) whenever the denominator is ~0, instead of dividing.
        """
        alpha, beta = dist.concentration1, dist.concentration0
        denom = alpha + beta - 2.0
        degenerate = denom.abs() < 1e-6
        safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
        mode = torch.where(degenerate, torch.full_like(denom, 0.5), (alpha - 1.0) / safe_denom)
        return mode

    # --- rollout ---------------------------------------------------------

    def act(self, trunk: torch.Tensor, deterministic: bool = False) -> HybridActionSample:
        """
        trunk: [batch, trunk_dim] (fusion.py's FusionTrunk output)

        deterministic=True returns the mode of every distribution instead of
        sampling -- for backtest/eval only. PPO rollout collection needs
        stochastic samples with valid log_probs; don't set this True there.
        """
        direction_logits = self.direction_head(trunk)
        direction_dist = Categorical(logits=direction_logits)
        direction_idx = direction_logits.argmax(dim=-1) if deterministic else direction_dist.sample()

        size_dist, limit_dist = self._continuous_dists(trunk, direction_idx)

        if deterministic:
            size = self._beta_mode(size_dist)
            limit_offset = self._beta_mode(limit_dist)
        else:
            size = size_dist.sample()
            limit_offset = limit_dist.sample()

        log_prob, discrete_entropy, continuous_entropy = self._joint_log_prob_and_split_entropy(
            direction_dist, direction_idx, size_dist, limit_dist, size, limit_offset
        )
        entropy = discrete_entropy + continuous_entropy

        direction = IDX_TO_DIRECTION.to(trunk.device)[direction_idx]

        return HybridActionSample(
            direction=direction,
            direction_idx=direction_idx,
            size=size,
            limit_offset=limit_offset,
            log_prob=log_prob,
            entropy=entropy,
        )

    # --- PPO update --------------------------------------------------------

    def evaluate_actions(
        self,
        trunk: torch.Tensor,
        direction_idx: torch.Tensor,  # MUST be the exact idx returned by act() at rollout time -- see docstring point 3
        size: torch.Tensor,           # raw Beta-space value stored from rollout, in (0, 1)
        limit_offset: torch.Tensor,   # raw Beta-space value stored from rollout, in (0, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recomputes (log_prob, entropy) for a previously-taken action under
        the CURRENT policy parameters -- this is what PPO's ratio
        (exp(new_log_prob - old_log_prob)) and entropy bonus are built from.
        Returns (log_prob, entropy), both [batch]. `entropy` here is the
        joint (discrete + masked continuous) sum -- use
        evaluate_actions_split() instead if you need the two components
        weighted by different entropy coefficients (training/ppo_hybrid.py
        does).
        """
        log_prob, discrete_entropy, continuous_entropy = self.evaluate_actions_split(
            trunk, direction_idx, size, limit_offset
        )
        return log_prob, discrete_entropy + continuous_entropy

    def evaluate_actions_split(
        self,
        trunk: torch.Tensor,
        direction_idx: torch.Tensor,
        size: torch.Tensor,
        limit_offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as evaluate_actions(), but returns the discrete and (masked)
        continuous entropy separately instead of pre-summed, so a training
        loop can apply independent entropy coefficients to each head.
        log_prob remains a single joint value -- the PPO ratio is always
        computed on the whole action, never per-component.
        Returns (log_prob, discrete_entropy, continuous_entropy), all [batch].
        """
        direction_logits = self.direction_head(trunk)
        direction_dist = Categorical(logits=direction_logits)
        size_dist, limit_dist = self._continuous_dists(trunk, direction_idx)
        return self._joint_log_prob_and_split_entropy(
            direction_dist, direction_idx, size_dist, limit_dist, size, limit_offset
        )

    @staticmethod
    def _joint_log_prob_and_split_entropy(
        direction_dist: Categorical,
        direction_idx: torch.Tensor,
        size_dist: Beta,
        limit_dist: Beta,
        size: torch.Tensor,
        limit_offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size_c = size.clamp(_EPS, 1.0 - _EPS)
        limit_c = limit_offset.clamp(_EPS, 1.0 - _EPS)

        # See module docstring point 2: continuous terms are masked to 0 on
        # FLAT steps, not included, since size/limit_offset were never
        # actually used for an order that step.
        continuous_mask = (direction_idx != FLAT_IDX).float()

        log_prob = (
            direction_dist.log_prob(direction_idx)
            + continuous_mask * (size_dist.log_prob(size_c) + limit_dist.log_prob(limit_c))
        )
        discrete_entropy = direction_dist.entropy()
        continuous_entropy = continuous_mask * (size_dist.entropy() + limit_dist.entropy())
        return log_prob, discrete_entropy, continuous_entropy

    # --- action-space rescaling (kept separate from distribution math) -----

    @staticmethod
    def rescale_size(size_raw: torch.Tensor, max_size: torch.Tensor) -> torch.Tensor:
        """
        size_raw: (0, 1) from HybridActionSample.size.
        max_size: [batch], the actual share cap for this step (e.g. from
            kelly_sizing.py's headroom, or a fixed per-order cap) -- computed
            OUTSIDE this module.
        Returns requested shares, unsigned, ready for execution_sim.py.
        """
        return size_raw * max_size

    @staticmethod
    def rescale_limit_offset(limit_offset_raw: torch.Tensor, max_offset_ticks: float) -> torch.Tensor:
        """
        limit_offset_raw: (0, 1) from HybridActionSample.limit_offset.
        Maps to [-max_offset_ticks, +max_offset_ticks] via an affine
        transform (not [0, max_offset_ticks]) since execution_sim.py treats
        limit_offset as a signed adjustment in the agent's favor, not an
        unsigned magnitude.
        """
        return (limit_offset_raw * 2.0 - 1.0) * max_offset_ticks