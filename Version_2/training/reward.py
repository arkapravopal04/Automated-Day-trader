"""
training/reward.py

Sharpe-shaped reward: rewards risk-adjusted return, not raw PnL, so the
policy isn't rewarded for a high-variance strategy that happened to have a
good mean return during training (a bull-run-survivorship failure mode --
see directional_bias_fix.pdf and the return-mirroring logic already added to
vec_trading_env.py).

A plain end-of-episode Sharpe ratio can't be used directly as a per-step RL
reward -- it needs the full return series. This implements the
DIFFERENTIAL Sharpe ratio (Moody & Saffell, 1998; Moody, Wu, Liao & Saffell,
1998): an online, step-wise approximation to d(Sharpe)/dt, built from
exponential moving estimates of the first and second moment of returns. It
gives a well-defined reward at every step while still optimizing toward the
same risk-adjusted objective a full-episode Sharpe ratio would.

    A_t = A_{t-1} + eta * (R_t - A_{t-1})            (EMA of return)
    B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})           (EMA of squared return)
    D_t = (B_{t-1}*dA_t - 0.5*A_{t-1}*dB_t) / (B_{t-1} - A_{t-1}^2)**1.5

where dA_t = R_t - A_{t-1}, dB_t = R_t^2 - B_{t-1}. D_t is proportional to
the derivative of the Sharpe ratio with respect to this step's return, given
everything observed so far -- maximizing sum(D_t) over a trajectory is
(to first order) equivalent to maximizing the Sharpe ratio of the whole
trajectory.

Checkpoint bonus (added on top of the differential Sharpe term):
Per-env cumulative simple return since the last reset() is tracked
separately from the Sharpe moment estimates above (compounded from the same
`step_return` the caller already passes into step()). Every time that
cumulative return crosses a NEW multiple of `checkpoint_step` further out
from zero than any level previously reached this episode, a one-time bonus
of `checkpoint_bonus_frac` * D_t is added to the reward for that step.

This is deliberately symmetric: a stream that is short and whose cumulative
return crosses +2.5% gets exactly the same bonus treatment as a stream that
is long and crosses +2.5% (profit is profit, `step_return` already nets out
direction -- see vec_trading_env.py's own module docstring on why this
project treats long/short symmetry as a first-class concern). The mirror
image is also symmetric: crossing a NEW loss milestone (-2.5%, -5%, ...)
applies an equal-magnitude subtraction, not a free pass -- a milestone
system that only ever adds reward on the way up and never corrects for the
way down would just be a disguised long-only (or short-only) bias, exactly
what the mirroring/diversity-bonus machinery elsewhere in this project
exists to avoid.

Each level is "spent" once: bouncing back and forth across the same
threshold doesn't re-trigger it (see _max_level_pos / _max_level_neg
below) -- like a checkpoint in a game, once reached it stays reached for
the rest of the episode, so the bonus rewards making genuine new progress,
not oscillating around a boundary.
"""

from typing import Optional

import torch

Tensor = torch.Tensor


class DifferentialSharpeReward:
    def __init__(
        self,
        n_envs: int,
        eta: float = 0.01,
        eps: float = 1e-8,
        warmup_steps: int = 2,
        clip: float = 10.0,
        enable_checkpoint_bonus: bool = True,
        checkpoint_step: float = 0.025,
        checkpoint_bonus_frac: float = 0.10,
        device: Optional[str] = None,
    ) -> None:
        """
        eta: EMA decay for the return-moment estimates. Larger eta adapts
            faster to a regime change but makes the reward noisier -- this
            is the main knob to tune if the shaped reward looks too jumpy
            during training.
        warmup_steps: number of steps per env, counted from the most recent
            reset(), during which D_t is forced to 0 instead of computed.
            Right after a reset, B_{t-1} - A_{t-1}^2 (the variance estimate)
            is ~0, so the (·)**1.5 denominator is near-zero and D_t would
            otherwise spike to a large, meaningless value on the first few
            steps of every episode.
        clip: hard clamp on |D_t| after computing it. The denominator can
            still get small during a genuinely low-volatility patch later in
            an episode (not just right after reset) and produce an
            oversized reward spike -- this is a safety rail on top of
            warmup_steps, not a replacement for it.
        enable_checkpoint_bonus: toggles the milestone bonus/penalty
            described in the module docstring. Set False to recover the
            plain differential-Sharpe reward with no checkpoint shaping.
        checkpoint_step: size, in cumulative simple return, of each
            milestone (0.025 == 2.5%). Milestones repeat at every multiple
            of this value in both directions (2.5%, 5%, 7.5%, ... and
            -2.5%, -5%, -7.5%, ...).
        checkpoint_bonus_frac: fraction of that step's |D_t| added (on a
            new upside milestone) or subtracted (on a new downside
            milestone) per level newly crossed this step. 0.10 == a 10%
            bonus/penalty on D_t per new checkpoint.
        """
        self.n_envs = n_envs
        self.eta = eta
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.clip = clip
        self.enable_checkpoint_bonus = enable_checkpoint_bonus
        self.checkpoint_step = checkpoint_step
        self.checkpoint_bonus_frac = checkpoint_bonus_frac
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._A = torch.zeros(self.n_envs, device=self.device)
        self._B = torch.zeros(self.n_envs, device=self.device)
        self._steps_since_reset = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

        # --- checkpoint-bonus state (independent of the Sharpe moment
        # estimates above; reset alongside them in reset())
        self._cum_return = torch.zeros(self.n_envs, device=self.device)
        self._max_level_pos = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self._max_level_neg = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

        # --- diagnostics from the most recent step(), for optional logging
        # by the caller (e.g. train.py's tick_callback). Not part of the
        # return value so existing call sites (`reward_shaper.step(...)`)
        # keep working unchanged; read these attributes after step() if
        # you want them.
        self.last_dsr: Tensor = torch.zeros(self.n_envs, device=self.device)
        self.last_checkpoint_bonus: Tensor = torch.zeros(self.n_envs, device=self.device)

    def reset(self, env_mask: Optional[Tensor] = None) -> None:
        if env_mask is None:
            self._A.zero_()
            self._B.zero_()
            self._steps_since_reset.zero_()
            self._cum_return.zero_()
            self._max_level_pos.zero_()
            self._max_level_neg.zero_()
        else:
            env_mask = env_mask.to(self.device)
            self._A[env_mask] = 0.0
            self._B[env_mask] = 0.0
            self._steps_since_reset[env_mask] = 0
            self._cum_return[env_mask] = 0.0
            self._max_level_pos[env_mask] = 0
            self._max_level_neg[env_mask] = 0

    def step(self, step_return: Tensor) -> Tensor:
        """
        step_return: [n_envs], this step's simple return. Use
            info["step_pnl"] / equity_before from vec_trading_env.py's
            StepResult (equity_before computed BEFORE the fill, i.e. the
            equity the env's reward math itself uses) rather than
            equity_after / equity_before - 1, so this stays consistent with
            how the env already guards equity_before against being ~0.

        Returns the shaped reward D_t (differential Sharpe, plus the
        checkpoint bonus/penalty if enabled), [n_envs], clamped to
        +/- self.clip before the checkpoint adjustment is applied.
        """
        step_return = step_return.to(self.device)

        D = self._differential_sharpe(step_return)
        bonus = self._checkpoint_bonus(step_return, D)

        self.last_dsr = D
        self.last_checkpoint_bonus = bonus
        self._steps_since_reset = self._steps_since_reset + 1

        return D + bonus

    def _differential_sharpe(self, step_return: Tensor) -> Tensor:
        """Computes D_t per the module docstring's formula, from the PRE-update A/B, then updates A/B in place."""
        A_prev, B_prev = self._A, self._B
        dA = step_return - A_prev
        dB = step_return.pow(2) - B_prev

        variance_est = (B_prev - A_prev.pow(2)).clamp(min=self.eps)
        denom = variance_est.pow(1.5)

        D = (B_prev * dA - 0.5 * A_prev * dB) / denom
        D = D.clamp(min=-self.clip, max=self.clip)

        is_warm = self._steps_since_reset >= self.warmup_steps
        D = torch.where(is_warm, D, torch.zeros_like(D))

        # Update the moving averages AFTER computing D_t -- D_t is defined in
        # terms of the PREVIOUS A, B (see the formula in the module
        # docstring), not the post-update ones.
        self._A = A_prev + self.eta * dA
        self._B = B_prev + self.eta * dB

        return D

    def _checkpoint_bonus(self, step_return: Tensor, D: Tensor) -> Tensor:
        """
        Compounds step_return into per-env cumulative return, then grants a
        one-time +/- checkpoint_bonus_frac * |D_t| adjustment for each NEW
        milestone (in either direction, independently tracked -- see module
        docstring) crossed this step. Returns 0 for every env when
        enable_checkpoint_bonus is False.
        """
        if not self.enable_checkpoint_bonus:
            return torch.zeros_like(D)

        self._cum_return = (1.0 + self._cum_return) * (1.0 + step_return) - 1.0

        level_signed = torch.trunc(self._cum_return / self.checkpoint_step).long()
        pos_level = level_signed.clamp(min=0)
        neg_level = (-level_signed).clamp(min=0)

        new_pos_crossings = (pos_level - self._max_level_pos).clamp(min=0)
        new_neg_crossings = (neg_level - self._max_level_neg).clamp(min=0)

        self._max_level_pos = torch.maximum(self._max_level_pos, pos_level)
        self._max_level_neg = torch.maximum(self._max_level_neg, neg_level)

        bonus_magnitude = self.checkpoint_bonus_frac * D.abs()
        return bonus_magnitude * (new_pos_crossings.float() - new_neg_crossings.float())