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
"""

from typing import Optional

import torch


class DifferentialSharpeReward:
    def __init__(
        self,
        n_envs: int,
        eta: float = 0.01,
        eps: float = 1e-8,
        warmup_steps: int = 2,
        clip: float = 10.0,
        device: Optional[str] = None,
    ):
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
        """
        self.n_envs = n_envs
        self.eta = eta
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.clip = clip
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._A = torch.zeros(self.n_envs, device=self.device)
        self._B = torch.zeros(self.n_envs, device=self.device)
        self._steps_since_reset = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        if env_mask is None:
            self._A.zero_()
            self._B.zero_()
            self._steps_since_reset.zero_()
        else:
            env_mask = env_mask.to(self.device)
            self._A[env_mask] = 0.0
            self._B[env_mask] = 0.0
            self._steps_since_reset[env_mask] = 0

    def step(self, step_return: torch.Tensor) -> torch.Tensor:
        """
        step_return: [n_envs], this step's simple return. Use
            info["step_pnl"] / equity_before from vec_trading_env.py's
            StepResult (equity_before computed BEFORE the fill, i.e. the
            equity the env's reward math itself uses) rather than
            equity_after / equity_before - 1, so this stays consistent with
            how the env already guards equity_before against being ~0.

        Returns D_t, [n_envs], clamped to +/- self.clip.
        """
        step_return = step_return.to(self.device)

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
        self._steps_since_reset = self._steps_since_reset + 1

        return D