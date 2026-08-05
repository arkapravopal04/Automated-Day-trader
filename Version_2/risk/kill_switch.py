"""
risk/kill_switch.py

Hard halt conditions, independent of risk_manager.py's graduated limits.
Where RiskManager scales an order down to stay within a limit, KillSwitch is
binary: once tripped for a stream, ALL new-exposure trading stops for that
stream until someone explicitly calls reset(). It never auto-clears itself
-- that decision is left to a human or a higher-level supervisor.

Designed to be polled from either a backtest step loop or live_loop.py, and
to be tripped from either code path:
    - live_loop.py calls record_broker_error() on an API exception and
      record_broker_success() on a clean round-trip.
    - Either loop calls check_daily_loss() once per step/sync with current
      equity, and check_state_mismatch() whenever a broker position sync
      happens.

Trip conditions:
    1. Daily realized+unrealized loss exceeds `daily_loss_limit_frac` of the
       equity marked at the start of that trading day (see start_new_day()).
    2. `broker_error_streak_limit` consecutive broker/API errors.
    3. A position/state mismatch: the broker-reported position disagrees
       with the internal ledger beyond `state_mismatch_tolerance` shares.
       This is "we can no longer trust our own book," which is a different
       failure mode from a market-risk breach, so it's tracked separately.
    4. A manual trip (trip_manual()), for an operator-initiated halt.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch


class HaltReason(Enum):
    NONE = "none"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    BROKER_ERROR_STREAK = "broker_error_streak"
    STATE_MISMATCH = "state_mismatch"
    MANUAL = "manual"


@dataclass
class KillSwitchStatus:
    halted: torch.Tensor       # [n_envs] bool
    reason: List[HaltReason]   # length n_envs -- first reason that tripped each env, kept until reset()


class KillSwitch:
    def __init__(
        self,
        n_envs: int,
        daily_loss_limit_frac: float = 0.03,
        broker_error_streak_limit: int = 3,
        state_mismatch_tolerance: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.n_envs = n_envs
        self.daily_loss_limit_frac = daily_loss_limit_frac
        self.broker_error_streak_limit = broker_error_streak_limit
        self.state_mismatch_tolerance = state_mismatch_tolerance
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._halted = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        self._reason: List[HaltReason] = [HaltReason.NONE] * self.n_envs
        self._day_start_equity = torch.zeros(self.n_envs, device=self.device)
        self._broker_error_streak = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

    # --- setup / lifecycle ---------------------------------------------

    def start_new_day(self, equity: torch.Tensor, env_mask: Optional[torch.Tensor] = None) -> None:
        """Call once at the start of each trading day/session to set the daily-loss reference point."""
        equity = equity.to(self.device)
        if env_mask is None:
            self._day_start_equity = equity.clone()
        else:
            env_mask = env_mask.to(self.device)
            self._day_start_equity[env_mask] = equity[env_mask]

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """Manually clears a halt. Not automatic -- a human/supervisor decides it's safe to resume."""
        if env_mask is None:
            self._halted.zero_()
            self._reason = [HaltReason.NONE] * self.n_envs
            self._broker_error_streak.zero_()
        else:
            env_mask = env_mask.to(self.device)
            for i in torch.nonzero(env_mask, as_tuple=True)[0].tolist():
                self._reason[i] = HaltReason.NONE
            self._halted[env_mask] = False
            self._broker_error_streak[env_mask] = 0

    # --- trip conditions -------------------------------------------------

    def record_broker_error(self, env_mask: torch.Tensor) -> None:
        """Call from live_loop.py whenever a broker/API call fails for the given envs (bool mask)."""
        env_mask = env_mask.to(self.device)
        self._broker_error_streak = torch.where(
            env_mask, self._broker_error_streak + 1, self._broker_error_streak
        )
        self._trip(env_mask & (self._broker_error_streak >= self.broker_error_streak_limit), HaltReason.BROKER_ERROR_STREAK)

    def record_broker_success(self, env_mask: torch.Tensor) -> None:
        """Call on a clean broker/API round-trip to reset that env's error streak."""
        env_mask = env_mask.to(self.device)
        self._broker_error_streak = torch.where(env_mask, torch.zeros_like(self._broker_error_streak), self._broker_error_streak)

    def check_daily_loss(self, equity: torch.Tensor) -> None:
        """Call once per step (backtest) or once per broker sync (live) with current equity."""
        equity = equity.to(self.device)
        loss_frac = (self._day_start_equity - equity) / self._day_start_equity.clamp(min=1e-6)
        self._trip(loss_frac >= self.daily_loss_limit_frac, HaltReason.DAILY_LOSS_LIMIT)

    def check_state_mismatch(self, internal_positions: torch.Tensor, broker_positions: torch.Tensor) -> None:
        """
        Compares the internal ledger (e.g. PortfolioState.positions[:, i] for
        ticker i) against what the broker actually reports. Anything beyond
        tolerance means the book can't be trusted right now, independent of
        any PnL condition.
        """
        internal_positions = internal_positions.to(self.device)
        broker_positions = broker_positions.to(self.device)
        mismatch = (internal_positions - broker_positions).abs() > self.state_mismatch_tolerance
        self._trip(mismatch, HaltReason.STATE_MISMATCH)

    def trip_manual(self, env_mask: torch.Tensor) -> None:
        """Explicit operator-initiated halt."""
        self._trip(env_mask.to(self.device), HaltReason.MANUAL)

    def _trip(self, mask: torch.Tensor, reason: HaltReason) -> None:
        if not mask.any():
            return
        self._halted = self._halted | mask
        for i in torch.nonzero(mask, as_tuple=True)[0].tolist():
            if self._reason[i] == HaltReason.NONE:
                self._reason[i] = reason

    # --- polling -----------------------------------------------------------

    def is_halted(self) -> torch.Tensor:
        """[n_envs] bool. Safe to call from anywhere, independent of any step loop."""
        return self._halted.clone()

    def status(self) -> KillSwitchStatus:
        return KillSwitchStatus(halted=self._halted.clone(), reason=list(self._reason))

    def apply(self, direction: torch.Tensor, size: torch.Tensor):
        """
        Convenience helper: zero direction/size for any currently-halted env.
        This freezes NEW orders; it does not itself submit a flattening
        order. Whether a trip should immediately flatten existing positions
        (vs. just freeze further trading) is a strategy decision left to the
        caller -- e.g. live_loop.py may want to submit one reduce-to-flat
        order the moment is_halted() flips True, then rely on this method
        every step after to keep it flat.
        """
        halted = self._halted
        direction = torch.where(halted, torch.zeros_like(direction), direction)
        size = torch.where(halted, torch.zeros_like(size), size)
        return direction, size