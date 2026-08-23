"""
training/ppo_hybrid.py

Vectorized rollout collection across all n_envs (== n_tickers) streams
simultaneously, GAE(lambda) advantage computation, and the PPO update step
with a mixed log-prob ratio (categorical direction log-prob + Beta size/
limit_offset log-prob, summed per action -- see hybrid_policy.py), a clipped
surrogate loss, and an entropy bonus applied to the discrete and continuous
heads with independent coefficients.

READ THIS BEFORE TOUCHING THE UPDATE LOOP:

Berserker's worst bug (see project notes) was `_full_forward` silently
passing zeroed risk features into fusion during PPO updates -- i.e. the
forward pass used to recompute log_probs/values at update time didn't
actually match the forward pass used to collect the rollout. That bug class
is the single most likely thing to reappear here, because this file
necessarily has TWO forward passes over the same trajectory (once live
during collect_rollout(), once replayed during update()), and they must
match exactly:
    - same obs (stored verbatim in the buffer, not recomputed)
    - same initial LSTM hidden state (buffer.initial_hidden, cloned fresh
      each PPO epoch -- never re-zeroed, never carried over from a
      different rollout)
    - same sequential order through time (the LSTM's hidden state is
      genuinely recurrent across the T env-steps in a rollout -- CNN can be
      batched across T since it's stateless, but the LSTM loop below MUST
      stay a sequential python loop over t; batching across t there would
      silently give every step the SAME (wrong) initial hidden state)
When you get to day 15's debugging: if the policy's loss goes to garbage
partway through an update but the rollout looked sane, check this file's
replay path against collect_rollout() line by line before looking anywhere
else.

Action pipeline (mirrors the modules built in earlier turns), applied once
per rollout step between the policy's normalized output and env.step():

    HybridActionSample (normalized, (0,1))
      -> HybridPolicyHead.rescale_size / rescale_limit_offset   (raw shares / ticks)
      -> KellySizer.apply        (soft, edge-based size cap)
      -> RiskManager.apply       (hard position/exposure/drawdown caps)
      -> KillSwitch.apply        (hard halt)
      -> env.step()

Limitation, noted rather than hidden: PPO epochs here are full-batch (no
minibatch shuffling) because minibatching would need to preserve time order
within each env stream for the LSTM replay to be valid, and this file
doesn't implement chunked/truncated-BPTT minibatching. Shuffling across the
env axis (not the time axis) would be safe to add if this becomes a memory
bottleneck; shuffling across time would not be.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.cnn_encoder import CNNEncoder
from model.lstm_encoder import LSTMEncoder, Hidden
from model.cross_attention import CrossAssetAttention
from model.fusion import FusionTrunk
from model.hybrid_policy import HybridPolicyHead, HybridActionSample, FLAT_IDX
from model.dual_critic import DualCriticHead

from risk.kelly_sizing import KellySizer
from risk.risk_manager import RiskManager
from risk.kill_switch import KillSwitch

from training.reward import DifferentialSharpeReward
from training.config import TrainingConfig

from env.vec_trading_env import VecTradingEnv, StepResult

Tensor = torch.Tensor
TickCallback = Callable[[int, "StepResult", Tensor, Tensor, "KillSwitch"], None]


# --------------------------------------------------------------------------
# Combined actor-critic network: cnn -> lstm -> cross-attention -> fusion,
# with the policy and critic heads hanging off the shared trunk.
# --------------------------------------------------------------------------

class HybridActorCritic(nn.Module):
    def __init__(self, n_features: int, cfg: TrainingConfig) -> None:
        super().__init__()
        m = cfg.model

        self.cnn = CNNEncoder(
            n_features=n_features,
            channels=m.cnn_channels,
            kernel_sizes=m.cnn_kernel_sizes,
            dilations=m.cnn_dilations,
            dropout=m.cnn_dropout,
        )
        self.lstm = LSTMEncoder(
            input_dim=self.cnn.output_dim,
            hidden_dim=m.lstm_hidden_dim,
            num_layers=m.lstm_num_layers,
            dropout=m.lstm_dropout,
        )
        self.cross_attn = CrossAssetAttention(
            embed_dim=self.lstm.output_dim,
            num_heads=m.attn_num_heads,
            ffn_dim=m.attn_ffn_dim,
            dropout=m.attn_dropout,
        )
        self.fusion = FusionTrunk(
            cnn_dim=self.cnn.output_dim,
            lstm_dim=self.lstm.output_dim,
            trunk_dim=m.trunk_dim,
            dropout=m.fusion_dropout,
        )
        self.policy_head = HybridPolicyHead(
            trunk_dim=self.fusion.output_dim,
            direction_embed_dim=m.policy_direction_embed_dim,
            hidden_dim=m.policy_hidden_dim,
            min_concentration=m.policy_min_concentration,
            dropout=m.policy_dropout,
        )
        self.critic_head = DualCriticHead(
            trunk_dim=self.fusion.output_dim,
            hidden_dim=m.critic_hidden_dim,
            dropout=m.critic_dropout,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Hidden:
        return self.lstm.init_hidden(batch_size, device)

    def forward_features(self, obs: Tensor, hidden: Hidden) -> Tuple[Tensor, Hidden]:
        """
        obs: [batch, window, n_features] for this single rollout step
             (batch == n_envs == n_tickers, per the project's convention).
        hidden: (h, c) carried in from the previous rollout step.

        Returns (trunk, new_hidden). `trunk` feeds both policy_head and
        critic_head; new_hidden must be threaded to the NEXT call.
        """
        cnn_seq = self.cnn(obs)                       # [batch, window, cnn_dim]
        cnn_last = cnn_seq[:, -1, :]                   # [batch, cnn_dim]
        lstm_seq, new_hidden = self.lstm(cnn_seq, hidden)  # [batch, window, lstm_dim]
        lstm_last = lstm_seq[:, -1, :]                 # [batch, lstm_dim]
        attn_out = self.cross_attn(lstm_last)          # [batch, lstm_dim] -- batch doubles as the ticker axis
        trunk = self.fusion(cnn_last, attn_out)         # [batch, trunk_dim]
        return trunk, new_hidden


# --------------------------------------------------------------------------
# Rollout buffer
# --------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    obs: Tensor              # [T, n_envs, window, n_features]
    direction_idx: Tensor    # [T, n_envs] long
    size: Tensor             # [T, n_envs] -- normalized (0,1), as sampled by act()
    limit_offset: Tensor     # [T, n_envs] -- normalized (0,1), as sampled by act()
    log_prob_old: Tensor     # [T, n_envs]
    value_old: Tensor        # [T, n_envs] -- dual-critic-selected value at rollout time
    position_before: Tensor  # [T, n_envs] -- sign of position entering each step, for value-head selection
    filled_qty: Tensor       # [T, n_envs] -- signed shares actually filled that step (0 = no trade), for
                               # trade-count/turnover metrics -- see vec_trading_env.py's info["filled_qty"]
    reward: Tensor           # [T, n_envs] -- shaped reward actually used for GAE
    done: Tensor              # [T, n_envs] bool
    initial_hidden: Hidden   # (h0, c0) at the START of this rollout -- replayed from fresh each PPO epoch
    advantages: Optional[Tensor] = None  # [T, n_envs], filled in by compute_gae()
    returns: Optional[Tensor] = None     # [T, n_envs], filled in by compute_gae()


# --------------------------------------------------------------------------
# Rollout collection
# --------------------------------------------------------------------------

def _run_action_pipeline(
    actor_critic: HybridActorCritic,
    action_sample: HybridActionSample,
    kelly_sizer: KellySizer,
    risk_manager: RiskManager,
    kill_switch: KillSwitch,
    env: VecTradingEnv,
    mid_price: Tensor,
    equity_before: Tensor,
    current_position_notional: Tensor,
    cfg: TrainingConfig,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Runs the fixed post-policy action pipeline for one rollout step:
    normalized policy output -> raw shares/ticks -> Kelly soft cap ->
    RiskManager hard caps -> KillSwitch hard halt.

    Returns (final_direction, final_size, final_limit_offset), exactly the
    triple env.step() expects. Must be called under the same no_grad scope
    as the rest of collect_rollout() -- see that function's docstring.
    """
    # Equity-relative ceiling, not a flat share count -- see
    # HybridPolicyHead.size_cap_shares for why the old absolute 10,000-share
    # scale left 99.95% of the size head's range dead.
    size_shares_uncapped = HybridPolicyHead.rescale_size(
        action_sample.size,
        HybridPolicyHead.size_cap_shares(
            equity_before, mid_price,
            cfg.risk.max_order_notional_frac, cfg.risk.max_order_shares,
        ),
    )
    limit_offset_ticks = HybridPolicyHead.rescale_limit_offset(
        action_sample.limit_offset, cfg.risk.max_limit_offset_ticks
    )

    kelly_result = kelly_sizer.apply(
        size=size_shares_uncapped,
        direction=action_sample.direction,
        mid_price=mid_price,
        equity=equity_before,
        current_position_notional=current_position_notional,
    )
    risk_result = risk_manager.apply(
        direction=action_sample.direction,
        size=kelly_result.size,
        limit_offset=limit_offset_ticks,
        mid_price=mid_price,
        portfolio=env.portfolio,
        ticker_idx=0,
    )
    final_direction, final_size = kill_switch.apply(risk_result.direction, risk_result.size)

    final_direction, final_size = _apply_flat_intent(
        raw_direction=action_sample.direction,
        final_direction=final_direction,
        final_size=final_size,
        position=env.portfolio.positions[:, 0],
        halted=kill_switch.is_halted(),
    )
    return final_direction, final_size, risk_result.limit_offset


def _apply_flat_intent(
    raw_direction: Tensor,
    final_direction: Tensor,
    final_size: Tensor,
    position: Tensor,
    halted: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Turns the policy's own "flat" action into an actual close.

    vec_trading_env.step() treats direction == 0 as "no order this step",
    which is right for a stream that is already flat but means a stream
    HOLDING a position can never deliberately exit: the only other route to
    flat is a continuous size draw landing exactly on the current position
    magnitude, which is effectively measure-zero. Two consecutive 50-rollout
    runs produced 6,043 and 2,380 flips against exactly **0 opens and 0
    closes**, with 98%+ long occupancy -- the policy was structurally unable
    to say "get me out", so it expressed every view as a reversal.

    Feeding position state into the observation (see
    VecTradingEnv._augment_obs_with_portfolio_state) was necessary but not
    sufficient: the second run showed the actor could finally SEE its
    inventory and still had no action that meant "sell it", and opens/closes
    stayed pinned at 0. This supplies the missing mechanism.

    WHY HERE, and not in vec_trading_env.step(): by the time an action
    reaches the env, a zero direction has three indistinguishable causes,
    and only the first is the policy's intent --
      1. the discrete head sampled "flat";
      2. risk_manager.py zeroed it (`direction = where(size == 0, 0,
         direction)`) after a cap or the min_order_notional dust gate
         clipped size to 0;
      3. kill_switch.py zeroed it because the stream is halted.
    An earlier attempt at this lived in the env and therefore treated all
    three alike, turning every dust rejection and every halt into a forced
    liquidation. Here `action_sample.direction` is still the raw policy
    output, so intent is unambiguous.

    Closes deliberately bypass the Kelly/Risk caps computed above, because
    every one of those caps exists to bound EXPOSURE and a full close takes
    exposure to zero: risk_manager.py's dust gate and drawdown halt already
    exempt reducing orders by design, and kelly_sizing.py's notional cap
    documents that "a pure reduce passes through untouched". Liquidity is
    still enforced downstream -- execution_sim.py's max_participation can
    partially fill a close on a thin bar, and a zero-volume bar fills none
    of it, which is correct.

    KillSwitch is the one thing NOT bypassed. Its apply() docstring states
    that it "freezes NEW orders; it does not itself submit a flattening
    order", leaving flatten-vs-freeze to the caller -- so a halted stream
    stays frozen rather than being force-flattened here.
    """
    wants_flat = raw_direction.to(position.device) == 0
    close_mask = wants_flat & (position != 0) & ~halted.to(position.device)
    direction = torch.where(close_mask, -torch.sign(position), final_direction.to(position.dtype))
    size = torch.where(close_mask, position.abs(), final_size.to(position.dtype))
    return direction, size


def collect_rollout(
    env: VecTradingEnv,
    actor_critic: HybridActorCritic,
    kelly_sizer: KellySizer,
    risk_manager: RiskManager,
    kill_switch: KillSwitch,
    reward_shaper: DifferentialSharpeReward,
    obs: Tensor,
    hidden: Hidden,
    cfg: TrainingConfig,
    tick_callback: Optional[TickCallback] = None,
) -> Tuple[RolloutBuffer, Tensor, Tensor, Hidden]:
    """
    Runs cfg.ppo.rollout_length steps across all streams simultaneously.

    tick_callback, if given, is called once per env-step (i.e. cfg.ppo.
    rollout_length times per call to collect_rollout()) as
    tick_callback(local_t, step_result, final_direction, final_size,
    kill_switch) -- AFTER env.step() for that tick, so step_result.info
    already reflects the fill/equity/position for that exact tick. This
    exists purely so a caller (train.py) can log tick-level metrics without
    collect_rollout() itself knowing anything about metrics/dashboards --
    it never imports MetricsWriter, matching this project's existing
    decoupling convention (see monitoring/dashboard.py's module docstring).
    Left as None, behavior is byte-identical to before this parameter
    existed.

    `obs` and `hidden` are the carry-over from the previous rollout (or from
    env.reset() / actor_critic.init_hidden() on the very first call) --
    collect_rollout() does NOT reset the environment itself, since whether
    to reset is a session-boundary decision made by the training loop, not
    by the rollout collector.

    Returns (buffer, final_obs, final_position_sign, final_hidden) --
    the last three are what the NEXT call to collect_rollout() should be
    given, and final_position_sign / final_obs / final_hidden are also used
    right here to compute the GAE bootstrap value.
    """
    T = cfg.ppo.rollout_length

    obs_buf: List[Tensor] = []
    direction_idx_buf: List[Tensor] = []
    size_buf: List[Tensor] = []
    limit_offset_buf: List[Tensor] = []
    log_prob_buf: List[Tensor] = []
    value_buf: List[Tensor] = []
    position_before_buf: List[Tensor] = []
    filled_qty_buf: List[Tensor] = []
    reward_buf: List[Tensor] = []
    done_buf: List[Tensor] = []

    initial_hidden = (hidden[0].clone().detach(), hidden[1].clone().detach())

    # Rollout collection never needs gradients -- only ppo_update()'s replay
    # does. Without this, every act()/forward_features() call below builds a
    # live autograd graph across all T steps, and initial_hidden (captured
    # just above) stays wired into it despite the .clone(). ppo_update()'s
    # first PPO epoch would then call .backward() through that graph and
    # free it; the second epoch, replaying from the same initial_hidden,
    # would hit "Trying to backward through the graph a second time" --
    # confirmed by reproducing it before adding this no_grad wrapper. This
    # is a distinct bug from Berserker's forward-pass-mismatch bug the
    # module docstring warns about, but the same family: collect_rollout's
    # graph must be fully isolated from ppo_update()'s replay graph.
    with torch.no_grad():
        for local_t in range(T):
            mid_price = env._current_prices()  # noqa: SLF001 -- same accessor env.step() itself uses internally
            equity_before = env.portfolio.equity(mid_price.unsqueeze(1))
            position_before = torch.sign(env.portfolio.positions[:, 0])
            current_position_notional = env.portfolio.positions[:, 0] * mid_price

            trunk, new_hidden = actor_critic.forward_features(obs, hidden)
            action_sample: HybridActionSample = actor_critic.policy_head.act(trunk, deterministic=False)
            values = actor_critic.critic_head(trunk)
            value_selected = DualCriticHead.select(values, position_before)

            final_direction, final_size, final_limit_offset = _run_action_pipeline(
                actor_critic=actor_critic,
                action_sample=action_sample,
                kelly_sizer=kelly_sizer,
                risk_manager=risk_manager,
                kill_switch=kill_switch,
                env=env,
                mid_price=mid_price,
                equity_before=equity_before,
                current_position_notional=current_position_notional,
                cfg=cfg,
            )

            step_result: StepResult = env.step(
                direction=final_direction, size=final_size, limit_offset=final_limit_offset
            )

            kelly_sizer.record_realized_pnl(
                step_result.info["realized_delta"],
                closed_mask=step_result.info.get("closed_trade"),
            )
            kill_switch.check_daily_loss(step_result.info["equity"])
            # Re-anchor the daily-loss reference on a REAL session boundary.
            #
            # KillSwitch.daily_loss_limit_frac reads as a 3% daily stop, and in
            # live/live_loop.py it is one. In training it was not: the trainers
            # call start_new_day() every kill_switch_reset_every_n_rollouts,
            # which is 1 rollout = 256 bars = 3.3 trading days, re-anchored at
            # a point that lands mid-session and drifts. So the limit measured
            # 3% over ~3.3 days from an arbitrary origin, and its name said
            # something else. The env now knows where sessions begin, so the
            # reference can be what it claims to be.
            #
            # The periodic kill_switch.reset() in the trainer loops is a
            # separate mechanism with a separate purpose (unsticking streams
            # halted early in training) and is deliberately left alone.
            if getattr(env, "session_just_started", False):
                kill_switch.start_new_day(step_result.info["equity"])

            step_return = step_result.info["step_pnl"] / equity_before.clamp(min=1e-6)
            # REWARD-EXPLOSION FIX: clamp per-step return to [-100%, +100%].
            # Without this, a stream whose equity has decayed toward ~0
            # produces step_pnl / 1e-6 ~ 1e8+ "returns". The DSR's own D_t is
            # clamped, but the checkpoint bonus in training/reward.py scales
            # with the number of milestone crossings, which is UNBOUNDED --
            # a single bankrupt stream can inject multi-thousand rewards into
            # the rollout mean, corrupting the EMA / best-checkpoint tracking
            # (observed: EMA reward jumped from -0.0003 to ~830 in one
            # rollout). Real 5-min equity moves never approach 100% in a bar,
            # so this clamp is free in practice.
            step_return = step_return.clamp(-1.0, 1.0)
            dsr_reward = reward_shaper.step(step_return)
            shaped_reward = cfg.reward.sharpe_weight * dsr_reward + cfg.reward.raw_weight * step_result.reward

            obs_buf.append(obs)
            direction_idx_buf.append(action_sample.direction_idx)
            size_buf.append(action_sample.size)
            limit_offset_buf.append(action_sample.limit_offset)
            log_prob_buf.append(action_sample.log_prob)
            value_buf.append(value_selected)
            position_before_buf.append(position_before)
            filled_qty_buf.append(step_result.info["filled_qty"])
            reward_buf.append(shaped_reward)
            done_buf.append(step_result.done)

            if tick_callback is not None:
                # Fires once per real env-step (a "tick"), not once per
                # rollout -- see this function's docstring. step_result.info
                # already reflects this exact tick's fill/equity/position;
                # final_direction/final_size are the fully-risk-processed
                # action that was actually submitted, not the raw policy
                # output.
                tick_callback(local_t, step_result, final_direction, final_size, kill_switch)

            # NOTE: vec_trading_env.py currently signals `done` globally (the
            # whole dataset pass ends at once, not per-stream) -- this handles
            # the general case anyway so it keeps working if that ever changes.
            if step_result.done.any():
                done_mask = step_result.done
                new_hidden = actor_critic.lstm.reset_hidden(new_hidden, done_mask)
                kelly_sizer.reset(done_mask)
                reward_shaper.reset(done_mask)
                # kill_switch is deliberately NOT reset here -- a halt persists
                # across episode boundaries until someone calls reset()
                # explicitly (see kill_switch.py's module docstring).

            obs = step_result.obs
            hidden = new_hidden

    buffer = RolloutBuffer(
        obs=torch.stack(obs_buf, dim=0),
        direction_idx=torch.stack(direction_idx_buf, dim=0),
        size=torch.stack(size_buf, dim=0),
        limit_offset=torch.stack(limit_offset_buf, dim=0),
        log_prob_old=torch.stack(log_prob_buf, dim=0),
        value_old=torch.stack(value_buf, dim=0),
        position_before=torch.stack(position_before_buf, dim=0),
        filled_qty=torch.stack(filled_qty_buf, dim=0),
        reward=torch.stack(reward_buf, dim=0),
        done=torch.stack(done_buf, dim=0),
        initial_hidden=initial_hidden,
    )

    # --- bootstrap value for GAE: one more forward pass on the final obs,
    # using the position the portfolio holds AFTER the last step in this
    # rollout (i.e. the position it will be "entering" the next rollout with).
    with torch.no_grad():
        final_trunk, final_hidden = actor_critic.forward_features(obs, hidden)
        final_values = actor_critic.critic_head(final_trunk)
        final_position = torch.sign(env.portfolio.positions[:, 0])
        final_value = DualCriticHead.select(final_values, final_position)

    return buffer, obs, final_value, hidden


# --------------------------------------------------------------------------
# GAE(lambda)
# --------------------------------------------------------------------------

def compute_gae(buffer: RolloutBuffer, final_value: Tensor, gamma: float, gae_lambda: float) -> None:
    """
    Fills in buffer.advantages and buffer.returns in place. final_value is
    the bootstrap V(s_{T}) returned by collect_rollout() -- the value of the
    state one step past the end of this buffer, using whichever critic head
    matches the position the portfolio holds at that point.
    """
    T, n_envs = buffer.reward.shape
    device = buffer.reward.device

    values_ext = torch.cat([buffer.value_old, final_value.unsqueeze(0)], dim=0)  # [T+1, n_envs]
    advantages = torch.zeros(T, n_envs, device=device)
    last_gae = torch.zeros(n_envs, device=device)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - buffer.done[t].float()
        delta = buffer.reward[t] + gamma * values_ext[t + 1] * next_non_terminal - values_ext[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    buffer.advantages = advantages
    buffer.returns = advantages + buffer.value_old


# --------------------------------------------------------------------------
# PPO update
# --------------------------------------------------------------------------

def _replay_trunk_sequence(
    actor_critic: HybridActorCritic,
    flat_obs: Tensor,
    T: int,
    n_envs: int,
    window: int,
    hidden: Hidden,
    use_amp: bool,
    use_checkpoint: bool,
) -> Tensor:
    """
    Replays the CNN -> (sequential) LSTM -> cross-attention -> fusion trunk
    over a stored trajectory, under the given autocast setting.

    CRITICAL: the LSTM loop over `t` MUST stay a sequential Python loop --
    the hidden state is genuinely recurrent across steps within a rollout.
    Batching across t would silently feed every step the same (wrong)
    initial hidden state. See this module's docstring before changing
    anything here.

    Returns flat_trunk of shape [T * n_envs, trunk_dim], still under
    autocast dtype if use_amp was True (the caller casts back to float()
    immediately after, once outside this function's `with autocast` scope).
    """
    with torch.cuda.amp.autocast(enabled=use_amp):
        if use_checkpoint and torch.is_grad_enabled():
            # CNN is stateless/side-effect-free (see cnn_encoder.py), so
            # checkpointing its forward is safe: recomputed during
            # backward instead of keeping every conv block's activations
            # around for all T*n_envs rows simultaneously -- this is the
            # single biggest activation-memory line item as n_envs grows
            # (see training/config.py's ModelConfig docstring).
            cnn_seq_all = torch.utils.checkpoint.checkpoint(
                actor_critic.cnn, flat_obs, use_reentrant=False
            )
        else:
            cnn_seq_all = actor_critic.cnn(flat_obs)
        cnn_seq_all = cnn_seq_all.reshape(T, n_envs, window, -1)

        trunks: List[Tensor] = []
        for t in range(T):  # MUST stay sequential -- see module docstring
            lstm_seq_t, hidden = actor_critic.lstm(cnn_seq_all[t], hidden)
            lstm_last_t = lstm_seq_t[:, -1, :]
            cnn_last_t = cnn_seq_all[t][:, -1, :]
            attn_out_t = actor_critic.cross_attn(lstm_last_t)
            trunks.append(actor_critic.fusion(cnn_last_t, attn_out_t))
        trunk_all = torch.stack(trunks, dim=0)  # [T, n_envs, trunk_dim]
        return trunk_all.reshape(T * n_envs, -1)


class AdaptiveEntropyCoef:
    """Dual-ascent controller holding discrete-head entropy near a target.

    WHY A CONTROLLER AND NOT A BIGGER CONSTANT. A fixed entropy bonus enters
    the loss as -coef * H, so its gradient pressure is constant while the
    advantage pressure toward "never trade" is not. Against a real cost floor
    that advantage is relentless and one-signed, so any fixed coef is either
    eventually overwhelmed or so large the policy is just noise. This project
    has already tried the constant: 0.02 -> 0.05 is recorded in PPOConfig as a
    fix for exactly this collapse, and the last run still went to
    entropy_discrete = 0.000 at rollout 81 and stayed there for 65 more
    rollouts -- 43% of the compute spent on a policy that could no longer
    explore, and it happened having seen 22% of the training split.

    The controller instead adapts in log space:

        log_coef <- log_coef + lr * (target - measured)

    so a persistent shortfall grows the coefficient without bound (up to
    max_coef) until entropy comes back. Falling below target is self-
    correcting rather than terminal.

    ON THE TARGET. For the 3-way SHORT/FLAT/LONG head, max entropy is
    ln(3) = 1.0986 nats. A target of 0.5 still permits ~85% of the mass on one
    action -- (0.8, 0.1, 0.1) measures 0.639 nats -- so this floors exploration
    without forcing churn. It does NOT prevent the policy from preferring FLAT;
    it prevents it from becoming incapable of considering anything else.

    DISCRETE HEAD ONLY, deliberately. The continuous size head also collapsed
    last run (entropy_continuous -> 0.000), but every observed fill was at
    exactly the Kelly cap -- 0.0799 x median equity, with kelly_fractional
    pinned to its floor in 98.4% of cells -- so that head's output had no
    effect on executed size. Forcing entropy into a channel the risk pipeline
    overrides would burn exploration budget for nothing. Fix the Kelly binding
    first, then revisit.

    CHECKPOINTED. KellySizer's un-checkpointed state is what made the Session 1
    lock re-arm on every restart; a controller that silently resets to its
    initial coefficient on --resume would repeat that class of bug exactly.
    """

    def __init__(self, target, init_coef, lr=0.1, min_coef=0.005, max_coef=2.0):
        self.target = float(target)
        self.lr = float(lr)
        self.min_coef = float(min_coef)
        self.max_coef = float(max_coef)
        self._log_coef = math.log(max(float(init_coef), 1e-8))

    def coef(self):
        return float(min(max(math.exp(self._log_coef), self.min_coef), self.max_coef))

    def observe(self, measured_entropy):
        """One dual-ascent step. Call once per PPO epoch, after the backward."""
        m = float(measured_entropy)
        if not math.isfinite(m):
            return
        self._log_coef += self.lr * (self.target - m)
        # Clamp in log space too, or the coefficient can wind up far outside
        # the usable band while coef() silently saturates and the controller
        # then takes many epochs to respond to a genuine entropy recovery.
        self._log_coef = min(max(self._log_coef, math.log(self.min_coef)),
                             math.log(self.max_coef))

    def state_dict(self):
        return {"log_coef": self._log_coef, "target": self.target, "lr": self.lr,
                "min_coef": self.min_coef, "max_coef": self.max_coef}

    def load_state_dict(self, d):
        if not d:
            return
        self._log_coef = float(d.get("log_coef", self._log_coef))
        self.target = float(d.get("target", self.target))
        self.lr = float(d.get("lr", self.lr))
        self.min_coef = float(d.get("min_coef", self.min_coef))
        self.max_coef = float(d.get("max_coef", self.max_coef))



def ppo_update(
    actor_critic: HybridActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: TrainingConfig,
    scaler: "Optional[torch.cuda.amp.GradScaler]" = None,
    entropy_ctl=None,
) -> Dict[str, float]:
    """
    Replays the stored trajectory through the CURRENT network parameters
    (CNN batched across time, LSTM sequential -- see module docstring
    warning about matching collect_rollout()'s forward pass exactly),
    recomputes the mixed log-prob and split entropy, and runs the clipped
    PPO objective for cfg.ppo.ppo_epochs full-batch epochs.

    cfg.run.use_amp / cfg.model.use_gradient_checkpointing (both off by
    default -- see training/config.py's fields for the full rationale):
    when use_amp is True, `scaler` MUST be a torch.cuda.amp.GradScaler the
    caller creates ONCE outside the rollout loop and passes into every
    ppo_update() call (a fresh scaler each call would reset its loss-scale
    adaptation every rollout, defeating the point). autocast covers ONLY
    the CNN/LSTM/cross-attention/fusion trunk below -- policy_head/
    critic_head and all loss math run in fp32 regardless of use_amp, since
    model/hybrid_policy.py's Beta log-prob math is explicitly flagged in
    that file's own docstring as numerically fragile near the (0,1)
    boundary. Gradient checkpointing (when enabled) wraps ONLY the CNN's
    T*n_envs mega-batch forward, not the sequential LSTM loop -- see
    training/config.py's ModelConfig.use_gradient_checkpointing docstring
    for why the LSTM loop is deliberately excluded.
    """
    T, n_envs, window, n_features = buffer.obs.shape
    p = cfg.ppo
    use_amp = bool(getattr(cfg.run, "use_amp", False)) and torch.cuda.is_available()
    use_checkpoint = bool(getattr(cfg.model, "use_gradient_checkpointing", False))

    flat_direction_idx = buffer.direction_idx.reshape(T * n_envs)
    flat_size = buffer.size.reshape(T * n_envs)
    flat_limit_offset = buffer.limit_offset.reshape(T * n_envs)
    flat_log_prob_old = buffer.log_prob_old.reshape(T * n_envs)
    flat_value_old = buffer.value_old.reshape(T * n_envs)
    flat_position_before = buffer.position_before.reshape(T * n_envs)
    flat_advantages = buffer.advantages.reshape(T * n_envs)
    flat_returns = buffer.returns.reshape(T * n_envs)
    flat_obs = buffer.obs.reshape(T * n_envs, window, n_features)

    adv_mean, adv_std = flat_advantages.mean(), flat_advantages.std()
    flat_advantages_norm = (flat_advantages - adv_mean) / adv_std.clamp(min=1e-8)

    last_stats: Dict[str, float] = {}

    for _ in range(p.ppo_epochs):
        # --- replay the trajectory from the SAME initial hidden every epoch
        # (a frozen reference point recorded at rollout time -- never
        # re-zeroed, never taken from a later/different rollout). See the
        # module docstring's warning about Berserker's forward-pass mismatch
        # bug before changing anything below.
        hidden = (buffer.initial_hidden[0].clone(), buffer.initial_hidden[1].clone())

        flat_trunk = _replay_trunk_sequence(
            actor_critic=actor_critic,
            flat_obs=flat_obs,
            T=T,
            n_envs=n_envs,
            window=window,
            hidden=hidden,
            use_amp=use_amp,
            use_checkpoint=use_checkpoint,
        )

        # --- deliberately outside autocast here: everything below is
        # policy_head/critic_head distribution math and the PPO loss
        # itself, kept fp32 regardless of use_amp (see this function's
        # docstring).
        flat_trunk = flat_trunk.float()

        log_prob_new, discrete_entropy, continuous_entropy = actor_critic.policy_head.evaluate_actions_split(
            flat_trunk, flat_direction_idx, flat_size, flat_limit_offset
        )
        values_new = actor_critic.critic_head(flat_trunk)
        value_new_selected = DualCriticHead.select(values_new, flat_position_before)

        ratio = torch.exp(log_prob_new - flat_log_prob_old)
        surr1 = ratio * flat_advantages_norm
        surr2 = torch.clamp(ratio, 1.0 - p.clip_range, 1.0 + p.clip_range) * flat_advantages_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        value_clipped = flat_value_old + torch.clamp(
            value_new_selected - flat_value_old, -p.value_clip_range, p.value_clip_range
        )
        value_loss_unclipped = (value_new_selected - flat_returns).pow(2)
        value_loss_clipped = (value_clipped - flat_returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # entropy_coef_discrete is the INITIAL value when a controller is
        # supplied; AdaptiveEntropyCoef then owns it. Passing None restores
        # the old fixed-coefficient behaviour exactly.
        coef_discrete = (entropy_ctl.coef() if entropy_ctl is not None
                         else p.entropy_coef_discrete)
        entropy_bonus = (
            coef_discrete * discrete_entropy + p.entropy_coef_continuous * continuous_entropy
        ).mean()

        loss = policy_loss + p.value_loss_coef * value_loss - entropy_bonus

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # must unscale before clip_grad_norm_, or the clip threshold is meaningless
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), p.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), p.max_grad_norm)
            optimizer.step()

        if entropy_ctl is not None:
            entropy_ctl.observe(discrete_entropy.mean().item())

        with torch.no_grad():
            approx_kl = (flat_log_prob_old - log_prob_new).mean().item()
            clip_frac = ((ratio - 1.0).abs() > p.clip_range).float().mean().item()

        last_stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_discrete": discrete_entropy.mean().item(),
            "entropy_continuous": continuous_entropy.mean().item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "grad_norm": float(grad_norm),
            "entropy_coef_discrete": float(coef_discrete),
        }

    return last_stats