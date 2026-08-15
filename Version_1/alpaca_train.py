"""
alpaca_train.py — scalar (single-env) PPO training loop.

Aligned with Vectorised_training.py:
  - Uses select_vectorized_action / store_vectorized_transition API
  - Captures raw state + LSTM h/c snapshots for correct TBPTT replay
  - Passes adjusted_action from env info for correct credit assignment
  - Bootstraps terminal value via get_vectorized_values
  - Calls agent.update(next_values=[v]) with proper signature
  - Dir-mean sourced from agent._last_dir_probs (not stale scalar attr)
  - Saves to best_model_vec.pkl so alpaca_test.py can load directly
  - Log output redirected via log_redirect (same as vectorized runner)
"""

import numpy as np
import random
import os
import sys
import csv
import time

from engine import Tensor
from Neural_Nets import (LSTM, Conv2D, Flatten, Attention, MultiHeadAttention,
                         FusionLayers, RegimeDetector)
from nlp import NLPEncoder
from environment import TradingEnvironment, low_friction, realistic_friction, high_friction
from agent import PPOAgent
from models_utils import save_model, load_model, save_log
from log_redirect import redirect_prints, restore_prints, reset_episode_log

from telemetry import Telemetry
from alpaca_data import load_data, transform_data, build_windows

import environment as _env_check
print(f"[IMPORT CHECK] env loaded from: {_env_check.__file__}")
print(f"[IMPORT CHECK] TradingEnvironment reset method: {_env_check.TradingEnvironment.reset}")

# ── TBPTT chunk size — must match agent.py CHUNK_SIZE ─────────────────────────
TBPTT_CHUNK = 32

# ── Terminal colours ──────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
ORANGE = "\033[33m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

currency_units = "$"
_ANN_FACTOR    = np.sqrt(252 * 78)


# ── Display helpers ───────────────────────────────────────────────────────────

def print_step(episode, ticker, step, total_steps, net_worth, position, price, dir_mean):
    pos_pct  = (abs(position) / (net_worth + 1e-8)) * 100.0
    pos_sign = "L" if position > 0 else ("S" if position < 0 else "-")
    msg = (
        f"  [{ticker}] Ep{episode} | Step {step}/{total_steps} "
        f"| NetWorth: {GREEN if net_worth >= INITIAL_BALANCE else RED}"
        f"{currency_units}{net_worth:8.2f}{RESET}"
        f" | Pos: {GREEN if position >= 0 else RED}{pos_sign}{pos_pct:5.1f}%{RESET}"
        f" | Price: {currency_units}{price:.2f}"
        f" | DirMean: {dir_mean:+.3f}"
    )
    sys.stdout.write('\r' + msg + ' ' * 10)
    sys.stdout.flush()


def print_episode(episode, ticker, net_worth, reward, trades, win_rate, std, entropy,
                  best, position, avg_dir_mean, bankrupt=False, mirrored=False):
    sys.stdout.write('\n')
    star         = '★' if net_worth >= best else ' '
    status       = f" {ORANGE}[BANKRUPT]{RESET}" if bankrupt else ""
    mirror_label = " (MIRRORED)" if mirrored else ""
    print(
        f"{star} Ep {episode:3d} | {ticker:6s}{mirror_label} | "
        f"{GREEN if net_worth >= INITIAL_BALANCE else RED}"
        f"{currency_units}{net_worth:9.2f}{RESET}{status} "
        f"| Pos: {GREEN if position >= 0 else RED}{currency_units}{position:8.2f}{RESET} "
        f"| Reward: {reward:10.2f} "
        f"| Trades: {trades:4d} | WR: {win_rate:.1%} "
        f"| AvgDir: {avg_dir_mean:+.3f} | Std: {std:.3f} | Ent: {entropy:.3f}",
        flush=True,
    )


def print_reward_breakdown(breakdown: dict, episode: int):
    parts = " | ".join(
        f"{k}: {v:+.3f}" for k, v in breakdown.items() if k != 'total'
    )
    print(f"  [REWARD Ep{episode}] {parts} | total: {breakdown['total']:+.3f}",
          flush=True)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_max_drawdown(net_worths: list) -> float:
    peak   = net_worths[0]
    max_dd = 0.0
    for nw in net_worths:
        if nw > peak:
            peak = nw
        dd = (peak - nw) / (peak + 1e-8)
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe(net_worths: list) -> float:
    if len(net_worths) < 2:
        return 0.0
    nw      = np.array(net_worths, dtype=np.float64)
    returns = np.diff(nw) / (nw[:-1] + 1e-8)
    mu      = float(np.mean(returns))
    sigma   = float(np.std(returns))
    return round(mu / sigma * _ANN_FACTOR, 4) if sigma > 1e-10 else 0.0


def compute_sortino(net_worths: list) -> float:
    if len(net_worths) < 2:
        return 0.0
    nw          = np.array(net_worths, dtype=np.float64)
    returns     = np.diff(nw) / (nw[:-1] + 1e-8)
    mu          = float(np.mean(returns))
    neg_returns = returns[returns < 0]
    if len(neg_returns) == 0:
        return round(mu * _ANN_FACTOR, 4) if mu > 0 else 0.0
    downside = float(np.std(neg_returns))
    return round(mu / downside * _ANN_FACTOR, 4) if downside > 1e-10 else 0.0


def load_best_net_worth(log_path: str, initial_balance: float) -> float:
    if not os.path.exists(log_path):
        return initial_balance
    best = initial_balance
    with open(log_path, 'r') as f:
        for row in csv.DictReader(f):
            nw = float(row['final_balance'])
            if nw > best:
                best = nw
    return best


# ── Config ────────────────────────────────────────────────────────────────────

BASE_PATH = '/kaggle/working' if os.path.exists('/kaggle') else '.'
os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/logs",   exist_ok=True)

# Shared paths with alpaca_test.py
BEST_MODEL_PATH = f"{BASE_PATH}/models/best_model_vec.pkl"
CHECKPOINT_PATH = f"{BASE_PATH}/models/checkpoint_vec.pkl"
LOG_PATH        = f"{BASE_PATH}/logs/training_log.csv"

# TICKERS = ["AAPL" , "ARKK" , "GLD" , "IWM" , "KRE" , "NFLX" , "NVDA", "PYPL" , "QQQ" , "SPY" , "UNG" , "USO" , "XBI" , "XLE"]
TICKERS = ["SPY"]
START_DATE   = None
END_DATE     = "2024-01-01"
WINDOW_SIZE  = 48

EPISODES      = 50
SAVE_EVERY    = 2
TERMINAL_PRINTER = 25

FRICTION_MODE = "realistic"
_FRICTION_MAP = {
    "low":       low_friction,
    "realistic": realistic_friction,
    "high":      high_friction,
}
if FRICTION_MODE not in _FRICTION_MAP:
    raise ValueError(f"Unknown FRICTION_MODE {FRICTION_MODE!r}")
_friction_config = _FRICTION_MAP[FRICTION_MODE]()

INITIAL_BALANCE = 10_000
EPISODE_STEPS   = 750   # max steps per episode (hard cap like vectorized)

RESET_CRITIC = False
RESET_ACTOR  = False

cnn_out_height   = WINDOW_SIZE - 3 + 1
cnn_out_width    = 5 - 5 + 1
CNN_FLAT_SIZE    = 16 * cnn_out_height * cnn_out_width
FUSED_STATE_SIZE = 75

# ── Telemetry ─────────────────────────────────────────────────────────────────

telemetry = Telemetry(max_history=20)
telemetry.initial_balance = INITIAL_BALANCE

# ── Data loading ──────────────────────────────────────────────────────────────

print("Loading data for all tickers...")
datasets = {}
for ticker in TICKERS[:]:
    print(f"  Attempting {ticker}...")
    try:
        raw         = load_data(ticker, START_DATE, END_DATE)
        if raw is None or len(raw) == 0:
            raise ValueError("Empty data returned")
        transformed = transform_data(raw)
        X, y, prices = build_windows(transformed, WINDOW_SIZE, raw_data=raw)
        if len(X) == 0:
            raise ValueError("No windows could be built")
        datasets[ticker] = (X, y, prices)
        print(f"  {ticker}: {X.shape}")
    except Exception as e:
        print(f"  {ticker}: failed — {type(e).__name__}: {e}")
        TICKERS.remove(ticker)

if not datasets:
    raise RuntimeError("No tickers loaded successfully. Aborting.")

print(f"Loaded {len(datasets)} ticker(s): {list(datasets.keys())}\n")

# ── Model construction ────────────────────────────────────────────────────────

print("Building models...")
lstm      = LSTM(input_size=5, hidden_size=64, num_layers=2)
attention = MultiHeadAttention(hidden_size=64, num_heads=4)
cnn       = Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 5))
flatten   = Flatten()
nlp       = NLPEncoder(hidden_size=64)
regime    = RegimeDetector(input_size=5, hidden_size=32)
fusion    = FusionLayers(
    lstm_hidden_size=64,
    cnn_out_channels=CNN_FLAT_SIZE,
    nlp_hidden_size=64,
    hidden_size=64,
    risk_size=8,
)

# num_envs=1 — scalar training, single buffer set
agent = PPOAgent(
    state_size=FUSED_STATE_SIZE,
    action_size=2,
    lstm=lstm,
    attention=attention,
    cnn=cnn,
    flatten=flatten,
    regime=regime,
    fusion=fusion,
    num_envs=1,
)

load_model(agent, CHECKPOINT_PATH)
if RESET_CRITIC:
    agent.reset_critic()
if RESET_ACTOR:
    agent.reset_actor()

best_net_worth = load_best_net_worth(LOG_PATH, INITIAL_BALANCE)
print(f"Resuming with best net worth: {currency_units}{best_net_worth:.2f}")
print(f"Starting training for {EPISODES} episodes...\n")

# ── Training loop ─────────────────────────────────────────────────────────────

redirect_prints()
telemetry.start()

try:
    for episode in range(1, EPISODES + 1):
        reset_episode_log(episode)

        ticker = random.choice(list(datasets.keys()))
        X, y, prices = datasets[ticker]

        max_start = max(0, len(X) - EPISODE_STEPS)
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        X_ep      = X[start_idx : start_idx + EPISODE_STEPS]
        y_ep      = y[start_idx : start_idx + EPISODE_STEPS]
        prices_ep = prices[start_idx : start_idx + EPISODE_STEPS]

        is_mirrored_episode = random.random() < 0.5

        env = TradingEnvironment(
            X_ep, y_ep, lstm, attention, cnn, flatten,
            regime, fusion, nlp, prices_ep,
            initial_balance=INITIAL_BALANCE,
            friction=_friction_config,
            symbol=ticker,
            mirror_data=is_mirrored_episode,
        )
        env.precomputed_nlp = Tensor(np.zeros((1, 64), dtype=np.float64))

        state = env.reset()
        done  = False
        info  = {'net_worth': INITIAL_BALANCE}

        total_reward       = 0.0
        num_trades         = 0
        winning_trades     = 0
        episode_net_worths = [INITIAL_BALANCE]
        episode_bankrupt   = False
        episode_dir_means  = []
        ep_breakdown       = {k: 0.0 for k in env.last_reward_breakdown}
        step_count         = 0

        while not done:
            # ── Vectorized forward pass (batch=1) ─────────────────────────────
            actions_matrix = agent.select_vectorized_action([state])
            action         = actions_matrix[0]

            # Dir mean from raw probability vector — same source as vectorized runner
            probs           = agent._last_dir_probs[0]
            current_dir_mean = float(probs[2] - probs[0])   # Prob(Long) - Prob(Short)
            episode_dir_means.append(current_dir_mean)

            # ── Environment step ──────────────────────────────────────────────
            next_state, reward, done, info = env.step(action)

            if info.get('is_bankrupt', False):
                episode_bankrupt = True

            # ── Capture raw input + LSTM h/c for TBPTT ───────────────────────
            raw_s, h_snap, c_snap = env.get_raw_state()

            # Only store boundary snapshots; None on all other steps
            # (mirrors the vectorized runner's is_chunk_boundary logic)
            is_chunk_boundary = (step_count % TBPTT_CHUNK == 0)
            if not is_chunk_boundary:
                h_snap = None
                c_snap = None

            # ── Store transition (vectorized API, env index 0) ────────────────
            # Pass risk-adjusted action from info so PPO trains on actual sizes,
            # not the pre-risk raw sizes sampled from the policy.
            adjusted_action = info.get(
                'adjusted_action',
                np.array([action[0], action[1]], dtype=np.float64)
            )

            agent.store_vectorized_transition(
                rewards          = [reward],
                dones            = [done],
                adjusted_actions = [adjusted_action],
                raw_inputs       = [raw_s],
                lstm_h_snaps     = [h_snap],
                lstm_c_snaps     = [c_snap],
            )

            # ── Bookkeeping ───────────────────────────────────────────────────
            if env.last_trade_pnl is not None:
                num_trades += 1
                if env.last_trade_pnl > 0:
                    winning_trades += 1
                env.last_trade_pnl = None

            if env.current_step % TERMINAL_PRINTER == 0:
                price_idx = min(env.current_step - 1, env.total_steps - 1)
                print_step(episode, ticker, env.current_step,
                           env.total_steps, info['net_worth'],
                           env.position, env.prices[price_idx], current_dir_mean)

            total_reward += reward
            episode_net_worths.append(info['net_worth'])

            for k, v in env.last_reward_breakdown.items():
                ep_breakdown[k] = ep_breakdown.get(k, 0.0) + v

            # Rich Telemetry Upgrade: Pass explicit action elements representing the 2D action space
            act_dir_val = float(action[0])
            act_sz_val  = float(action[1])

            telemetry.update_step(
                ticker=ticker, episode=episode,
                step=env.current_step, total_steps=env.total_steps,
                net_worth=info['net_worth'], balance=env.balance,
                position=env.position,
                price=env.prices[min(env.current_step - 1, env.total_steps - 1)],
                std=agent.std, num_trades=num_trades,
                winning_trades=winning_trades, total_reward=total_reward,
                milestones_crossed=env.milestones_crossed,
                r_trade=env.last_reward_breakdown['trade'],
                r_step=env.last_reward_breakdown['step'],
                r_hold_loser=env.last_reward_breakdown['hold_loser'],
                r_stress=env.last_reward_breakdown['stress'],
                r_premature_close=env.last_reward_breakdown['premature_close'],
                r_milestone=env.last_reward_breakdown['milestone'],
                r_terminal=env.last_reward_breakdown['terminal'],
                r_total=env.last_reward_breakdown['total'],
                dir_mean=current_dir_mean,
                action_direction=act_dir_val,
                action_size=act_sz_val
            )

            if next_state is not None:
                state = next_state

            step_count += 1

        # ── Bootstrap terminal value, then PPO update ─────────────────────────
        final_next_values = agent.get_vectorized_values([state])

        head_norm, ext_norm, fus_norm = agent.update(next_values=final_next_values)
        telemetry.update_grad_norms(head_norm, ext_norm, fus_norm)

        # ── Episode metrics ───────────────────────────────────────────────────
        final_net_worth  = info['net_worth']
        win_rate         = winning_trades / num_trades if num_trades > 0 else 0.0
        max_drawdown     = compute_max_drawdown(episode_net_worths)
        sharpe           = compute_sharpe(episode_net_worths)
        sortino          = compute_sortino(episode_net_worths)

        _p0              = float(prices_ep[0]) if prices_ep[0] > 0 else float(prices_ep[prices_ep > 0][0])
        benchmark_return = round(float((prices_ep[-1] - _p0) / (_p0 + 1e-8)), 4)
        agent_growth     = round((final_net_worth / INITIAL_BALANCE) - 1.0, 4)
        alpha_vs_bh      = round(agent_growth - benchmark_return, 4)

        is_new_best = final_net_worth > best_net_worth
        avg_dir     = float(np.mean(episode_dir_means)) if episode_dir_means else 0.0

        # ── Logging ───────────────────────────────────────────────────────────
        log_data = {
            'episode':          episode,
            'ticker':           ticker,
            'friction_mode':    FRICTION_MODE,
            'total_reward':     round(total_reward,    4),
            'final_balance':    round(final_net_worth, 2),
            'growth_pct':       round(agent_growth * 100, 2),
            'benchmark_pct':    round(benchmark_return * 100, 2),
            'alpha_vs_bh_pct':  round(alpha_vs_bh * 100, 2),
            'sharpe':           sharpe,
            'sortino':          sortino,
            'num_trades':       num_trades,
            'win_rate':         round(win_rate,     4),
            'max_drawdown':     round(max_drawdown, 4),
            'avg_dir_mean':     round(avg_dir,      4),
            'std':              round(agent.std,     4),
            'entropy':          round(agent.entropy_coef, 4),
            'head_grad_norm':   round(head_norm, 4),
            'ext_grad_norm':    round(ext_norm,  4),
            'fus_grad_norm':    round(fus_norm,  4),
            'new_best':         is_new_best,
            'is_bankrupt':      episode_bankrupt,
            'is_mirrored':      is_mirrored_episode,
        }
        save_log(log_data, LOG_PATH)

        print_episode(episode, ticker, final_net_worth, total_reward,
                      num_trades, win_rate, agent.std, agent.entropy_coef,
                      best_net_worth, env.position, avg_dir,
                      episode_bankrupt, is_mirrored_episode)
        print(
            f"  [METRICS] Sharpe: {sharpe:+.3f} | Sortino: {sortino:+.3f} "
            f"| MaxDD: {max_drawdown:.2%} | BH: {benchmark_return:+.2%} "
            f"| Agent: {agent_growth:+.2%} | Alpha: {alpha_vs_bh:+.2%} "
            f"| GradNorms H/E/F: {head_norm:.4f}/{ext_norm:.4f}/{fus_norm:.4f}",
            flush=True,
        )

        if episode % 10 == 0:
            print_reward_breakdown(ep_breakdown, episode)

        telemetry.log_episode(
            episode=episode, ticker=ticker,
            final_balance=final_net_worth, total_reward=total_reward,
            num_trades=num_trades, win_rate=win_rate,
            max_drawdown=max_drawdown, std=agent.std,
            bankrupt=episode_bankrupt,
            dir_mean=avg_dir,
            sharpe=sharpe,
            sortino=sortino,
            benchmark_return=benchmark_return,
            alpha_vs_bh=alpha_vs_bh,
        )

        if is_new_best:
            best_net_worth = final_net_worth
            save_model(agent, BEST_MODEL_PATH)
            print(f"  ★ NEW BEST  {currency_units}{best_net_worth:.2f}"
                  f"  — model saved → {BEST_MODEL_PATH}")

        if episode % SAVE_EVERY == 0:
            save_model(agent, CHECKPOINT_PATH)
            print(f"  [Checkpoint] episode {episode} → {CHECKPOINT_PATH}")

finally:
    telemetry.stop()
    restore_prints()

print(f"\nTraining complete.")
print(f"Best Net Worth achieved: {currency_units}{best_net_worth:.2f}")
print(f"Best model saved to:     {BEST_MODEL_PATH}")