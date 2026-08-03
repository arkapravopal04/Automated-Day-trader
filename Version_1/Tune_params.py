"""
tune_hyperparams.py — automated hyperparameter optimization harness.

Implements a lightweight Random Search sweep across the key PPO parameters.
Evaluates configurations across a validation subset of tickers, tracking
growth, Sharpe ratio, and trading activity to find the most profitable profile.
"""

import os
import sys
import random
import numpy as np
import csv
import time

from engine import Tensor
from Neural_Nets import (LSTM, Conv2D, Flatten, MultiHeadAttention,
                         FusionLayers, RegimeDetector)
from nlp import NLPEncoder
from environment import TradingEnvironment, realistic_friction
from agent import PPOAgent
from models_utils import save_model, save_log
from alpaca_data import load_data, transform_data, build_windows

# ── Configuration ──────────────────────────────────────────────────────────────
TICKERS_VAL    = ["AAPL", "SPY", "QQQ"]   # Representative subset for rapid evaluation
START_DATE     = None
END_DATE       = "2024-01-01"
WINDOW_SIZE    = 48
VAL_EPISODES   = 3       # Validation episodes per candidate
EPISODE_STEPS  = 600     # Shortened step count for faster tuning cycles
INITIAL_BALANCE = 10_000

BASE_PATH      = '.'
os.makedirs(f"{BASE_PATH}/tuning", exist_ok=True)
TUNING_LOG_PATH = f"{BASE_PATH}/tuning/hyperparam_sweep.csv"

# ── Search space ───────────────────────────────────────────────────────────────
PARAM_SPACE = {
    'gamma':         [0.98, 0.99, 0.995],
    'epsilon':       [0.10, 0.15, 0.20, 0.25],
    'head_lr':       [5e-5, 1e-4, 2e-4],
    'extractor_lr':  [2e-5, 5e-5, 8e-5],
    'entropy_coef':  [0.005, 0.01, 0.02, 0.03],
    'entropy_decay': [0.992, 0.995, 0.998],
    'std_min':       [0.10, 0.15, 0.20],
    'std_decay':     [0.990, 0.995, 0.998],
    'lam': [0.90, 0.92, 0.95, 0.97],
    'epochs': [2, 4, 8]
}


def sample_parameters(space: dict) -> dict:
    return {k: random.choice(v) for k, v in space.items()}


def evaluate_parameters(params: dict, datasets: dict, cnn_flat_size: int) -> float:
    """
    Runs a shortened validation loop for a candidate hyperparameter set.

    Returns a unified objective score weighting:
      1. Returns (growth)
      2. Risk-adjusted consistency (Sharpe)
      3. Market participation (penalise agents that never trade)
    """
    scores = []

    for ep in range(1, VAL_EPISODES + 1):
        ticker = random.choice(list(datasets.keys()))
        X, y, prices = datasets[ticker]

        max_start = max(0, len(X) - EPISODE_STEPS)
        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
        X_ep      = X[start_idx : start_idx + EPISODE_STEPS]
        y_ep      = y[start_idx : start_idx + EPISODE_STEPS]
        prices_ep = prices[start_idx : start_idx + EPISODE_STEPS]

        # ── Fresh network per trial (avoids state bleed between candidates) ──
        lstm      = LSTM(input_size=5, hidden_size=64, num_layers=2)
        attention = MultiHeadAttention(hidden_size=64, num_heads=4)
        cnn       = Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 5))
        flatten   = Flatten()
        nlp       = NLPEncoder(hidden_size=64)
        regime    = RegimeDetector(input_size=5, hidden_size=32)
        fusion    = FusionLayers(
            lstm_hidden_size=64,
            cnn_out_channels=cnn_flat_size,
            nlp_hidden_size=64,
            hidden_size=64,
            risk_size=8,
        )

        agent = PPOAgent(
            state_size=75,
            action_size=2,
            lstm=lstm,
            attention=attention,
            cnn=cnn,
            flatten=flatten,
            regime=regime,
            fusion=fusion,
        )

        # Inject candidate hyperparameters
        agent.gamma              = params['gamma']
        agent.epsilon            = params['epsilon']
        agent.head_lr            = params['head_lr']
        agent.extractor_lr       = params['extractor_lr']
        agent.entropy_coef_start = params['entropy_coef']
        agent.entropy_coef       = params['entropy_coef']
        agent.entropy_decay      = params['entropy_decay']
        agent.std_min            = params['std_min']
        agent.std_decay          = params['std_decay']
        agent.lam = params['lam']
        agent.epochs = params['epochs']

        friction = realistic_friction()
        env = TradingEnvironment(
            X_ep, y_ep, lstm, attention, cnn, flatten,
            regime, fusion, nlp, prices_ep,
            initial_balance=INITIAL_BALANCE,
            friction=friction,
            symbol=ticker,
            mirror_data=False
        )
        env.precomputed_nlp = Tensor(np.zeros((1, 64), dtype=np.float64))

        state = env.reset()
        # Clear agent buffers before the rollout
        agent.states, agent.actions, agent.rewards, agent.log_probs, agent.values = [], [], [], [], []

        done         = False
        info         = {'net_worth': INITIAL_BALANCE}
        total_trades = 0
        episode_net_worths = [INITIAL_BALANCE]

        while not done:
            action     = agent.select_action(state)
            agent.store_transition()                    # stores (s, a, logp, v)

            next_state, reward, done, info = env.step(action)
            agent.rewards.append(reward)               # reward appended after step

            if env.last_trade_pnl is not None:
                total_trades += 1
                env.last_trade_pnl = None

            episode_net_worths.append(info['net_worth'])
            if next_state is not None:
                state = next_state

        # Bootstrap value from terminal state (or 0 if episode ended naturally)
        if state is not None and not done:
            bootstrap_val = float(agent._critic_forward(state).data.flat[0])
        else:
            bootstrap_val = 0.0

        agent.update(next_value=bootstrap_val)

        # ── Performance metrics ──────────────────────────────────────────────
        final_nw = info['net_worth']
        growth   = (final_nw / INITIAL_BALANCE) - 1.0

        nws     = np.array(episode_net_worths, dtype=np.float64)
        returns = np.diff(nws) / (nws[:-1] + 1e-8)
        std_ret = float(np.std(returns))
        sharpe  = (float(np.mean(returns)) / std_ret * np.sqrt(252 * 78)
                   if std_ret > 1e-8 else -2.0)

        participation_penalty = -2.0 if total_trades < 2 else 0.0

        scores.append(growth * 10.0 + sharpe * 0.5 + participation_penalty)

    return float(np.mean(scores))


def main():
    print("=" * 60)
    print("  PPO HYPERPARAMETER TUNING HARNESS STARTED")
    print("=" * 60)

    # ── Load validation datasets ───────────────────────────────────────────────
    datasets = {}
    print("Loading validation datasets...")
    for ticker in TICKERS_VAL:
        try:
            raw = load_data(ticker, START_DATE, END_DATE)
            if raw is None or len(raw) == 0:
                continue
            transformed = transform_data(raw)
            X, y, prices = build_windows(transformed, WINDOW_SIZE, raw_data=raw)
            datasets[ticker] = (X, y, prices)
            print(f"  Loaded {ticker}: {X.shape}")
        except Exception as e:
            print(f"  Error loading {ticker}: {e}")

    if not datasets:
        print("No validation datasets loaded. Aborting.")
        sys.exit(1)

    cnn_out_height = WINDOW_SIZE - 3 + 1
    cnn_out_width  = 5 - 5 + 1
    cnn_flat_size  = 16 * cnn_out_height * cnn_out_width

    num_trials   = 15
    best_score   = -9999.0
    best_params  = {}

    fieldnames = ['trial', 'score'] + list(PARAM_SPACE.keys())
    file_exists = os.path.exists(TUNING_LOG_PATH)
    with open(TUNING_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    print(f"\nBeginning random search over {num_trials} trials...\n")

    for trial in range(1, num_trials + 1):
        t0        = time.time()
        candidate = sample_parameters(PARAM_SPACE)

        print(f"[Trial {trial}/{num_trials}] Candidate parameters:")
        for k, v in candidate.items():
            print(f"  {k:<15} : {v}")

        try:
            score   = evaluate_parameters(candidate, datasets, cnn_flat_size)
            elapsed = time.time() - t0
            print(f"--> Done in {elapsed:.1f}s | Objective Score: {score:+.4f}")

            log_entry = {'trial': trial, 'score': round(score, 4)}
            log_entry.update(candidate)
            with open(TUNING_LOG_PATH, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(log_entry)

            if score > best_score:
                best_score  = score
                best_params = candidate
                print(f"  ★★★ NEW BEST! Score: {best_score:+.4f}")

        except Exception as e:
            print(f"  [ERROR] Trial {trial} failed: {e}")
            import traceback; traceback.print_exc()

        print("-" * 50)

    print("\n" + "=" * 60)
    print("  TUNING COMPLETE")
    print("=" * 60)
    print(f"Best Objective Score : {best_score:+.4f}")
    print("Optimal Hyperparameter Profile:")
    for k, v in best_params.items():
        print(f"  {k:<20} : {v}")
    print(f"Trace saved to: {TUNING_LOG_PATH}\n")


if __name__ == "__main__":
    main()