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
from environment import TradingEnvironment, realistic_friction
from agent import PPOAgent
from models_utils import save_model, load_model
from Vectorised_telemetry import VectorizedTelemetry
from alpaca_data import load_data, transform_data, build_windows
from log_redirect import redirect_prints, restore_prints, reset_episode_log

RED    = "\033[91m"
GREEN  = "\033[92m"
ORANGE = "\033[33m"
BLUE   = "\033[94m"
RESET  = "\033[0m"

currency_units = "$"
_ANN_FACTOR = np.sqrt(252 * 78)


def compute_sharpe(net_worths: list) -> float:
    if len(net_worths) < 2:
        return 0.0
    nw = np.array(net_worths, dtype=np.float64)
    returns = np.diff(nw) / (nw[:-1] + 1e-8)
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns))
    return round(mu / sigma * _ANN_FACTOR, 4) if sigma > 1e-10 else 0.0


def compute_sortino(net_worths: list) -> float:
    if len(net_worths) < 2:
        return 0.0
    nw = np.array(net_worths, dtype=np.float64)
    returns = np.diff(nw) / (nw[:-1] + 1e-8)
    mu = float(np.mean(returns))
    neg_returns = returns[returns < 0]
    if len(neg_returns) == 0:
        return round(mu * _ANN_FACTOR, 4) if mu > 0 else 0.0
    downside = float(np.std(neg_returns))
    return round(mu / downside * _ANN_FACTOR, 4) if downside > 1e-10 else 0.0


class VectorizedEnvRunner:
    """
    Synchronous Vectorized Environment Runner.
    Manages a suite of parallel TradingEnvironments loaded with different tickers.
    Aggregates states to allow batched network inferences, accelerating rollouts.
    """
    def __init__(self, tickers, datasets, lstm, attention, cnn, flatten, regime, fusion, nlp,
                 initial_balance=10000.0, friction_config=None, episode_steps=750):
        self.tickers       = tickers
        self.datasets      = datasets
        self.lstm          = lstm
        self.attention     = attention
        self.cnn           = cnn
        self.flatten       = flatten
        self.regime        = regime
        self.fusion        = fusion
        self.nlp           = nlp
        self.initial_balance  = initial_balance
        self.friction_config  = friction_config
        self.episode_steps    = episode_steps

        self.envs           = []
        self.active_tickers = []
        self.episode_start_indices = {}
        self.reset_all()

    def reset_all(self):
        """Initializes and resets all parallel environments with random window segments."""
        self.envs = []
        self.active_tickers = []
        self.episode_start_indices = {}

        for ticker in self.tickers:
            X, y, prices = self.datasets[ticker]
            max_start = max(0, len(X) - self.episode_steps)
            start_idx = np.random.randint(0, max_start) if max_start > 0 else 0

            X_ep      = X[start_idx : start_idx + self.episode_steps]
            y_ep      = y[start_idx : start_idx + self.episode_steps]
            prices_ep = prices[start_idx : start_idx + self.episode_steps]

            mirror = random.random() < 0.5

            env = TradingEnvironment(
                X_ep, y_ep, self.lstm, self.attention, self.cnn, self.flatten,
                self.regime, self.fusion, self.nlp, prices_ep,
                initial_balance=self.initial_balance,
                friction=self.friction_config,
                symbol=ticker,
                mirror_data=mirror
            )
            env.precomputed_nlp = Tensor(np.zeros((1, 64), dtype=np.float64))

            # Store the episode price slice so B&H is computed from the same window
            self.episode_start_indices[ticker] = (start_idx, prices_ep)

            self.envs.append(env)
            self.active_tickers.append(ticker)

        states = [env.reset() for env in self.envs]
        return states

    def step(self, actions):
        """Steps all environments forward by one timestep."""
        next_states, rewards, dones, infos = [], [], [], []
        for idx, env in enumerate(self.envs):
            next_state, reward, done, info = env.step(actions[idx])
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return next_states, rewards, dones, infos


def run_vectorized_training():
    BASE_PATH = '.'
    os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/logs",   exist_ok=True)

    BEST_MODEL_PATH = f"{BASE_PATH}/models/best_model_vec.pkl"
    CHECKPOINT_PATH = f"{BASE_PATH}/models/checkpoint_vec.pkl"
    LOG_PATH        = f"{BASE_PATH}/logs/vectorized_training_log.csv"

    # TICKERS = ["AAPL" , "ARKK" , "GLD" , "IWM" , "KRE" , "NFLX" , "NVDA", "PYPL" , "QQQ" , "SPY" , "UNG" , "USO" , "XBI" , "XLE"]
    TICKERS = ["SPY"]
    START_DATE = None
    END_DATE   = "2024-01-01"
    WINDOW_SIZE = 48

    # ── Training schedule ─────────────────────────────────────────────────────
    EPISODES      = 25
    EPISODE_STEPS = 750
    SAVE_EVERY    = 1
    # ──────────────────────────────────────────────────────────────────────────

    friction_config = realistic_friction()
    INITIAL_BALANCE = 10_000

    cnn_out_height = WINDOW_SIZE - 3 + 1
    cnn_out_width  = 5 - 5 + 1
    CNN_FLAT_SIZE  = 16 * cnn_out_height * cnn_out_width
    FUSED_STATE_SIZE = 75

    telemetry = VectorizedTelemetry(max_history=20)
    telemetry.set_episode_info(0, EPISODES)

    print("=" * 60)
    print("  INITIALIZING VECTORIZED PPO TRAINING PIPELINE")
    print(f"  Episodes: {EPISODES}  |  Steps/episode: {EPISODE_STEPS}")
    print(f"{BLUE}Run: 'Get-Content training_run.log -Wait' to see live...{RESET}")
    print("=" * 60)

    # ── Data loading ──────────────────────────────────────────────────────────
    datasets = {}
    for ticker in TICKERS[:]:
        try:
            raw = load_data(ticker, START_DATE, END_DATE)
            if raw is None or len(raw) == 0:
                raise ValueError("Empty data")
            transformed = transform_data(raw)
            X, y, prices = build_windows(transformed, WINDOW_SIZE, raw_data=raw)
            datasets[ticker] = (X, y, prices)
            print(f"  Loaded {ticker}: {X.shape}")
        except Exception as e:
            print(f"  {ticker} failed to load: {e}")
            TICKERS.remove(ticker)

    if not datasets:
        raise RuntimeError("No datasets successfully loaded. Aborting.")

    # ── Network construction ──────────────────────────────────────────────────
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

    agent = PPOAgent(
        state_size=FUSED_STATE_SIZE,
        action_size=2,
        lstm=lstm,
        attention=attention,
        cnn=cnn,
        flatten=flatten,
        regime=regime,
        fusion=fusion,
        num_envs=len(TICKERS)
    )

    load_model(agent, CHECKPOINT_PATH)

    vec_runner = VectorizedEnvRunner(
        tickers=TICKERS, datasets=datasets, lstm=lstm, attention=attention,
        cnn=cnn, flatten=flatten, regime=regime, fusion=fusion, nlp=nlp,
        initial_balance=INITIAL_BALANCE, friction_config=friction_config,
        episode_steps=EPISODE_STEPS
    )

    best_net_worth = INITIAL_BALANCE
    
    redirect_prints()   
    telemetry.start()

    try:
        for episode in range(1, EPISODES + 1):
            reset_episode_log(episode)
            telemetry.set_episode_info(episode, EPISODES)

            # ── Reset environments ────────────────────────────────────────────
            states = vec_runner.reset_all()

            env_net_worths  = {t: [INITIAL_BALANCE] for t in TICKERS}
            env_done_flags  = {t: False              for t in TICKERS}
            env_trade_counts = {t: 0                 for t in TICKERS}
            env_win_counts   = {t: 0                 for t in TICKERS}
            env_dir_means    = {t: 0.0               for t in TICKERS}

            step_count = 0
            TBPTT_CHUNK = 32   # must match agent.py CHUNK_SIZE

            while step_count < EPISODE_STEPS:
                # ── True Vectorized Forward Pass ──────────────────────────────
                actions_matrix = agent.select_vectorized_action(states)
                actions_list   = list(actions_matrix)

                # Store telemetry tracking probabilities natively
                for s_idx, t in enumerate(TICKERS):
                    probs = agent._last_dir_probs[s_idx]
                    env_dir_means[t] = float(probs[2] - probs[0])

                next_states, rewards, dones, infos = vec_runner.step(actions_list)

                # ── Snapshot raw inputs and hidden states ─────────────────────
                # Snapshot h/c at chunk boundaries so TBPTT replay can seed correctly
                is_chunk_boundary = (step_count % TBPTT_CHUNK == 0)

                raw_inputs   = []
                lstm_h_snaps = []
                lstm_c_snaps = []

                for idx, env in enumerate(vec_runner.envs):
                    raw_s, h_snap, c_snap = env.get_raw_state()
                    raw_inputs.append(raw_s)
                    # Only store h/c at boundaries — None otherwise saves memory
                    if is_chunk_boundary:
                        lstm_h_snaps.append(h_snap)
                        lstm_c_snaps.append(c_snap)
                    else:
                        lstm_h_snaps.append(None)
                        lstm_c_snaps.append(None)

                # Transition data correctly segregated by ticker internally
                adjusted_actions = [infos[i]['adjusted_action'] for i in range(len(TICKERS))]
                agent.store_vectorized_transition(
                    rewards, dones,
                    adjusted_actions=adjusted_actions,
                    raw_inputs=raw_inputs,
                    lstm_h_snaps=lstm_h_snaps,
                    lstm_c_snaps=lstm_c_snaps,
                )

                # ── Telemetry + bookkeeping ───────────────────────────────────
                batch_telemetry     = []
                avg_reward_breakdown = {}

                for idx, t in enumerate(TICKERS):
                    env_ref = vec_runner.envs[idx]

                    if not env_done_flags[t]:
                        env_net_worths[t].append(infos[idx]['net_worth'])
                        env_done_flags[t] = dones[idx]
                        env_trade_counts[t] = env_ref.n_trades_this_episode

                        if env_ref.last_trade_pnl is not None:
                            if env_ref.last_trade_pnl > 0:
                                env_win_counts[t] += 1
                            env_ref.last_trade_pnl = None

                    wr = (env_win_counts[t] / env_trade_counts[t]
                          if env_trade_counts[t] > 0 else 0.0)

                    batch_telemetry.append({
                        'ticker':      t,
                        'step':        step_count,
                        'total_steps': EPISODE_STEPS,
                        'net_worth':   infos[idx]['net_worth'],
                        'position':    env_ref.position,
                        'price':       infos[idx]['price'],
                        'reward':      env_ref.last_reward_breakdown['total'],
                        'trades':      env_trade_counts[t],
                        'win_rate':    wr,
                        'dir_mean':    env_dir_means[t],       
                    })

                    for k, v in env_ref.last_reward_breakdown.items():
                        avg_reward_breakdown[k] = avg_reward_breakdown.get(k, 0.0) + (v / len(TICKERS))

                telemetry.update_step(batch_telemetry)
                telemetry.update_rewards(avg_reward_breakdown)

                states = next_states
                step_count += 1

            # ── Rollout complete — bootstrap value + PPO update ───────────────
            final_next_values = agent.get_vectorized_values(states)

            head_norm, ext_norm, fus_norm = agent.update(next_values=final_next_values)
            telemetry.update_grad_norms(head_norm, ext_norm, fus_norm)

            # ── Analytical metrics per env ────────────────────────────────────
            avg_sharpes, avg_sortinos, avg_alphas = [], [], []
            avg_bhs, avg_growths                  = [], []
            avg_final_balances, avg_trades, avg_wrs = [], [], []

            for idx, t in enumerate(TICKERS):
                nw_arr = env_net_worths[t]

                sharpe  = compute_sharpe(nw_arr)
                sortino = compute_sortino(nw_arr)

                _, prices_ep = vec_runner.episode_start_indices[t]
                p0   = float(prices_ep[0])
                pend = float(prices_ep[min(EPISODE_STEPS - 1, len(prices_ep) - 1)])
                bh_return = (pend - p0) / (p0 + 1e-8)

                growth = (nw_arr[-1] / INITIAL_BALANCE) - 1.0
                alpha  = growth - bh_return

                trades = env_trade_counts[t]
                wr     = (env_win_counts[t] / trades) if trades > 0 else 0.0

                avg_sharpes.append(sharpe)
                avg_sortinos.append(sortino)
                avg_alphas.append(alpha)
                avg_bhs.append(bh_return)
                avg_growths.append(growth)
                avg_final_balances.append(nw_arr[-1])
                avg_trades.append(trades)
                avg_wrs.append(wr)

            overall_avg_nw  = float(np.mean(avg_final_balances))
            overall_growth  = float(np.mean(avg_growths))
            overall_bh      = float(np.mean(avg_bhs))
            overall_alpha   = float(np.mean(avg_alphas))
            overall_sharpe  = float(np.mean(avg_sharpes))
            overall_sortino = float(np.mean(avg_sortinos))
            overall_trades  = float(np.mean(avg_trades))
            overall_wr      = float(np.mean(avg_wrs))

            is_new_best = overall_avg_nw > best_net_worth

            telemetry.log_iteration(
                episode, overall_avg_nw, overall_growth, overall_bh, overall_alpha,
                overall_sharpe, overall_sortino, overall_wr, overall_trades, is_new_best
            )

            # ── Multi-Row CSV logging ─────────────────────────────────────────
            log_file_exists = os.path.isfile(LOG_PATH)
            with open(LOG_PATH, 'a', newline='') as csvfile:
                fieldnames = ['episode', 'ticker', 'net_worth', 'growth_pct', 'benchmark_pct',
                              'alpha_pct', 'sharpe', 'sortino', 'win_rate', 'trades',
                              'head_grad_norm', 'ext_grad_norm', 'fus_grad_norm', 'is_new_best']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not log_file_exists:
                    writer.writeheader()

                # Appends an independent row for every ticker representing this episode
                for idx, t in enumerate(TICKERS):
                    writer.writerow({
                        'episode': episode,
                        'ticker': t,
                        'net_worth': round(avg_final_balances[idx], 2),
                        'growth_pct': round(avg_growths[idx] * 100, 2),
                        'benchmark_pct': round(avg_bhs[idx] * 100, 2),
                        'alpha_pct': round(avg_alphas[idx] * 100, 2),
                        'sharpe': round(avg_sharpes[idx], 3),
                        'sortino': round(avg_sortinos[idx], 3),
                        'win_rate': round(avg_wrs[idx], 3),
                        'trades': round(avg_trades[idx], 1),
                        'head_grad_norm': round(head_norm, 4),
                        'ext_grad_norm': round(ext_norm, 4),
                        'fus_grad_norm': round(fus_norm, 4),
                        'is_new_best': is_new_best
                    })

            if is_new_best:
                best_net_worth = overall_avg_nw
                save_model(agent, BEST_MODEL_PATH)

            if episode % SAVE_EVERY == 0:
                save_model(agent, CHECKPOINT_PATH)

    finally:
        telemetry.stop()
        restore_prints()


if __name__ == "__main__":
    run_vectorized_training()