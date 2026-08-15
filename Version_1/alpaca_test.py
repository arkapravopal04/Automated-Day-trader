"""
alpaca_test.py — out-of-sample evaluation.

Loads the best saved model, runs one full pass through 2024-onwards data
for every ticker, collects metrics, and writes eval_log.csv.

No gradient updates.  No std decay.  No checkpoint saving.
Pure forward pass + metric collection.
"""

import numpy as np
import os
import sys
import csv

from engine import Tensor
from Neural_Nets import (LSTM, Conv2D, Flatten, Attention, MultiHeadAttention,
                         FusionLayers, RegimeDetector)
from nlp import NLPEncoder
from environment import TradingEnvironment, low_friction, realistic_friction, high_friction
from agent import PPOAgent
from models_utils import load_model, save_log

from telemetry import Telemetry
from alpaca_data import load_data, transform_data, build_windows


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

def print_episode_eval(episode, ticker, net_worth, reward, trades, win_rate,
                       position, avg_dir_mean, bankrupt=False):
    sys.stdout.write('\n')
    status = f" {RED}[BANKRUPT]{RESET}" if bankrupt else ""
    print(
        f"  Ep {episode:3d} | {ticker:6s} | "
        f"{GREEN if net_worth >= INITIAL_BALANCE else RED}"
        f"{currency_units}{net_worth:9.2f}{RESET}{status} "
        f"| Pos: {GREEN if position >= 0 else RED}{currency_units}{position:8.2f}{RESET} "
        f"| Reward: {reward:10.2f} "
        f"| Trades: {trades:4d} | WR: {win_rate:.1%} | AvgDir: {avg_dir_mean:+.3f}",
        flush=True,
    )

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

_ANN_FACTOR = np.sqrt(252 * 78)

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

def print_summary_table(results: list):
    print("\n" + "═" * 95)
    print(f"  {'TICKER':<8} {'GROWTH':>8} {'BH':>8} {'ALPHA':>8} "
          f"{'SHARPE':>8} {'SORTINO':>8} {'MAXDD':>7} {'TRADES':>7} {'WR':>6} {'BANKRUPT'}")
    print("─" * 95)

    for r in results:
        g_col = GREEN if r['agent_growth'] >= 0 else RED
        a_col = GREEN if r['alpha_vs_bh'] >= 0 else RED
        print(
            f"  {r['ticker']:<8} "
            f"{g_col}{r['agent_growth']:>+7.2%}{RESET} "
            f"{r['benchmark_return']:>+8.2%} "
            f"{a_col}{r['alpha_vs_bh']:>+8.2%}{RESET} "
            f"{r['sharpe']:>+8.3f} "
            f"{r['sortino']:>+8.3f} "
            f"{r['max_drawdown']:>7.2%} "
            f"{r['num_trades']:>7d} "
            f"{r['win_rate']:>6.1%} "
            f"{'YES' if r['bankrupt'] else 'no'}"
        )

    print("─" * 95)

    live   = [r for r in results if not r['bankrupt']]
    sample = live if live else results

    def _mean(key): return float(np.mean([r[key] for r in sample]))

    print(
        f"  {'MEAN (live)':<8} "
        f"{_mean('agent_growth'):>+8.2%}  "
        f"{_mean('benchmark_return'):>+7.2%} "
        f"{_mean('alpha_vs_bh'):>+8.2%} "
        f"{_mean('sharpe'):>+8.3f} "
        f"{_mean('sortino'):>+8.3f} "
        f"{_mean('max_drawdown'):>7.2%} "
        f"{int(_mean('num_trades')):>7d} "
        f"{_mean('win_rate'):>6.1%} "
        f"{sum(1 for r in results if r['bankrupt'])}/{len(results)} bankrupt"
    )
    print("═" * 95 + "\n")


BASE_PATH = '/kaggle/working' if os.path.exists('/kaggle') else '.'
os.makedirs(f"{BASE_PATH}/logs", exist_ok=True)

BEST_MODEL_PATH = f"{BASE_PATH}/models/best_model_vec.pkl" 
EVAL_LOG_PATH   = f"{BASE_PATH}/logs/eval_log.csv"

currency_units = "$"

TICKERS = ["SPY", "QQQ", "IWM", "XLE", "XBI", "GLD", "USO", "ARKK", "AAPL", "NVDA",
           "PYPL", "KRE", "UNG", "NFLX"]

EVAL_START_DATE = "2024-01-01"
EVAL_END_DATE   = None

WINDOW_SIZE      = 48
INITIAL_BALANCE  = 10_000
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

cnn_out_height = WINDOW_SIZE - 3 + 1
cnn_out_width  = 5 - 5 + 1
CNN_FLAT_SIZE  = 16 * cnn_out_height * cnn_out_width

FUSED_STATE_SIZE = 75

RED    = "\033[91m"
GREEN  = "\033[92m"
ORANGE = "\033[33m"
RESET  = "\033[0m"

telemetry = Telemetry(max_history=20)
telemetry.initial_balance = INITIAL_BALANCE

print("Loading test data...")
datasets = {}
for ticker in TICKERS[:]:
    try:
        raw = load_data(ticker, EVAL_START_DATE, EVAL_END_DATE)
        if raw is None or len(raw) == 0:
            raise ValueError("Empty data returned")
        transformed = transform_data(raw)
        X, y, prices = build_windows(transformed, WINDOW_SIZE, raw_data=raw)
        if len(X) == 0:
            raise ValueError("No windows could be built")
        datasets[ticker] = (X, y, prices)
        print(f"  {ticker}: {X.shape[0]:,} windows")
    except Exception as e:
        print(f"  {ticker}: skipped — {type(e).__name__}: {e}")
        TICKERS.remove(ticker)

if not datasets:
    raise RuntimeError("No tickers loaded. Run alpaca_prefetch_data.py first.")

print(f"\nLoaded {len(datasets)} ticker(s): {TICKERS}\n")

print("Building model architecture...")
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
    num_envs=1  # IMPORTANT: Set to 1 for sequential testing
)

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"No trained model found at {BEST_MODEL_PATH}.\n"
        f"Run vectorized_train.py first to produce a best_model_vec.pkl."
    )
load_model(agent, BEST_MODEL_PATH)
print(f"Model loaded from {BEST_MODEL_PATH}\n")

eval_std = agent.std

telemetry.start()
all_results = []

try:
    for episode, ticker in enumerate(sorted(datasets.keys()), start=1):
        X, y, prices = datasets[ticker]

        X_ep      = X
        y_ep      = y
        prices_ep = prices

        env = TradingEnvironment(
            X_ep, y_ep, lstm, attention, cnn, flatten,
            regime, fusion, nlp, prices_ep,
            initial_balance=INITIAL_BALANCE,
            friction=_friction_config,
            symbol=ticker,
            mirror_data=False, 
        )
        env.precomputed_nlp = Tensor(np.zeros((1, 64), dtype=np.float64))

        state = env.reset()
        agent.std = eval_std

        done  = False
        info  = {'net_worth': INITIAL_BALANCE}

        total_reward       = 0.0
        num_trades         = 0
        winning_trades     = 0
        episode_net_worths = [INITIAL_BALANCE]
        episode_bankrupt   = False
        episode_dir_means  = []

        print(f"[EVAL] {ticker} — {len(X_ep):,} steps", flush=True)

        while not done:
            # Trick the agent by wrapping the single state into a list, 
            # then extract the 0th action result
            actions_matrix = agent.select_vectorized_action([state])
            action = actions_matrix[0]

            # Extract the raw probabilities for the 0th (and only) environment
            probs = agent._last_dir_probs[0]
            current_dir_mean = float(probs[2] - probs[0]) # Prob(Long) - Prob(Short)
            
            episode_dir_means.append(current_dir_mean)

            next_state, reward, done, info = env.step(action)

            if info.get('is_bankrupt', False):
                episode_bankrupt = True

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

            # Capture action elements
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

        final_net_worth  = info['net_worth']
        win_rate         = winning_trades / num_trades if num_trades > 0 else 0.0
        max_drawdown     = compute_max_drawdown(episode_net_worths)
        sharpe           = compute_sharpe(episode_net_worths)
        sortino          = compute_sortino(episode_net_worths)
        avg_dir          = float(np.mean(episode_dir_means)) if episode_dir_means else 0.0

        _p0              = float(prices_ep[0]) if prices_ep[0] > 0 else float(prices_ep[prices_ep > 0][0])
        benchmark_return = round(float((prices_ep[-1] - _p0) / (_p0 + 1e-8)), 4)
        agent_growth     = round((final_net_worth / INITIAL_BALANCE) - 1.0, 4)
        alpha_vs_bh      = round(agent_growth - benchmark_return, 4)

        print_episode_eval(episode, ticker, final_net_worth, total_reward,
                           num_trades, win_rate, env.position, avg_dir,
                           episode_bankrupt)
        print(
            f"  [METRICS] Sharpe: {sharpe:+.3f} | Sortino: {sortino:+.3f} "
            f"| MaxDD: {max_drawdown:.2%} | BH: {benchmark_return:+.2%} "
            f"| Agent: {agent_growth:+.2%} | Alpha: {alpha_vs_bh:+.2%}",
            flush=True,
        )

        log_data = {
            'episode':          episode,
            'ticker':           ticker,
            'friction_mode':    FRICTION_MODE,
            'eval_start':       EVAL_START_DATE,
            'total_steps':      len(X_ep),
            'total_reward':     round(total_reward,    4),
            'final_balance':    round(final_net_worth, 2),
            'growth_pct':       round(agent_growth * 100, 2),
            'benchmark_pct':    round(benchmark_return * 100, 2),
            'alpha_vs_bh_pct':  round(alpha_vs_bh * 100, 2),
            'sharpe':           sharpe,
            'sortino':          sortino,
            'num_trades':       num_trades,
            'win_rate':         round(win_rate,        4),
            'max_drawdown':     round(max_drawdown,    4),
            'avg_dir_mean':     round(avg_dir,         4),
            'is_bankrupt':      episode_bankrupt,
        }
        save_log(log_data, EVAL_LOG_PATH)

        all_results.append({
            'ticker':           ticker,
            'agent_growth':     agent_growth,
            'benchmark_return': benchmark_return,
            'alpha_vs_bh':      alpha_vs_bh,
            'sharpe':           sharpe,
            'sortino':          sortino,
            'max_drawdown':     max_drawdown,
            'num_trades':       num_trades,
            'win_rate':         win_rate,
            'bankrupt':         episode_bankrupt,
        })

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

finally:
    telemetry.stop()

if all_results:
    print_summary_table(all_results)

print(f"Eval log saved → {EVAL_LOG_PATH}")
print("Done.\n")
