"""
Main training loop — PPO agent on a multi-ticker stock trading environment.
"""
 
import numpy as np
import random
import os
import sys
import csv
 
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Attention, FusionLayers, RegimeDetector
from losses import CrossEntropyLoss
from data import load_data, transform_data, build_windows, generate_regime_labels, DataLoader
from nlp import NLPEncoder
from env import TradingEnvironment
from agent import PPOAgent
from models_utils import save_model, load_model, save_log

def print_step(episode, ticker, step, total_steps, net_worth, position, price):
    msg = (f"  [{ticker}] Ep{episode} | Step {step}/{total_steps} "
           f"| NetWorth: {currency_units}{net_worth:8.2f} | Pos: {currency_units}{position:8.2f} | Price: {currency_units}{price:.2f}")
    sys.stdout.write('\r' + msg + ' ' * 10)
    sys.stdout.flush()
 
 
def print_episode(episode, ticker, net_worth, reward, trades, win_rate, std, best, position):
    sys.stdout.write('\n')
    star = '★' if net_worth >= best else ' '
    print(
        f"{star} Ep {episode:3d} | {ticker:6s} | NetWorth: {currency_units}{net_worth:9.2f} "
        f"| Pos: {currency_units}{position:8.2f} | Reward: {reward:10.2f} "
        f"| Trades: {trades:4d} | WR: {win_rate:.1%} | Std: {std:.3f}",
        flush=True
    )
 
 
def compute_max_drawdown(net_worths):
    peak = net_worths[0]
    max_dd = 0.0
    for nw in net_worths:
        if nw > peak:
            peak = nw
        dd = (peak - nw) / (peak + 1e-8)
        if dd > max_dd:
            max_dd = dd
    return max_dd

def load_best_net_worth(log_path, initial_balance):
    if not os.path.exists(log_path):
        return initial_balance
    best = initial_balance
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nw = float(row['final_balance'])
            if nw > best:
                best = nw
    return best
 

BASE_PATH = '/kaggle/working' if os.path.exists('/kaggle') else '.'
os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/logs", exist_ok=True)
 
BEST_MODEL_PATH  = f"{BASE_PATH}/models/best_model.pkl"
CHECKPOINT_PATH  = f"{BASE_PATH}/models/checkpoint.pkl"
LOG_PATH         = f"{BASE_PATH}/logs/training_log.csv"

# $ , ₹, €
currency_units = "₹"  # use symbol corresponding to currency
TICKERS        = ["RELIANCE.NS"]   
START_DATE     = "2015-01-01"
END_DATE       = "2025-01-01"
WINDOW_SIZE    = 10
EPISODES       = 50
SAVE_EVERY     = 10
# do update initial balance in env.py
INITIAL_BALANCE = 10000

CNN_FLAT_SIZE  = 128
STATE_SIZE     = 64 + CNN_FLAT_SIZE + 64 + 3   
FUSED_STATE_SIZE = 67

print("Loading data for all tickers...")
datasets = {}
for ticker in TICKERS:
    try:
        raw = load_data(ticker, START_DATE, END_DATE)
        transformed = transform_data(raw)
        X, y, prices = build_windows(transformed, WINDOW_SIZE, raw_data=raw)
        datasets[ticker] = (X, y, prices)
        print(f"  {ticker}: {X.shape}")
    except Exception as e:
        print(f"  {ticker}: failed — {e}")
 
if not datasets:
    raise RuntimeError("No tickers loaded successfully. Aborting.")
 
print(f"Loaded {len(datasets)} ticker(s)\n")

print("Building models...")
lstm      = LSTM(input_size=5, hidden_size=64, num_layers=2)
attention = Attention(hidden_size=64)
cnn       = Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 5))
flatten   = Flatten()
nlp       = NLPEncoder(hidden_size=64)
regime    = RegimeDetector(input_size=5, hidden_size=32)
fusion    = FusionLayers(
    lstm_hidden_size=64,
    cnn_out_channels=CNN_FLAT_SIZE,
    nlp_hidden_size=64,
    hidden_size=64
)
# better than before but still not great. The agent can learn some patterns but struggles to consistently outperform the baseline.
agent = PPOAgent(
    state_size=FUSED_STATE_SIZE,
    action_size=2,
    lstm=lstm,
    attention=attention,
    cnn=cnn,
    flatten=flatten,
    regime=regime,
    fusion=fusion,
)


load_model(agent, CHECKPOINT_PATH)


extractor_param_count = len(agent._extractor_parameters())
print(f"Extractor params: {extractor_param_count}")
print(f"Extractor optimizer: {'active' if agent.extractor_optimizer else 'NONE'}")
 
best_net_worth = load_best_net_worth(LOG_PATH, INITIAL_BALANCE)
print(f"Resuming with best net worth: {currency_units}{best_net_worth:.2f}")
print(f"Starting training for {EPISODES} episodes...\n")
for episode in range(1, EPISODES + 1):
    ticker = random.choice(list(datasets.keys()))
    X, y, prices = datasets[ticker]
 
    env = TradingEnvironment(
        X, y, lstm, attention, cnn, flatten,
        regime, fusion, nlp, prices,
        initial_balance=INITIAL_BALANCE
    )
    env.precomputed_nlp = Tensor(np.zeros((1, 64), dtype=np.float64))
 
    state = env.reset()
    done  = False
 
    total_reward    = 0.0
    num_trades      = 0
    winning_trades  = 0
    episode_net_worths = [INITIAL_BALANCE]
 
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.rewards.append(reward)
 
        if env.last_trade_pnl is not None:
            num_trades += 1
            if env.last_trade_pnl > 0:
                winning_trades += 1
            env.last_trade_pnl = None   
 
        if env.current_step % 25 == 0:
            price_idx = min(env.current_step - 1, env.total_steps - 1)
            print_step(episode, ticker, env.current_step,
                       env.total_steps, env.net_worth,
                       env.position, env.prices[price_idx])
 
        total_reward += reward
        episode_net_worths.append(env.net_worth)
 
        if next_state is not None:
            state = next_state
    agent.update()
 
    final_net_worth = env.net_worth
    win_rate        = winning_trades / num_trades if num_trades > 0 else 0.0
    max_drawdown    = compute_max_drawdown(episode_net_worths)
 
    is_new_best = final_net_worth > best_net_worth
 
    log_data = {
        'episode':       episode,
        'ticker':        ticker,
        'total_reward':  round(total_reward, 4),
        'final_balance': round(final_net_worth, 2),
        'num_trades':    num_trades,
        'win_rate':      round(win_rate, 4),
        'max_drawdown':  round(max_drawdown, 4),
        'std':           round(agent.std, 4),
        'new_best':      is_new_best,
    }
    save_log(log_data, LOG_PATH)
 
    print_episode(episode, ticker, final_net_worth, total_reward,
                  num_trades, win_rate, agent.std, best_net_worth, env.position)
 
    if is_new_best:
        best_net_worth = final_net_worth
        save_model(agent, BEST_MODEL_PATH)
        print(f"  ★ NEW BEST  {currency_units}{best_net_worth:.2f}  — model saved → {BEST_MODEL_PATH}")
 
    if episode % SAVE_EVERY == 0:
        save_model(agent, CHECKPOINT_PATH)
        print(f"  [Checkpoint] episode {episode} → {CHECKPOINT_PATH}")
 
 
print(f"\nTraining complete.")
print(f"Best Net Worth achieved: {currency_units}{best_net_worth:.2f}")
print(f"Best model saved to:     {BEST_MODEL_PATH}")
 