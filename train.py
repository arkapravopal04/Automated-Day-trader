'''
this is where the train loop of the entire model sits at
'''

import numpy as np
import random
import os
import sys
from engine import Tensor
from Neural_Nets import LSTM, Conv2D, Flatten, Linear, Attention, FusionLayers, RegimeDetector, LayerNorm, Dropout
from losses import CrossEntropyLoss
from data import load_data, transform_data, build_windows, generate_regime_labels, DataLoader
from nlp import NLPEncoder
from env import TradingEnvironment
from agent import PPOAgent
from models_utils import save_model, load_model, save_log

def print_step(episode, ticker, step, total_steps, net_worth, position, price):
    msg = f"  [{ticker}] Ep{episode} | Step {step}/{total_steps} | NetWorth: ${net_worth:8.2f} | Pos: ${position:8.2f} | Price: ${price:.2f}"
    sys.stdout.write('\r' + msg + ' ' * 10)
    sys.stdout.flush()

def print_episode(episode, ticker, net_worth, reward, trades, win_rate, std, best, position):
    sys.stdout.write('\n')  
    star = '★' if net_worth >= best else ' '
    print(f"{star} Ep {episode:3d} | {ticker:6s} | NetWorth: ${net_worth:9.2f} | Pos: ${position:8.2f}| Reward: {reward:10.2f} | Trades: {trades:4d} | WR: {win_rate:.1%} | Std: {std:.3f}", flush=True)

if os.path.exists('/kaggle'):
    BASE_PATH = '/kaggle/working'
else:
    BASE_PATH = '.'

os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/logs", exist_ok=True)

BEST_MODEL_PATH = f"{BASE_PATH}/models/best_model.pkl"
CHECKPOINT_PATH = f"{BASE_PATH}/models/checkpoint.pkl"
LOG_PATH = f"{BASE_PATH}/logs/training_log.csv"

def compute_max_drawdown(net_worths):
    peak = net_worths[0]
    drawdown = 0
    for nw in net_worths:
        if nw > peak:
            peak = nw
        dd = (peak - nw) / peak
        if dd > drawdown:
            drawdown = dd
    return drawdown

# TICKERS = ["JPM", "AAPL", "NVDA", "TSLA", "META", "AMZN", "SPY", "GOOGL", "MSFT","PLTR"] 
TICKERS = ["JPM"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
WINDOW_SIZE = 10
EPISODES = 10
SAVE_EVERY = 50
INITIAL_BALANCE = 10000 

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

print(f"Loaded {len(datasets)} tickers\n")

print("Building models...")
lstm      = LSTM(input_size=5, hidden_size=64, num_layers=2)
attention = Attention(hidden_size=64)
cnn       = Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 5))
flatten   = Flatten()
nlp       = NLPEncoder(hidden_size=64)
regime    = RegimeDetector(input_size=5, hidden_size=32)
fusion    = FusionLayers(lstm_hidden_size=64, cnn_out_channels=128,
                         nlp_hidden_size=64, hidden_size=64)
agent     = PPOAgent(state_size=67, action_size=2)

best_net_worth = 0
print(f"Starting training for {EPISODES} episodes...\n")

for episode in range(1, EPISODES + 1):
    ticker = random.choice(list(datasets.keys()))
    X, y, prices = datasets[ticker]
    
    env = TradingEnvironment(
        X, y, lstm, attention, cnn, flatten,
        regime, fusion, nlp, prices,
        initial_balance=INITIAL_BALANCE
    )
    
    env.precomputed_nlp = Tensor(np.zeros((1, 64)))

    state = env.reset()
    done  = False
    threashold = 0.5
    total_reward   = 0
    num_trades     = 0
    winning_trades = 0
    episode_net_worths = [INITIAL_BALANCE]
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.rewards.append(reward)
        direction, size = action
        if abs(direction) > threashold: 
            num_trades += 1
            if reward > 0:
                winning_trades += 1

        if env.current_step % 50 == 0:
            print_step(episode, ticker, env.current_step, 
                    env.total_steps, env.net_worth, 
                    env.position, env.prices[env.current_step-1])
            
        total_reward += reward
        episode_net_worths.append(env.net_worth)
        
        if next_state is not None:
            state = next_state
    
    agent.update()
    final_net_worth = env.net_worth
    win_rate        = winning_trades / num_trades if num_trades > 0 else 0
    max_drawdown    = compute_max_drawdown(episode_net_worths)
    log_data = {
        'episode':       episode,
        'ticker':        ticker,
        'total_reward':  round(total_reward, 4),
        'final_balance': round(final_net_worth, 2), 
        'num_trades':    num_trades,
        'win_rate':      round(win_rate, 4),
        'max_drawdown':  round(max_drawdown, 4),
        'std':           round(agent.std, 4),
    }
    save_log(log_data, LOG_PATH)
    
    print_episode(episode, ticker, final_net_worth, total_reward, 
              num_trades, win_rate, agent.std,
              best_net_worth, env.position)
    
    if final_net_worth > best_net_worth:
        best_net_worth = final_net_worth
        save_model(agent, BEST_MODEL_PATH)
        print(f"  New best Net Worth: ${best_net_worth:.2f}")
    
    if episode % SAVE_EVERY == 0:
        save_model(agent, CHECKPOINT_PATH)
        print(f" Checkpoint saved at episode {episode}")

print(f"\nTraining complete.")
print(f"Best Net Worth achieved: ${best_net_worth:.2f}")
print(f"Model saved to: {BEST_MODEL_PATH}")