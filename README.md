# Automated Day Trader

A reinforcement-learning trading system, I built twice: once from scratch in raw NumPy to learn how it works, then rewritten in PyTorch to actually run.

> **Paper trade before real money. Always.** Personal project, use at your own risk.

---

## Version 1...built from nothing

No PyTorch. No TensorFlow. Just NumPy and a custom autograd engine.

Hand-written LSTM, MultiHeadAttention, Conv2D, LayerNorm. PPO with GAE. FinBERT sentiment fused into the price stream. Fractional Kelly sizing, ATR stops, black-swan detection.

It worked as a proof of concept, but it wasreally slow and couldnt be scaled up. Honestly that was the point of the project [have a quick look](Version_1/README.md).

## Version 2...actually built to deploy

One idea drives the rewrite: **the same action pipeline runs everywhere.** Training, backtest, and live execute an identical sequence: policy → Kelly → risk caps → kill switch → execution. Nothing is stubbed out for training and nothing gets bolted on at deployment. A backtest here is a description of the live system, not an approximation of it.

```
Alpaca 5-min bars (99 tickers)
   ↓ fetch / preprocess → parquet feature store
   ↓ vectorized multi-env tensors  [n_envs × window × features]
   ↓ CNN + LSTM encoders → cross-attention → dual critic
   ↓ action: direction, size, limit offset
   ↓ KellySizer → RiskManager → KillSwitch
   ↓ Alpaca — paper by default, live only on explicit opt-in
   ↓ reconciliation every cycle + live dashboard
```

What the design actually buys you:

- **99 tickers trained in parallel**:  vectorized envs, PPO + DDP. Far more sample-efficient than one agent per symbol.
- **Deterministic live inference**: PPO's exploration noise never touches capital.
- **Paper and live keys are separate env vars**: a misconfigured "paper" run cannot silently reach real money.
- **Reconciliation before every order**: internal ledger vs broker. Mismatch or unreachable broker halts that cycle.
- **A kill switch that never auto-resets**: daily loss limit, broker-error streak, state mismatch. Resuming is a human decision.

`env/` · `model/` · `training/` · `risk/` · `live/` · `eval/` · `monitoring/` · `data/`

## Where does it stands now?

Training in progress on Kaggle. No live capital. Backtest and paper results get posted when they exist — not before.

**Next:** cost-aware walk-forward backtests (Sharpe, max drawdown, vs buy-and-hold) → extended paper run → small real size with hard loss limits.

## Setup

```bash
cd Version_2
pip install -r requirements.txt
# TRADING_ALPACA_PAPER_KEY / _SECRET   (paper)
# TRADING_ALPACA_LIVE_KEY  / _SECRET   (live)
```

Module-by-module breakdown: [`Version_2/README.md`](Version_2/README.md)

thanks!