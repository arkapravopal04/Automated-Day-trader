# Automated Day Trader

A reinforcement-learning day-trading system built across two generations: a from-scratch learning-first prototype, and a production-oriented vectorized rewrite with a hardened risk layer and paper-first live trading.

> **Paper trade before real money. Always.**
> This is a personal project — use at your own risk. Do not risk money you cannot afford to lose.

---

## Version 1 — The from-scratch foundation (legacy)

The learning project that proved the concepts. Everything built from first principles — **no PyTorch, no TensorFlow, just NumPy**:

- Custom autograd engine with full backprop (`Tensor`, matmul, im2col conv, softmax, etc.)
- Hand-written layers: LSTM, MultiHeadAttention, Conv2D, LayerNorm, FusionLayers
- PPO agent with GAE, three-speed optimiser, directional-symmetry penalty
- FinBERT (frozen) NLP sentiment stream fused with price features
- Fractional Kelly sizing, ATR stop-loss, black-swan detection, live `rich` telemetry dashboard

It worked as a proof of concept but was unscalable as a training platform. Its full documentation lives in [`Version_1/README.md`](Version_1/README.md).

## Version 2 — The production rewrite (current)

A clean rewrite on PyTorch, built around one idea: **one action pipeline everywhere**. Training, backtesting, and live trading run the exact same sequence — policy → Kelly sizing → risk caps → kill switch → execution — so backtest results describe what the live system actually does.

```
Alpaca 5-min OHLCV bars (99 US tickers)
        ↓
fetch / preprocess → parquet feature store (returns, vol z-scores, windows)
        ↓
vectorized multi-env tensors  [n_envs × window × features]
        ↓
hybrid PPO: CNN + LSTM encoders → cross-attention fusion → dual critic
        ↓
action: direction, size, limit offset   (deterministic in live)
        ↓
risk stack: KellySizer → RiskManager (position/exposure caps) → KillSwitch
        ↓
Alpaca broker — paper by default, live only with explicit opt-in
        ↓
reconciliation every cycle + metrics/dashboard
```

Key design decisions:

- **Vectorized multi-env training** — ~99 tickers in parallel (PPO + DDP for multi-GPU), far more sample-efficient than single-symbol agents
- **Deterministic live inference** — PPO's stochastic exploration never runs with capital, paper or real
- **Paper-first, explicit live opt-in** — separate paper/live credential env vars, so a misconfiguration can't silently point a "paper" run at real money
- **Reconciliation before every order** — internal ledger checked against the broker; a mismatch or unreachable broker blocks trading that cycle
- **Kill switch that never auto-resets** — daily loss limit, broker-error streak, state mismatch: resuming is a human decision
- **Same risk sequence in train/backtest/live** — no backtest-only logic that vanishes at deployment

Layout: `env/` (vectorized trading env, portfolio ledger, execution sim) · `model/` (hybrid policy) · `training/` (PPO, rewards, config) · `risk/` (kelly, risk manager, kill switch) · `live/` (loop, broker client, reconciliation) · `eval/` (metrics, backtest reports) · `monitoring/` (dashboard) · `data/` (processed features).

## Status

- Version 2 training in progress (Kaggle); validation/backtest results to be posted
- No live capital — paper trading only until backtest and paper results justify otherwise

## Roadmap

1. **Backtest validation** — cost-aware walk-forward backtests, honest metrics (Sharpe, max drawdown, vs buy-and-hold)
2. **Paper trading** — extended paper run, monitoring fill quality vs backtest assumptions
3. **Live trading** — small real size, hard daily loss limits, full trade logging

## Setup

```bash
cd Version_2
pip install -r requirements.txt
# set TRADING_ALPACA_PAPER_KEY / TRADING_ALPACA_PAPER_SECRET (paper) or
# TRADING_ALPACA_LIVE_KEY / TRADING_ALPACA_LIVE_SECRET (live)
```

See [`Version_2/README.md`](Version_2/README.md) for the module-by-module breakdown.
