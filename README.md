# Automated Day Trader

A full autonomous trading system built from scratch — custom autograd engine, deep learning components, NLP sentiment analysis, and a reinforcement learning agent trained via PPO. No PyTorch. No TensorFlow. Just NumPy and first principles.

> *Paper trade before real money. Always.*
> *This is a personal project — use at your own risk. Do not risk money you cannot afford to lose.* 😭

---

## Project Philosophy

Every component is built from the ground up to maximise understanding. The goal was never to wrap a library — it was to know exactly what is happening at every layer, from the backward pass of a matmul to the GAE advantage estimate inside PPO.

---

## Architecture Overview

```
[News Headlines via NewsAPI]      [Raw OHLCV — Alpaca 5m bars]
          ↓                                    ↓
   [NLP Encoder]            [Conv2D]      [LSTM × 2 layers]
   FinBERT (frozen)          Pattern       Temporal sequence
   + Linear(3 → 64)          detection     hidden: 64
   sentiment vector               ↓              ↓
          ↓                       └──────────────┘
          ↓                              ↓
          └──────────────────────────────┘
                                   ↓
                          [Fusion Layer]
                    Project + concat all streams
                    Cross-attention across signals
                    Output: 64-dim fused state
                                   ↓
                        [Regime Detector]
                    LSTM + Attention + Linear
                    Bull / Bear / Sideways
                                   ↓
                        [Risk Manager]
                    Kelly sizing, ATR stop loss
                    Black swan detection
                    8 risk features → fused state
                                   ↓
                        [PPO Agent — 75-dim state]
                    Actor:  state → [direction, size]
                    Critic: state → value estimate
                    GAE advantage, clipped surrogate
                                   ↓
                         Buy / Sell / Hold
                                   ↓
                    [Paper Broker / Live API]
                    Alpaca — US equities, 5m intraday
```

---

## Parameter Count

| Component | Trainable params |
|---|---|
| LSTM (2-layer, hidden=64) | 50,944 |
| Attention (LSTM) | 12,288 |
| RegimeDetector (LSTM + Attn + Linear) | 8,035 |
| Conv2D (1→16, kernel 3×5) | 256 |
| FusionLayers (5 projections + MHA + LN) | 77,504 |
| NLPEncoder Linear (3→64) | 256 |
| Actor head (75→64→32→2) | 7,202 |
| Critic head (75→64→32→1) | 7,169 |
| **Total trainable** | **~163,654** |

FinBERT is frozen (~110M params, not trained).

---

## Current Status

### Completed

**Custom Deep Learning Engine (`engine.py`)**
- Full autograd `Tensor` class with topological backpropagation
- Operations: matmul, add, mul, div, pow, exp, log, sum, mean, max, softmax, concat, slice, stack, getitem, im2col, reshape, transpose, flatten
- Activations: ReLU, Tanh, Sigmoid (numerically stable split formula)
- All ops support arbitrary broadcast with correct gradient reduction

**Neural Network Modules (`Neural_Nets.py`, `module.py`)**
- `Linear` — He initialisation, configurable weight scale
- `Conv2D` — im2col-based, fan-in init
- `LSTM` / `LSTMCell` — stacked layers, forget gate bias = 1, truncated BPTT every 16 steps
- `MultiHeadAttention` — scaled dot-product, stable softmax clamped to [-20, 0]
- `LayerNorm`, `Dropout`
- Optimisers: SGD with momentum, AdamW (decoupled weight decay)
- `FusionLayers` — five stream projections + cross-attention + output projection
- `RegimeDetector` — LSTM + Attention + Linear → regime embedding

**Data Pipeline (`alpaca_data.py`, `alpaca_prefetch_data.py`)**
- Alpaca REST API — up to 6 years of 5m OHLCV bars
- 10 tickers: SPY, QQQ, IWM, XLE, XBI, GLD, USO, ARKK, AAPL, NVDA
- Incremental caching to `./alpaca_cache/` — only fetches the gap since last pull
- `transform_data` — log-returns, percentage change features
- `build_windows` — sliding 48-bar windows with labels
- `prefetch_data.py` — run once before overnight training, no API calls during train

**NLP Pipeline (`nlp.py`)**
- FinBERT (frozen) + trainable `Linear(3 → 64)` projection
- NewsAPI integration for live financial headlines
- Multi-headline sentiment averaging
- Falls back to zero vector if no headlines available

**RL Agent (`agent.py`)**
- PPO with clipped surrogate objective (ε = 0.2)
- Gaussian policy over continuous [direction, size] action space
- Generalised Advantage Estimation (GAE, λ = 0.95)
- Welford online return normalisation
- 8 PPO epochs per episode, mini-batches of 64
- Three separate optimisers at different LRs:
  - Head (actor + critic): 3e-4
  - Extractor (LSTM, CNN, attention, regime): 7e-6
  - Fusion: 7e-6
- Exploration std decay with rebound floor (min 0.2)
- Directional symmetry penalty to prevent long/short bias collapse

**Trading Environment (`environment.py`)**
- 750-step episodic environment, random start slice per episode
- Composite reward: trade PnL + vol-normalised step return + hold-loser penalty + stress penalty + milestone bonuses + terminal alpha vs buy-and-hold benchmark
- All rewards clipped to [-8, 8]
- Bankruptcy trigger at 80% drawdown from starting balance

**Risk Management (`risk.py`)**
- Fractional Kelly criterion — separate long/short trade history to prevent bullish data bias
- ATR-based dynamic stop loss: `max(1.5%, 2.5 × ATR)` — noise floor prevents microstructure stops
- Max drawdown hard limit (30%) — halts all trading for episode
- Black swan detection via Z-score (threshold: 5σ) — blocks new entries
- 8 risk features fed directly into fusion layer
- Kelly halved in high-volatility regimes

**Persistence (`models_utils.py`)**
- Pickle-based weight serialisation, keyed by layer name
- NaN/inf guard on all parameters before every save
- Separate best model and rolling checkpoint

**Telemetry (`telemetry.py`)**
- Live `rich` dashboard: net worth, position, win rate, drawdown, step reward breakdown, gradient norms per optimiser group
- `dir_mean` colour-coded: green < ±0.3, yellow < ±0.8, red ≥ ±0.8 — immediate visual warning of policy directional collapse
- Scrolling episode history table with per-episode avg directional bias

---

## File Structure

```
automated_daytrader/
├── engine.py                 ← autograd Tensor engine
├── module.py                 ← base Module class
├── Neural_Nets.py            ← all layers, optimisers, LSTM, Attention, Fusion
├── alpaca_data.py            ← Alpaca data pipeline + caching
├── alpaca_prefetch_data.py   ← run once to pre-cache all tickers
├── nlp.py                    ← NLPEncoder, NewsAPI integration
├── environment.py            ← TradingEnvironment (training)
├── agent.py                  ← PPO Agent
├── risk.py                   ← RiskManager
├── losses.py                 ← CrossEntropyLoss, MSELoss
├── alpaca_train.py           ← main training loop
├── models_utils.py           ← save / load / log
└── telemetry.py              ← live training dashboard
```

---

## Setup

```bash
pip install numpy pandas requests transformers torch rich alpaca-trade-api
```

Add your Alpaca API key and secret to `alpaca_data.py`:
```python
API_KEY    = "your_key"
API_SECRET = "your_secret"
BASE_URL   = "https://paper-api.alpaca.markets"
```

Run order:
```bash
python alpaca_prefetch_data.py   # fetch + cache all tickers (~3-5 min first time)
python alpaca_train.py           # train overnight
```

---

## Deployment Roadmap

The system is being hardened for live deployment across four phases. Each phase has defined gate tests — **do not move to the next phase until all tests pass**.

### Phase 1 — Realistic training environment
Add transaction costs (spread, commission, market impact) and a hard train/test date split.

Gate tests:
- Sharpe > 0.5 on held-out data
- Max drawdown < 20% on test set
- Beats SPY buy-and-hold on the same period

### Phase 2 — Walk-forward backtest
Roll forward: train on 1 year, test on 3 months, repeat across 2018–2024.

Gate tests:
- Positive performance on 3+ test windows
- No single window blows up
- Win rate > 52% out-of-sample

### Phase 3 — Paper trading (minimum 30 days)
Continuous execution via Alpaca paper API. Persistent LSTM state across bars. Market-hours-aware loop. Hard kill switch.

Gate tests:
- 30 trading days without infrastructure crash
- Live P&L tracks backtest range
- Execution latency under one 5m bar
- No runaway positions

### Phase 4 — Live trading (start at $100)
Real capital. Hard daily loss limit. Weekly retraining cadence. Full order and fill logging.

Gate tests:
- Live Sharpe tracks paper trading Sharpe
- No single day exceeds 3% loss
- Can explain every trade the system made
- Consistent performance across 3 months

---

## Key Design Decisions

- **Custom autograd engine** — no framework dependency, complete understanding of every gradient
- **Truncated BPTT every 16 steps** — prevents extractor gradient explosion over 48-step windows
- **Three-speed optimiser** — head trains fast, extractors train slow, prevents feature overwriting
- **FinBERT frozen** — pretrained on financial text, only the 3→64 projection layer trains
- **Continuous action space** — direction + size enables nuanced position sizing, not just buy/sell/hold
- **PPO over REINFORCE** — variance reduction via GAE critical for noisy financial reward signals
- **Fractional Kelly, long/short separate** — prevents bullish data bias inflating short Kelly estimates
- **ATR stop-loss noise floor** — 1.5% minimum prevents microstructure noise triggering stops
- **Directional symmetry penalty** — discourages the policy from collapsing to permanent long or short bias

---

## V2 Backlog

```
Architecture:
  - Multi-timeframe inputs (1m, 1h, 1d)
  - Order book / level 2 data
  - Options flow signal
  - Uncertainty estimation per signal stream

NLP:
  - Reddit / Twitter / earnings call integration
  - MultiSourceNLP (sector peers, macro indicators)
  - Fine-tune sentiment layer on actual price reactions

Regime:
  - Specialist sub-models per detected regime
  - Attention weight visualisation

Validation & Operations:
  - Online retraining pipeline
  - Performance degradation detector
  - Walk-forward automation script
  - A/B testing framework for architecture changes
```