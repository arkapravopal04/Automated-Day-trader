# Version 1 — Archived

> The original build. No longer developed — the active system is [Version 2](../README.md).
> Kept because how it was built is the interesting part.

---

# Automated Day Trader (V1)

A trading agent built from absolute scratch. No PyTorch. No TensorFlow. A hand-written autograd engine, hand-written LSTMs, hand-written attention, and a PPO agent trained on top of all of it.

The goal was never to ship fast. It was to be unable to hide behind a library — to know exactly what happens in the backward pass of a matmul, and exactly how an advantage estimate is computed inside PPO.

> *Paper trade before real money. Always. Personal project, use at your own risk.*

---

## What it does

Reads 5-minute price bars and financial news headlines. Fuses both into one picture of the market. Decides what regime it's in, sizes a position accordingly, and trades it — with a risk layer that can override the model whenever it wants to do something reckless.

```
[News headlines]              [5-min price bars]
       ↓                              ↓
  FinBERT (frozen)      Conv2D          LSTM ×2
  sentiment vector      patterns        temporal
       ↓                   ↓               ↓
       └───────────────────┴───────────────┘
                           ↓
                     Fusion layer
              cross-attention across streams
                           ↓
                    Regime detector
                 bull / bear / sideways
                           ↓
                     Risk manager
          Kelly sizing · ATR stops · black-swan block
                           ↓
                   PPO agent (75-dim state)
            actor → direction + size · critic → value
                           ↓
                   Buy / Sell / Hold → Alpaca
```

~163K trainable parameters. FinBERT's 110M are frozen — only the projection into the fusion layer learns.

## Everything here was written by hand

**The autograd engine** — a `Tensor` class with topological backprop. matmul, conv via im2col, softmax, slicing, broadcasting with correct gradient reduction. Every gradient in the system flows through code in this repo.

**The layers** — LSTM with forget-gate bias initialized to 1 and truncated BPTT every 16 steps. Multi-head attention with a numerically stable clamped softmax. LayerNorm, Dropout, Conv2D, He-initialized Linear. AdamW with decoupled weight decay, and SGD with momentum.

**The agent** — PPO with a clipped surrogate objective, GAE at λ=0.95, Welford online return normalization, and three separate optimizers running at different learning rates so the heads can learn fast without the feature extractors overwriting themselves.

**The risk layer** — fractional Kelly with long and short trade histories kept separate, because a bull-market dataset will otherwise quietly inflate your confidence in shorts. ATR stop-loss with a 1.5% noise floor so microstructure jitter doesn't stop you out. A 30% drawdown halt. A 5σ black-swan detector that blocks new entries.

**The dashboard** — live `rich` telemetry: net worth, win rate, drawdown, per-optimizer gradient norms, and a directional-bias readout that turns red the moment the policy starts collapsing into permanent long or permanent short.

## What it taught me

- **Truncated BPTT every 16 steps** stopped extractor gradients exploding across 48-step windows.
- **Three-speed optimizers** stopped a fast-learning head from destroying slow-learned features.
- **Separate long/short Kelly** — the single most important fix against bullish-data bias.
- **A directional symmetry penalty** was necessary; without it, the policy collapses to one direction and stops being a trader.
- **PPO over REINFORCE** — GAE's variance reduction is not optional when the reward signal is this noisy.

## Why it was replaced

It worked. It proved every concept. It was also unscalable as a training platform — single-symbol, CPU-bound, and slow enough that iteration hurt.

Version 2 kept the ideas and rebuilt the machinery: PyTorch, 100 tickers trained in parallel, and one action pipeline shared by training, backtest, and live.

## Running it

```bash
pip install numpy pandas requests transformers torch rich alpaca-trade-api

python alpaca_prefetch_data.py   # cache all tickers (~3-5 min first run)
python alpaca_train.py           # train
```

Alpaca key/secret go in `alpaca_data.py`.

## Files

```
engine.py          autograd Tensor engine
Neural_Nets.py     layers, optimisers, LSTM, attention, fusion
module.py          base Module class
alpaca_data.py     data pipeline + incremental caching
nlp.py             FinBERT encoder + NewsAPI
environment.py     trading environment
agent.py           PPO agent
risk.py            risk manager
alpaca_train.py    training loop
telemetry.py       live dashboard
```