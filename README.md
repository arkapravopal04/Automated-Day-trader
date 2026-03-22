# Automated Day Trader

A full autonomous trading system built from scratch including a custom autograd engine, deep learning components, NLP sentiment analysis, and a reinforcement learning agent.

---

## Project Philosophy

Every component in this system is built from the ground up to maximise understanding. No PyTorch, no TensorFlow just numpy, pure Python, and first principles.

---

## Architecture Overview

```
[News / Tweets / Reddit]          [Raw Price / Volume Data]
         ↓                                    ↓
   [NLP Encoder]              [CNN Encoder] + [LSTM Encoder]
   FinBERT + Linear           Pattern        Temporal
   sentiment vector           detection      momentum
         ↓                         ↓              ↓
         └──────────────┬───────────┘              │
                        ↓                          │
              [Fusion Layer]  ←────────────────────┘
              Cross Attention across all signals
                        ↓
              [Regime Detector]
              Bull / Bear / Sideways
                        ↓
                  [RL Agent - PPO]
                  Continuous actions
                  direction + position size
                        ↓
                 Buy / Sell / Hold
                        ↓
              [Risk Management Layer]
              Position sizing, stop loss,
              drawdown limits
                        ↓
              [Backtester / Broker API]
```

---

## Current Status

### Completed

**Step 1 — Custom Deep Learning Engine**
- `engine.py` — Full autograd Tensor class with backpropagation
- Operations: matmul, add, mul, div, pow, exp, log, sum, mean, concat, slice, getitem, im2col, reshape, transpose, flatten
- Activations: ReLU, Tanh, Sigmoid
- `module.py` — Base Module class
- `Neural_Nets.py` — Linear, Conv2D, Flatten, Sequential, LayerNorm, Dropout
- Optimizers: SGD with momentum, Adam
- Loss: CrossEntropyLoss (with correct backward pass)
- `LSTMCell` — 8 weights, Xavier init, forget gate bias = ones
- `LSTM` — stacked layers, full sequence processing, returns all hidden states
- `Attention` — full QKV attention with scaled dot product and stable softmax

**Step 2 — Data Pipeline**
- `data.py` — yfinance integration
- `load_data` — fetch and clean OHLCV data
- `transform_data` — percentage returns transformation
- `build_windows` — sliding window sequences (2253, 10, 5)
- `generate_regime_labels` — bull/bear/sideways labels from price data
- `DataLoader` — batching and shuffling
- CNN + LSTM tested on real AAPL price data
- Temporal attention over LSTM timesteps verified
- Full backward pass with gradient flow confirmed

**Step 3 — NLP Pipeline**
- `nlp.py` — NLPEncoder class using FinBERT
- FinBERT frozen, Linear(3, 64) trainable projection
- NewsAPI integration — real financial headlines
- `fetch_news` — pulls headlines for any ticker
- `get_sentiment_vector` — averages sentiment across multiple headlines
- Verified: bankruptcy → negative, earnings beat → positive/neutral

**Step 4 — Fusion Layer**
- `FusionLayers` — projects CNN, LSTM, NLP, Regime to same size
- Cross attention across all four signals
- Softmax on regime logits before fusion
- `RegimeDetector` — LSTM + Attention + Linear(hidden, 3)
- Regime labels: Bear 31%, Sideways 14%, Bull 55% (AAPL 2015-2024)
- Full pipeline tested: LSTM + CNN + NLP + Regime → Fused (64,) → Loss → Backward ✓

**Step 5 — RL Environment (in progress)**
- `env.py` — TradingEnvironment complete
- Continuous action space: [direction, size]
- Long and short selling supported
- State vector: fused (64,) + portfolio (3,) = (67,)
- `reset()`, `step()`, `_get_state()` implemented

### Currently Working On

**Week 5 — PPO Agent (`agent.py`)**
- Actor network: state (67,) → action (2,) [direction, size]
- Critic network: state (67,) → value (1,)
- PPO clipping for stable policy updates
- Continuous action space with exploration noise

---

## Problems Faced & How We Fixed Them

### 1. CrossEntropyLoss — Missing Backward Pass
**Problem:** Gradient stopped at the target indexing step. Weights never updated despite no errors or crashes.

**Root cause:**
```python
target_tensor = Tensor(target_prob, (probs,), 'target_select')
# _backward was never defined — defaulted to lambda: None
```

**Fix:** Manually defined `_backward` to route gradient back through the indexing operation:
```python
def _backward():
    grad = np.zeros_like(probs.data)
    grad[target_idx] = target_tensor.grad
    probs.grad += grad
target_tensor._backward = _backward
```

---

### 2. CrossEntropyLoss — Wrong Axis
**Problem:** `np.max(logits.data, axis=1)` crashed because logits were 1D `(2,)` not 2D `(batch, 2)`.

**Fix:** Changed `axis=1` to `axis=0` for single sample inference.

---

### 3. Matmul Backward — Shape Mismatch
**Problem:** Backward pass through Linear layer crashed with shape mismatch when inputs were 1D vectors instead of 2D matrices.

**Root cause:** `.T` on a 1D numpy array does nothing — shapes didn't align for gradient computation.

**Fix:** Force all inputs to 2D before gradient computation, reshape back after:
```python
out_grad   = out.grad.reshape(1, -1)
self_data  = self.data.reshape(1, -1)
other_data = other.data if other.data.ndim > 1 else other.data.reshape(-1, 1)
self.grad  += (out_grad @ other_data.T).reshape(self.data.shape)
other.grad += (self_data.T @ out_grad).reshape(other.data.shape)
```

---

### 4. Concat Backward — Wrong Axis Handling
**Problem:** Gradient slicing in concat backward was hardcoded for axis=0, broke for other axes.

**Fix:** Used `np.split` which respects the axis parameter:
```python
grads = np.split(out.grad, [self.data.shape[axis]], axis=axis)
self.grad  += grads[0]
other.grad += grads[1]
```

---

### 5. FinBERT Corrupted Cache
**Problem:** Accidentally loaded FinBERT with `num_labels=2` (should be 3). This corrupted the cached model. Subsequent loads used the corrupted cache — bankruptcy was classified as positive.

**Fix:** Deleted the HuggingFace cache folder at:
```
C:\Users\username\.cache\huggingface\hub\models--ProsusAI--finbert
```
Redownloaded fresh. Now correctly classifies:
- Bankruptcy → negative ✓
- Earnings beat → neutral/positive ✓
- Stable markets → neutral ✓

---

### 6. Dropout — Wrong Implementation
**Problem:** Original dropout zeroed the entire tensor based on one random number instead of independently dropping each neuron.

**Fix:** Generate a per-element binary mask using numpy broadcasting:
```python
mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
mask_tensor = Tensor(mask / (1 - self.p))
return x * mask_tensor
```

---

### 7. LayerNorm — Outside Computation Graph
**Problem:** `np.sqrt(var + self.eps)` stepped outside the autograd graph — gradients couldn't flow through normalisation.

**Fix:** Used engine's `**` operator instead:
```python
x_normalized = (x - mean) / ((var + self.eps) ** 0.5)
```

---

### 8. LSTM — Vanishing Gradients
**Problem:** LSTM gradients are nearly zero (`~0.000000`) by the time they reach early layers.

**Status:** Known issue, expected at this stage. Will improve with:
- LayerNorm between components in training loop
- Attention mechanism (partially implemented)
- More training signal from batch training

---

### 9. Attention — Stacking Tensors
**Problem:** `np.stack(hidden_states)` crashed because hidden states were Tensor objects not numpy arrays.

**Fix:**
```python
h_stack = Tensor(np.stack([h.data for h in hidden_states], axis=0))
```

---

## File Structure

```
automated_daytrader/
├── engine.py          ← autograd Tensor engine
├── module.py          ← base Module class
├── Neural_Nets.py     ← all layers, optimizers, LSTM, Attention, Fusion
├── losses.py          ← CrossEntropyLoss
├── data.py            ← data pipeline, DataLoader
├── nlp.py             ← NLPEncoder, NewsAPI integration
├── env.py             ← TradingEnvironment
├── agent.py           ← PPO Agent (in progress)
└── tester.py          ← integration tests(not added as its a tester file...will be added when done)
```

---

## V2 Feature Backlog

```
Risk Management:
  - Real price tracking (fix percentage return issue)
  - Kelly Criterion position sizing
  - Max daily drawdown hard limit
  - Volatility-adjusted trade sizing
  - Emergency exit logic

Architecture:
  - Multi-head attention upgrade
  - Multi-timeframe inputs (1min, 1hr, 1day)
  - Uncertainty estimation per signal
  - Order book data
  - Options flow signal
  - Macro indicators

NLP:
  - Reddit PRAW integration
  - MultiSourceNLP (competitors, supply chain, macro)
  - Twitter/X integration

Regime:
  - Specialist models per regime
  - Regime-aware reward function
  - Attention visualisation

Feedback:
  - Trade journal
  - Online retraining pipeline
  - Performance degradation detector
  - A/B testing framework
  - Anomaly detection
```

---

## Setup

```bash
pip install numpy yfinance pandas transformers torch requests
```

---

## Key Design Decisions

- **Custom autograd engine** — built from scratch for deep understanding
- **FinBERT frozen** — pretrained on financial text, only projection layer trains
- **Continuous action space** — direction + size enables nuanced position management
- **Short selling supported** — agent can profit in both bull and bear markets
- **PPO over REINFORCE** — more stable for noisy financial environments
- **Regime detection** — market context fed into fusion layer via softmax probabilities

---

*Paper trade before real money. Always.*

Also I forgot to write this in, this is a personal project and use it at your own risk 
verify any actions taken by the bot
do not risk money that you are affraid to loose😭😭😭