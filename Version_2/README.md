# Version 2 — Production Rewrite

The active codebase. A vectorized, risk-hardened rewrite of the original
from-scratch prototype (see the [root README](../README.md) for the two-version
story). Built around one rule: **the same action pipeline runs in training,
backtesting, and live trading** — policy → Kelly sizing → risk caps → kill
switch → execution — so backtest results describe what the live system
actually does.

- **Data**: 5-minute OHLCV bars, 99 US tickers (broad ETFs + mega-cap tech +
  financials + healthcare + consumer + energy/industrials)
- **Model**: hybrid PPO — CNN + LSTM encoders, cross-attention fusion, dual critic
- **Training**: vectorized multi-env (99 parallel streams), single-GPU
  (`train.py`) or dual-GPU DDP (`train_ddp.py`, Kaggle 2×T4)
- **Risk**: Kelly sizing → hard position/exposure caps → kill switch, in that
  order, everywhere
- **Live**: Alpaca paper by default; live requires explicit opt-in and separate
  live credentials

## Usage

Single entrypoint (or call the module scripts directly — both work):

```bash
python main.py train [--kaggle | --local] [--total-rollouts N] [--resume PATH | --fresh]
python main.py backtest --checkpoint PATH [--use-raw-policy]
python main.py live --checkpoint PATH --feature-builder module.path:function_name [--live]
python main.py monitor [--metrics-path PATH] [--kaggle | --local]
```

Notes:

- `--resume PATH` accepts a checkpoint path, `latest` (highest `checkpoint_N.pt`),
  or `best` (`checkpoint_best.pt`); `--fresh` deletes existing checkpoints for a
  cold start. If checkpoints exist and **neither** flag is given, training
  refuses to start rather than guess.
- `--kaggle` / `--local` force mode-specific paths and checkpointing; the
  monitoring dashboard reads the same flag from `sys.argv`, so passing it once
  on any subcommand is enough.
- `--total-rollouts` is an END count (target), not a budget — training runs to
  it, logging/checkpointing along the way.
- Live mode requires a `--feature-builder` callable that reproduces the offline
  preprocessing features exactly (see the warning in `live/live_loop.py` —
  train/serve skew is the biggest live risk in this project).
- Dual-GPU: `python train_ddp.py --resume latest [--total-rollouts N]`
  (checkpoint dir / metrics path / logging cadence all overridable).

## Module Map

### Entry points
| File | Purpose |
|---|---|
| `main.py` | Thin single entrypoint for train / backtest / live / monitor |
| `train.py` | Single-GPU training (mode-aware paths, refuse-to-guess resume) |
| `train_ddp.py` | Dual-GPU DDP training (Kaggle 2×T4) |
| `hyperparam_sweep.py` | Hyperparameter sweeps |
| `data_digonastics.py` / `env_digonastics.py` / `diagnostics_gpu_and_learning.py` | Data / env / training diagnostics |

### data/
| File | Purpose |
|---|---|
| `fetch_alpaca.py` | Pulls 5-min bars from Alpaca, paginated, caches to parquet by ticker |
| `preprocess.py` | Feature engineering (log returns, vol z-scores, session encoding) → windowed tensors `[n_envs, window, features]`; date-range splits to avoid temporal leakage; NaN/gap sanity checks |
| `dataset.py` | Per-ticker feature tensors + metadata (feature names, normalization stats computed on train split only) |
| `paths.py` | Mode-aware path resolution (kaggle vs local) |
| `data/processed/` | Cached parquet feature store (~100 ticker files + metadata.json) |

### env/
| File | Purpose |
|---|---|
| `vec_trading_env.py` | Vectorized multi-env trading simulator (99 parallel streams) |
| `portfolio_state.py` | Per-stream ledger: cash, signed positions, equity, peak/drawdown tracking |
| `execution_sim.py` | Fill simulation (limit-order fills, partial fills) |

### model/
| File | Purpose |
|---|---|
| `hybrid_policy.py` | Actor-critic head: direction / size / limit-offset actions + rescaling; deterministic mode for live |
| `cnn_encoder.py` / `lstm_encoder.py` | Bar-pattern and temporal-sequence encoders |
| `cross_attention.py` | Cross-attention fusion of encoder streams |
| `dual_critic.py` | Dual value heads |
| `fusion.py` | Stream projection/concat/fusion |

### training/
| File | Purpose |
|---|---|
| `ppo_hybrid.py` | PPO training loop + `HybridActorCritic` |
| `reward.py` | Reward shaping (trade PnL, vol-normalized returns, penalties) |
| `config.py` | `TrainingConfig`: env / model / risk / run settings, ticker universe, friction presets |

### risk/  (the hard limits between the policy and the broker)
| File | Purpose |
|---|---|
| `kelly_sizing.py` | Edge-aware position sizing (Kelly fraction, lookback, cap, default) |
| `risk_manager.py` | Hard caps: per-order notional, per-ticker position, gross exposure, concentration, drawdown halt — reducing exposure is always allowed |
| `kill_switch.py` | Binary halt: daily loss limit, broker-error streak, state mismatch, manual trip. Never auto-resets — resuming is a human decision |

### live/
| File | Purpose |
|---|---|
| `broker_client.py` | Broker abstraction (Alpaca). Paper/live credentials from **different** env vars (`TRADING_ALPACA_PAPER_*` vs `TRADING_ALPACA_LIVE_*`) so a misconfigured "paper" run can't silently hit real capital |
| `live_loop.py` | Bar poll → deterministic inference → risk pipeline → order submission, with reconciliation before every order |
| `reconciliation.py` | Syncs the internal ledger against broker-reported positions each cycle; mismatch or unreachable broker blocks trading and feeds the kill switch |

### eval/ and monitoring/
| File | Purpose |
|---|---|
| `eval/metrics.py` / `eval/backtest_report.py` | Backtest metrics and report generation |
| `monitoring/dashboard.py` | Live dashboard + structured metrics writer |

## Risk pipeline (the part that makes it deployable)

Every decision that reaches a broker passes through, in order:

1. `KellySizer.apply()` — soft, edge-based size cap
2. `RiskManager.apply()` — hard position/exposure/concentration/drawdown caps
   (only ever restricts exposure-INCREASING orders)
3. `KillSwitch.apply()` — zeroes orders for any halted stream

In live, `Reconciler.sync()` runs before orders each cycle: an unreachable
broker or a ledger/broker mismatch blocks trading regardless of what the rest
of the pipeline decided. The kill switch halts on daily loss, consecutive
broker errors, or state mismatch — and only a human can reset it.

## Credentials

```bash
# paper
TRADING_ALPACA_PAPER_KEY=...
TRADING_ALPACA_PAPER_SECRET=...

# live (required only for explicit --live)
TRADING_ALPACA_LIVE_KEY=...
TRADING_ALPACA_LIVE_SECRET=...
```

There is no fallback between the two: live mode refuses to construct without
the live vars, paper mode refuses without the paper vars.

## Status

- Training in progress (Kaggle); validation/backtest results to be posted here
- Paper trading only until backtest and paper results justify otherwise
