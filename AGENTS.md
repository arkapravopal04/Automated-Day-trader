# AGENTS.md — Automated-Day-trader / Version_2

Project-specific context. Read this instead of re-discovering the codebase each fresh session.

## Status
As of the last full sweep, all ~29 files across paths/fetch/preprocess/dataset/diagnostics/env/model/risk/training/eval/monitoring/live/entrypoints layers were individually tested and cross-integration-tested. No known open bugs or gaps in plumbing. Current work is tuning/training for profitability, not structural rewrites — treat "make it profitable" as a hyperparameter/reward-shaping/backtest-iteration problem, not a rewrite problem, unless a specific new bug is found.

## Architecture (high level)
- Env: `env/portfolio_state.py`, `env/execution_sim.py`, `env/vec_trading_env.py` — vectorized multi-asset, one rollout stream per ticker, hybrid action space (direction/size/limit_offset). Direction contract `{-1,0,1}` is hard-enforced in execution_sim.
- Risk: `risk/risk_manager.py`, `risk/kelly_sizing.py`, `risk/kill_switch.py` — pipeline order: policy action → Kelly → RiskManager → KillSwitch → env.step().
- Model: CNNEncoder → LSTMEncoder → CrossAssetAttention → FusionTrunk → HybridActorCritic (dual critic: long/short/flat heads).
- Training: `training/reward.py` (Differential Sharpe), `training/config.py` (nested dataclasses, friction presets), `training/ppo_hybrid.py` (rollout/GAE/PPO update).
- Live: `live/broker_client.py` (Alpaca, paper/live credential isolation, live never falls back to paper creds), `live/reconciliation.py`, `live/live_loop.py`.
- Entrypoints: `train.py`, `main.py` (train/backtest/live/monitor subcommands).

## Known weak spots (last flagged)
- UNG and NFLX were the weakest per-ticker performers in training diagnostics.
- G2/G3 gate thresholds were calibrated for low-friction mode — recalibration needed under realistic/high-friction settings before trusting those gates fully.
- Expanded 100-ticker universe dropped the SH/SQQQ/PSQ inverse ETFs added earlier for directional-bias mitigation — not re-added, flagged only.

## Environment
- Kaggle: repo path `/kaggle/working/Automated-Day-trader/Version_2`, T4 GPU, checkpoints to `/kaggle/working/`.
- `requirements.txt` deliberately excludes torch/pandas/pyarrow/matplotlib (Kaggle preinstalls, avoid ABI mismatch) — installs with `--no-deps`.
- `.env.example` documents all real env vars grepped from the codebase (Alpaca data keys vs. separate live/paper trading keys — live never reads paper creds).

## Working agreement for this project
- Training runs (Kaggle or local) → background them, don't babysit interactively (see SOUL.md Budget & Session Hygiene).
- One tuning hypothesis per session (e.g. reward shaping OR position sizing OR curriculum mix — not all three in one sprawling session).
- Before claiming a change improved profitability, run it through `eval/backtest_report.py`'s deterministic backtest — don't eyeball training curves alone.