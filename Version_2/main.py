"""
main.py

Optional single entrypoint for train/backtest/live/monitor modes, instead of
juggling four separate scripts. Each subcommand is a thin wrapper around the
module that actually does the work -- this file owns argument parsing and
wiring, nothing else. If you'd rather call train.py or a notebook cell
directly, that keeps working unchanged; this is purely a convenience layer.

Usage:
    python main.py train [--kaggle | --local] [--total-rollouts N] [--resume PATH]
    python main.py backtest --checkpoint PATH [--use-raw-policy]
    python main.py live --checkpoint PATH --feature-builder module.path:function_name [--live]
    python main.py monitor [--metrics-path PATH] [--kaggle | --local]

`--kaggle` / `--local` on any subcommand are also what
monitoring/dashboard.py's resolve_mode() looks for in sys.argv -- passing
one here is enough, nothing needs to re-parse it elsewhere.
"""

import argparse
import importlib
import os
import sys

import torch

# See train.py's matching comment: force line-buffered stdout so print()
# calls show up immediately in a Kaggle/Jupyter `!python main.py ...` cell
# instead of sitting in CPython's default full-buffering for a piped stdout
# until the buffer fills or the process exits (looks exactly like a hang).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# See train.py's module-level comment: vec_trading_env.py / portfolio_state.py
# live in env/, which needs to be on sys.path before anything imports them --
# done once, here, so it covers every subcommand (train delegates to train.py
# which does its own bootstrap too, but backtest/live import these directly).
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from training.config import TrainingConfig


def _load_actor_critic(checkpoint_path: str, cfg: TrainingConfig, n_features: int, device: torch.device):
    from training.ppo_hybrid import HybridActorCritic

    actor_critic = HybridActorCritic(n_features=n_features, cfg=cfg).to(device)
    # weights_only=True: checkpoints are pickle files -- loading one from an
    # untrusted source with arbitrary-object unpickling is RCE. Our own
    # checkpoints only ever contain tensors + plain Python types, which
    # weights_only loads fine.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()
    return actor_critic


def cmd_train(args: argparse.Namespace) -> None:
    # train.py's own argparse handles --kaggle/--local/--total-rollouts/--resume;
    # just re-dispatch into it with the leftover argv.
    import train

    sys.argv = ["train.py", *args.rest]
    train.main()


def cmd_backtest(args: argparse.Namespace) -> None:
    from dataset import MultiTickerRolloutDataset  # noqa: E402 -- see train.py's module docstring assumption note
    from eval.backtest_report import generate_report, run_backtest
    from env.vec_trading_env import VecTradingEnv

    cfg = TrainingConfig()
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    # See train.py's module docstring / the matching fix note there:
    # MultiTickerRolloutDataset doesn't take a `tickers` kwarg -- it
    # auto-discovers tickers from metadata.json.
    test_dataset = MultiTickerRolloutDataset(
        window_size=cfg.env.window_size, split="test", device=str(device)
    )
    # Every friction/behaviour param cfg.env carries, matching train.py's
    # VecTradingEnv construction term for term -- this was previously just
    # initial_cash + device, so the backtest silently ran on VecTradingEnv's
    # raw class defaults for everything else. Two of those defaults are not
    # cosmetic: impact_coef=0.1 is the un-recalibrated, ~7x-too-high value
    # Session 1 fixed to 0.015, and trade_cooldown_bars=0 drops the churn
    # constraint entirely. Either would make backtest results incomparable
    # to what the policy was actually trained under.
    #
    # enable_mirroring is force-FALSE regardless of cfg.env, not read from
    # it: mirroring fabricates a synthetic price path by reflecting log
    # prices (see vec_trading_env.py's docstring), which is a training-time
    # augmentation. Test data must be the real, unaltered market -- the ONE
    # evaluation this project treats as ground truth cannot run against a
    # coin-flip mix of real and synthetic streams. This is also why the
    # verdict must never be trusted from a run where this line regressed.
    env = VecTradingEnv(
        dataset=test_dataset,
        initial_cash=cfg.env.initial_cash,
        max_position_frac=cfg.env.max_position_frac,
        tick_size=cfg.env.tick_size,
        spread_bps=cfg.env.spread_bps,
        impact_coef=cfg.env.impact_coef,
        max_participation=cfg.env.max_participation,
        commission_per_share=cfg.env.commission_per_share,
        commission_bps=cfg.env.commission_bps,
        min_commission=cfg.env.min_commission,
        platform_fee_per_trade=cfg.env.platform_fee_per_trade,
        execution_price_column=cfg.env.execution_price_column,
        r_step_scale=cfg.env.r_step_scale,
        hold_loser_penalty=cfg.env.hold_loser_penalty,
        enable_mirroring=False,
        overtrade_window=cfg.env.overtrade_window,
        overtrade_free_trades=cfg.env.overtrade_free_trades,
        overtrade_penalty_coef=cfg.reward.overtrade_penalty_coef,
        bias_window=cfg.env.bias_window,
        diversity_bonus_coef=cfg.env.diversity_bonus_coef,
        trade_cooldown_bars=cfg.env.trade_cooldown_bars,
        min_hold_bars=cfg.env.min_hold_bars,
        trading_window=cfg.env.trading_window,
        flatten_at_session_close=cfg.env.flatten_at_session_close,
        device=str(device),
    )

    # env.feature_names, not test_dataset.feature_names -- VecTradingEnv appends
    # portfolio-state channels to the observation, so sizing off the dataset
    # builds a network too narrow and the checkpoint load fails on shape.
    actor_critic = _load_actor_critic(args.checkpoint, cfg, len(env.feature_names), device)

    output = run_backtest(
        env,
        actor_critic,
        cfg,
        use_risk_pipeline=not args.use_raw_policy,
        max_drawdown_limit=args.max_drawdown,
    )
    report_path = generate_report(output, env.tickers, args.out_dir)

    print(f"[backtest] verdict: {'GO' if output.go_no_go else 'NO-GO'}")
    for reason in output.go_no_go_reasons:
        print(f"[backtest]   - {reason}")
    print(f"[backtest] report written to {report_path}")


def cmd_live(args: argparse.Namespace) -> None:
    from live.broker_client import AlpacaBrokerClient
    from live.live_loop import AlpacaBarPoller, LiveLoop
    from monitoring.dashboard import MetricsWriter

    cfg = TrainingConfig()
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    paper = not args.live  # default is paper trading; --live is an explicit, unmissable opt-in

    module_path, func_name = args.feature_builder.split(":")
    feature_builder = getattr(importlib.import_module(module_path), func_name)

    broker = AlpacaBrokerClient(paper=paper)
    bar_poller = AlpacaBarPoller(tickers=cfg.env.tickers, window_size=cfg.env.window_size, paper=paper)

    # n_features has to come from wherever feature_builder's output width is
    # defined -- there's no dataset object in live mode to read it off of, so
    # this is an explicit CLI arg rather than an inferred value.
    actor_critic = _load_actor_critic(args.checkpoint, cfg, args.n_features, device)

    metrics_writer = MetricsWriter(cfg.run.metrics_path)

    loop = LiveLoop(
        broker=broker,
        bar_poller=bar_poller,
        feature_builder=feature_builder,
        actor_critic=actor_critic,
        cfg=cfg,
        tickers=cfg.env.tickers,
        metrics_writer=metrics_writer,
        bar_interval_seconds=args.bar_interval_seconds,
        auto_correct_mismatches=args.auto_correct_mismatches,
    )

    mode_str = "LIVE (real capital)" if not paper else "paper"
    print(f"[live] starting in {mode_str} mode against {len(cfg.env.tickers)} tickers")
    loop.run_forever()


def cmd_monitor(args: argparse.Namespace) -> None:
    from monitoring.dashboard import TrainingDashboard, resolve_mode

    cfg = TrainingConfig()
    metrics_path = args.metrics_path or cfg.run.metrics_path
    # Match train.py/train_ddp.py's Kaggle redirect so `monitor` finds the
    # metrics file where training actually writes it on Kaggle, not where
    # the local default ("logs/metrics.jsonl", CWD-relative) points. Only
    # auto-redirect when the user didn't pass an explicit --metrics-path.
    if args.metrics_path is None:
        from paths import is_kaggle

        if is_kaggle() and metrics_path == "logs/metrics.jsonl":
            metrics_path = "/kaggle/working/logs/metrics.jsonl"
    mode = resolve_mode(cfg.run.display_mode)

    dashboard = TrainingDashboard(metrics_path, mode=mode, history_window=args.history_window)
    print(f"[monitor] watching {metrics_path} in {mode.value} mode (Ctrl+C to stop)")
    dashboard.run_polling_loop(poll_interval_seconds=args.poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single entrypoint for train/backtest/live/monitor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run PPO training (delegates to train.py).")
    train_parser.add_argument("rest", nargs=argparse.REMAINDER, help="Forwarded to train.py's own argparse.")
    train_parser.set_defaults(func=cmd_train)

    backtest_parser = subparsers.add_parser("backtest", help="Run the held-out backtest and produce the go/no-go report.")
    backtest_parser.add_argument("--checkpoint", type=str, required=True)
    backtest_parser.add_argument("--out-dir", type=str, default="backtest_output")
    backtest_parser.add_argument("--max-drawdown", type=float, default=0.20)
    backtest_parser.add_argument(
        "--use-raw-policy", action="store_true",
        help="Bypass Kelly/RiskManager/KillSwitch -- debugging only, see eval/backtest_report.py.",
    )
    backtest_parser.set_defaults(func=cmd_backtest)

    live_parser = subparsers.add_parser("live", help="Run the real-time trading loop (paper by default).")
    live_parser.add_argument("--checkpoint", type=str, required=True)
    live_parser.add_argument(
        "--feature-builder", type=str, required=True,
        help="'module.path:function_name' -- see live/live_loop.py's module docstring.",
    )
    live_parser.add_argument("--n-features", type=int, required=True)
    live_parser.add_argument("--bar-interval-seconds", type=float, default=300.0)
    live_parser.add_argument(
        "--live", action="store_true", help="REAL capital. Omit this for paper trading (the default).",
    )
    live_parser.add_argument(
        "--auto-correct-mismatches", action="store_true",
        help="Auto-overwrite the internal ledger to match the broker when a state mismatch "
             "is detected (the kill-switch halt still requires a manual reset()). "
             "Default: detect-and-halt only.",
    )
    live_parser.set_defaults(func=cmd_live)

    monitor_parser = subparsers.add_parser("monitor", help="Standalone dashboard process watching a metrics file.")
    monitor_parser.add_argument("--metrics-path", type=str, default=None)
    monitor_parser.add_argument("--history-window", type=int, default=200)
    monitor_parser.add_argument("--poll-interval", type=float, default=2.0)
    monitor_parser.set_defaults(func=cmd_monitor)

    return parser


def main() -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()