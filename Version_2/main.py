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
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    from vec_trading_env import VecTradingEnv

    cfg = TrainingConfig()
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    # See train.py's module docstring / the matching fix note there:
    # MultiTickerRolloutDataset doesn't take a `tickers` kwarg -- it
    # auto-discovers tickers from metadata.json.
    test_dataset = MultiTickerRolloutDataset(
        window_size=cfg.env.window_size, split="test", device=str(device)
    )
    env = VecTradingEnv(dataset=test_dataset, initial_cash=cfg.env.initial_cash, device=str(device))

    actor_critic = _load_actor_critic(args.checkpoint, cfg, len(test_dataset.feature_names), device)

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
    )

    mode_str = "LIVE (real capital)" if not paper else "paper"
    print(f"[live] starting in {mode_str} mode against {len(cfg.env.tickers)} tickers")
    loop.run_forever()


def cmd_monitor(args: argparse.Namespace) -> None:
    from monitoring.dashboard import TrainingDashboard, resolve_mode

    cfg = TrainingConfig()
    metrics_path = args.metrics_path or cfg.run.metrics_path
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