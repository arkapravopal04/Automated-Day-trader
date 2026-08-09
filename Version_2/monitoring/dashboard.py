"""
monitoring/dashboard.py

Metrics/visualization layer, built Kaggle-safe from the start.

Decoupling:
    The training loop NEVER renders anything and NEVER imports Rich. It
    only calls MetricsWriter.log(step, **metrics) every N steps.

JSONL over SQLite, on purpose: SQLite's file-locking semantics are a known
footgun on containerized/network-mounted filesystems. JSONL appends have 
no locking to get wrong.

Overhaul Features:
    - htop-style Full-screen Layout for local mode to prevent terminal scroll.
    - Real-time sparklines and animated last-updated timestamps.
    - Dedicated rendering for per-environment worker metrics.
"""

import enum
import io
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from rich.text import Text

try:
    from IPython.display import HTML as _IPyHTML
    from IPython.display import display as _ipy_display
    from IPython.display import update_display as _ipy_update_display
    from IPython import get_ipython as _get_ipython
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


class MetricsWriter:
    """
    Append-only structured metrics log (JSONL). One line per log() call.
    """

    def __init__(self, path: str, flush_every_call: bool = True):
        self.path = path
        self.flush_every_call = flush_every_call
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "a", buffering=1)

    def log(self, step: int, **metrics: Any) -> None:
        """
        step: the training step/rollout index this record belongs to.
        **metrics: 
            Can include overall metrics (reward, net_worth) AND 
            a list of dicts for per-environment tracking, e.g.:
            env_metrics=[{"env_id": 0, "trades": 5, "reward": 0.2}, ...]
        """
        record = {"step": step, "wall_time": time.time(), **metrics}
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        if self.flush_every_call:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    return str(obj)


class MetricsReader:
    def __init__(self, path: str):
        self.path = path

    def tail(self, n: int) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r") as f:
            lines = f.readlines()
        records: List[Dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


class DisplayMode(str, enum.Enum):
    AUTO = "auto"
    KAGGLE = "kaggle"
    LOCAL = "local"


def _looks_like_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ


def resolve_display_mode(mode: DisplayMode) -> DisplayMode:
    if mode != DisplayMode.AUTO:
        return mode
    return DisplayMode.KAGGLE if _looks_like_kaggle() else DisplayMode.LOCAL


def resolve_mode(cfg_display_mode: str = "auto", argv: Optional[List[str]] = None) -> DisplayMode:
    argv = argv if argv is not None else sys.argv[1:]
    if "--kaggle" in argv:
        return DisplayMode.KAGGLE
    if "--local" in argv:
        return DisplayMode.LOCAL
    try:
        cfg_mode = DisplayMode(cfg_display_mode)
    except ValueError:
        cfg_mode = DisplayMode.AUTO
    return resolve_display_mode(cfg_mode)


class TrainingDashboard:
    def __init__(
        self,
        metrics_path: str,
        mode: DisplayMode = DisplayMode.AUTO,
        history_window: int = 200,
        console: Optional[Console] = None,
    ):
        self.reader = MetricsReader(metrics_path)
        self.mode = resolve_display_mode(mode)
        self.history_window = history_window
        self.console = console if console is not None else Console()
        self._tick_counter = 0

        self._live: Optional[Live] = None
        if self.mode == DisplayMode.LOCAL:
            # screen=True puts the terminal in an alternate buffer (like htop).
            # This completely solves the "re-printing/scrolling" issue.
            self._live = Live(
                console=self.console, 
                auto_refresh=False, 
                transient=False,
                screen=True  
            )

        self._notebook_capable = _HAS_IPYTHON and _get_ipython() is not None
        self._display_id = f"training_dashboard_{id(self)}"
        self._displayed_once = False
        self._prev_net_worth: Optional[float] = None

    def start(self) -> None:
        if self._live is not None:
            self._live.__enter__()

    def stop(self) -> None:
        if self._live is not None:
            self._live.__exit__(None, None, None)

    def __enter__(self) -> "TrainingDashboard":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def render_once(self) -> None:
        history = self.reader.tail(self.history_window)
        if not history:
            return
        
        latest = history[-1]
        self._tick_counter += 1
        
        renderable = self._build_layout(latest, history)

        if self.mode == DisplayMode.LOCAL:
            self._live.update(renderable)
            self._live.refresh()
        elif self._notebook_capable:
            html = self._render_html(renderable)
            if not self._displayed_once:
                _ipy_display(_IPyHTML(html), display_id=self._display_id)
                self._displayed_once = True
            else:
                _ipy_update_display(_IPyHTML(html), display_id=self._display_id)
        else:
            self.console.print(renderable)

    def _render_html(self, renderable) -> str:
        buf = io.StringIO()
        tmp_console = Console(file=buf, record=True, width=120, force_terminal=False)
        tmp_console.print(renderable)
        return tmp_console.export_html(
            inline_styles=True,
            code_format='<pre style="white-space:pre-wrap;font-family:monospace;background:#1e1e1e;color:#d4d4d4;padding:10px">{code}</pre>',
        )

    def run_polling_loop(self, poll_interval_seconds: float = 1.0, max_iterations: Optional[int] = None) -> None:
        i = 0
        try:
            self.start()
            while max_iterations is None or i < max_iterations:
                self.render_once()
                time.sleep(poll_interval_seconds)
                i += 1
        finally:
            self.stop()

    def _build_layout(self, latest: Dict[str, Any], history: List[Dict[str, Any]]) -> Layout:
        """Constructs a responsive UI grid layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=9),
            Layout(name="body")
        )
        layout["body"].split_row(
            Layout(name="portfolio", ratio=1),
            Layout(name="envs", ratio=1)
        )

        # 1. Header (Overall Status & Sparklines)
        layout["header"].update(self._build_header(latest, history))
        
        # 2. Portfolio / Tickers
        layout["portfolio"].update(self._build_portfolio_table(latest))
        
        # 3. Environment Workers
        layout["envs"].update(self._build_envs_table(latest))

        return layout

    def _build_header(self, latest: Dict[str, Any], history: List[Dict[str, Any]]) -> Panel:
        reward = latest.get("reward")
        net_worth = latest.get("net_worth")

        net_worth_color = None
        if net_worth is not None and self._prev_net_worth is not None:
            if net_worth > self._prev_net_worth:
                net_worth_color = "green"
            elif net_worth < self._prev_net_worth:
                net_worth_color = "red"
        if net_worth is not None:
            self._prev_net_worth = net_worth

        # Dynamic alive indicator (rotates characters based on tick)
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = spinners[self._tick_counter % len(spinners)]
        now = datetime.now().strftime("%H:%M:%S")
        status_line = f"[bold cyan]{spinner} LIVE[/bold cyan] | Updated: [dim]{now}[/dim]"

        # Sparklines for historical feel
        rewards_hist = [h.get("reward", 0) for h in history if h.get("reward") is not None]
        sparkline = _generate_sparkline(rewards_hist[-50:]) if rewards_hist else "N/A"

        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="left", ratio=1)
        
        left_lines = [
            f"Step:            [bold]{latest.get('step')}[/bold]",
            f"Net Worth:       [bold]{_fmt_colored(net_worth, good_is=None, color=net_worth_color, dollar=True)}[/bold]",
            f"Reward (Latest): {_fmt_colored(reward, good_is='positive')}",
            f"Sharpe:          {_fmt(latest.get('sharpe'))}"
        ]
        
        right_lines = [
            f"Drawdown:        {_fmt_colored(latest.get('drawdown'), good_is='small_pct', pct=True)}",
            f"Total Trades:    {_fmt_int(latest.get('total_trades'))}",
            f"Reward Trend:    [magenta]{sparkline}[/magenta]",
            status_line
        ]

        grid.add_row("\n".join(left_lines), "\n".join(right_lines))
        return Panel(grid, title="[bold white]Global Agent Metrics[/bold white]", border_style="cyan")

    def _build_portfolio_table(self, latest: Dict[str, Any]) -> Panel:
        table = Table(expand=True, show_edge=False, header_style="bold yellow")
        table.add_column("Ticker")
        table.add_column("Position", justify="right")
        table.add_column("Unrealized PnL", justify="right")

        tickers = latest.get("tickers")
        position = latest.get("position")

        if isinstance(tickers, list) and isinstance(position, list):
            unrealized = latest.get("unrealized_pnl")
            unrealized = unrealized if isinstance(unrealized, list) else [None] * len(tickers)
            for ticker, pos, upnl in zip(tickers, position, unrealized):
                table.add_row(str(ticker), _fmt(pos), _fmt_colored(upnl, good_is="positive"))
        else:
            table.add_row("(No positions)", "-", "-")

        return Panel(table, title="[bold yellow]Portfolio State[/bold yellow]", border_style="yellow")

    def _build_envs_table(self, latest: Dict[str, Any]) -> Panel:
        table = Table(expand=True, show_edge=False, header_style="bold green")
        table.add_column("Env ID", justify="left")
        table.add_column("Trades", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("Status", justify="center")

        env_metrics = latest.get("env_metrics", [])
        
        if isinstance(env_metrics, list) and env_metrics:
            # Sort environments by ID to keep the display stable
            env_metrics = sorted(env_metrics, key=lambda x: x.get("env_id", 0))
            for env in env_metrics:
                trades = env.get("trades", 0)
                reward = env.get("reward", 0)
                # Dynamic visual status logic based on trades
                status = "[green]Active[/green]" if trades > 0 else "[dim]Idle[/dim]"
                table.add_row(
                    f"Worker-{env.get('env_id', '?')}",
                    _fmt_int(trades),
                    _fmt_colored(reward, good_is="positive"),
                    status
                )
        else:
            table.add_row("(No environment metrics provided)", "-", "-", "-")

        return Panel(table, title="[bold green]Environment Workers[/bold green]", border_style="green")


def _generate_sparkline(values: List[float]) -> str:
    """Creates a text-based sparkline for a list of floats."""
    if not values:
        return ""
    bars = "  ▂▃▄▅▆▇█"
    min_v, max_v = min(values), max(values)
    range_v = max_v - min_v
    if range_v == 0:
        return bars[4] * len(values)
    
    spark = []
    for v in values:
        idx = int((v - min_v) / range_v * 8)
        idx = max(0, min(idx, 8))
        spark.append(bars[idx])
    return "".join(spark)


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:.2%}" if pct else f"{v:.4f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_colored(
    value: Any,
    good_is: Optional[str] = "positive",
    color: Optional[str] = None,
    pct: bool = False,
    dollar: bool = False,
) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    text = f"${v:,.2f}" if dollar else (f"{v:.2%}" if pct else f"{v:.4f}")

    resolved_color = color
    if resolved_color is None:
        if good_is == "positive":
            resolved_color = "green" if v >= 0 else "red"
        elif good_is == "small_pct":
            resolved_color = "green" if v <= 0.10 else "red"

    return f"[{resolved_color}]{text}[/{resolved_color}]" if resolved_color else text