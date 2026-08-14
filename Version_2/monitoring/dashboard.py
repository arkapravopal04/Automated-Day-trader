"""
monitoring/dashboard.py

Metrics/visualization layer, built Kaggle-safe from the start.

v1's bug wasn't Rich itself -- it was Rich's Live(auto_refresh=True), which
runs a background thread doing in-place ANSI cursor redraws that raced
against Kaggle's own stdout buffering and corrupted the display. Live() is
still only ever constructed with auto_refresh=False here, refreshed by an
explicit .refresh() tied to the same cadence as everything else.

v2 threaded-notebook bug (fixed in this revision): IPython's
display(display_id=...) / update_display() pairing is NOT guaranteed to
target the right output cell when called from a background polling thread
-- in practice on Kaggle this showed up as "the dashboard just keeps
printing itself again" instead of updating in place, because the comm
message from a background thread doesn't reliably land against the cell
that's still executing. The fix: when ipywidgets is available, render into
a persistent `ipywidgets.Output` widget and .clear_output(wait=True) inside
it every frame -- Output widgets are explicitly designed to capture output
from any thread and route it to the right place, which display_id/
update_display are not. Falls back to IPython.display.clear_output(wait=True)
+ display() (still same-cell in-place, just without the widget) if
ipywidgets isn't installed, and to Rich's Live() outside any notebook.

Decoupling (unchanged): the training loop NEVER renders anything and NEVER
imports Rich. It only calls MetricsWriter.log(step, **metrics) every N
steps, appending one JSON line to a flat file. Whether a dashboard is
watching that file, crashed, or was never started has zero effect on
training.

JSONL over SQLite, on purpose: no file-locking to get wrong on a
containerized/network-mounted filesystem -- a half-written last line just
gets skipped by MetricsReader.tail().
"""

import enum
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from IPython.display import HTML as _IPyHTML
    from IPython.display import display as _ipy_display
    from IPython.display import clear_output as _ipy_clear_output
    from IPython import get_ipython as _get_ipython
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

try:
    import ipywidgets as _ipywidgets
    _HAS_IPYWIDGETS = True
except ImportError:
    _HAS_IPYWIDGETS = False


# --------------------------------------------------------------------------
# Metrics writer -- the training loop calls THIS, never the dashboard directly.
# --------------------------------------------------------------------------

class MetricsWriter:
    """Write metrics to durable rollout and bounded tick JSONL logs.

    ``path`` remains the durable rollout log used by existing callers. Records
    with ``record_type == "tick"`` are routed to a separate bounded log.
    Legacy records without ``record_type`` continue to be treated as rollouts.
    """

    _TICK_SUFFIX = ".ticks.jsonl"
    _DEFAULT_TICK_MAX_BYTES = 8 * 1024 * 1024
    _DEFAULT_TICK_BACKUP_COUNT = 2

    def __init__(
        self,
        path: str,
        flush_every_call: bool = True,
        tick_path: Optional[str] = None,
        tick_max_bytes: int = _DEFAULT_TICK_MAX_BYTES,
        tick_backup_count: int = _DEFAULT_TICK_BACKUP_COUNT,
    ):
        self.path = path
        self.flush_every_call = flush_every_call
        self.tick_path = tick_path or self._derive_tick_path(path)
        self.tick_max_bytes = max(1, int(tick_max_bytes))
        self.tick_backup_count = max(0, int(tick_backup_count))

        self._fh = self._open_append(path)
        self._tick_fh = self._open_append(self.tick_path)

    @classmethod
    def _derive_tick_path(cls, path: str) -> str:
        if path.endswith(".jsonl"):
            return path[:-6] + cls._TICK_SUFFIX
        return path + cls._TICK_SUFFIX

    @staticmethod
    def _open_append(path: str):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        return open(path, "a", buffering=1, encoding="utf-8")

    def _rotate_tick_log_if_needed(self, line_size: int) -> None:
        """Rotate the tick stream before it exceeds the configured bound."""
        try:
            current_size = os.path.getsize(self.tick_path)
        except OSError:
            current_size = 0

        if current_size == 0 or current_size + line_size <= self.tick_max_bytes:
            return

        self._tick_fh.close()

        if self.tick_backup_count > 0:
            oldest = f"{self.tick_path}.{self.tick_backup_count}"
            try:
                os.remove(oldest)
            except FileNotFoundError:
                pass

            for index in range(self.tick_backup_count - 1, 0, -1):
                source = f"{self.tick_path}.{index}"
                target = f"{self.tick_path}.{index + 1}"
                try:
                    os.replace(source, target)
                except FileNotFoundError:
                    pass

            try:
                os.replace(self.tick_path, f"{self.tick_path}.1")
            except FileNotFoundError:
                pass
        else:
            try:
                os.remove(self.tick_path)
            except FileNotFoundError:
                pass

        self._tick_fh = self._open_append(self.tick_path)

    def _write(
        self,
        file_handle,
        path: str,
        record: Dict[str, Any],
        fsync: bool,
    ) -> None:
        line = json.dumps(record, default=_json_default) + "\n"
        if path == self.tick_path:
            self._rotate_tick_log_if_needed(len(line.encode("utf-8")))
            file_handle = self._tick_fh

        file_handle.write(line)
        if self.flush_every_call:
            file_handle.flush()
            if fsync:
                os.fsync(file_handle.fileno())

    def log(self, step: int, fsync: bool = True, **metrics: Any) -> None:
        """Append one metric record to the appropriate JSONL stream."""
        record = {"step": step, "wall_time": time.time(), **metrics}
        if record.get("record_type") == "tick":
            self._write(self._tick_fh, self.tick_path, record, fsync)
        else:
            self._write(self._fh, self.path, record, fsync)

    def close(self) -> None:
        for file_handle in (self._fh, self._tick_fh):
            if not file_handle.closed:
                file_handle.close()

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


# --------------------------------------------------------------------------
# Metrics reader
# --------------------------------------------------------------------------

class MetricsReader:
    """Read recent metrics from the durable rollout and tick log streams."""

    def __init__(
        self,
        path: str,
        tick_path: Optional[str] = None,
        tick_backup_count: int = MetricsWriter._DEFAULT_TICK_BACKUP_COUNT,
    ):
        self.path = path
        self.tick_path = tick_path
        self.tick_backup_count = max(0, int(tick_backup_count))

    def _paths(self) -> List[str]:
        if self.tick_path is None:
            return [self.path]
        return [
            self.path,
            self.tick_path,
            *(
                f"{self.tick_path}.{index}"
                for index in range(1, self.tick_backup_count + 1)
            ),
        ]

    @staticmethod
    def _tail_file(path: str, n: int) -> List[Dict[str, Any]]:
        """Read the last ``n`` complete JSONL records from one file."""
        if n <= 0 or not os.path.exists(path):
            return []

        chunk_size = 65536
        try:
            file_size = os.path.getsize(path)
        except OSError:
            return []
        if file_size == 0:
            return []

        with open(path, "rb") as file_handle:
            data = b""
            position = file_size
            while position > 0 and data.count(b"\n") <= n:
                read_size = min(chunk_size, position)
                position -= read_size
                file_handle.seek(position)
                data = file_handle.read(read_size) + data

        records: List[Dict[str, Any]] = []
        for line in data.decode("utf-8", errors="ignore").splitlines()[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A concurrent write may leave the final line temporarily torn.
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _sort_key(
        record: Dict[str, Any],
        source_index: int,
        record_index: int,
    ) -> Tuple[float, int, int]:
        try:
            wall_time = float(record.get("wall_time"))
        except (TypeError, ValueError):
            wall_time = float("-inf")
        return wall_time, source_index, record_index

    def tail(self, n: int) -> List[Dict[str, Any]]:
        """Return recent records, merging rollout and tick streams chronologically.

        For a single path, this preserves the historical ``tail(n)`` behavior.
        With a tick path configured, ``n`` records are collected per source
        (including rotated tick files), then globally ordered by ``wall_time``.
        """
        if n <= 0:
            return []

        per_path_records = [
            self._tail_file(path, n)
            for path in self._paths()
        ]

        merged: List[Tuple[Tuple[float, int, int], Dict[str, Any]]] = []
        for source_index, records in enumerate(per_path_records):
            for record_index, record in enumerate(records):
                merged.append((
                    self._sort_key(record, source_index, record_index),
                    record,
                ))

        merged.sort(key=lambda item: item[0])
        return [record for _, record in merged]



# --------------------------------------------------------------------------
# Display mode
# --------------------------------------------------------------------------

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
        configured = DisplayMode(cfg_display_mode)
    except ValueError:
        configured = DisplayMode.AUTO

    return resolve_display_mode(configured)


# --------------------------------------------------------------------------
# Dashboard constants
# --------------------------------------------------------------------------

_SPINNER_FRAMES = ["⠷", "⠯", "⠟", "⠿", "⡿", "⢿", "⣻", "⣾"]
_MAX_TRADE_EVENTS = 12
_MAX_PORTFOLIO_ROWS = 30
_GRID_THRESHOLD = 20


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def _fmt_uptime(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _as_list(value: Any, n: int) -> List[Any]:
    if isinstance(value, list) and len(value) == n:
        return value
    return [None] * n


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "—"

    number = _number(value)
    if number is None:
        return str(value)

    return f"{number:.2%}" if pct else f"{number:.4f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"

    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_money(value: Any, signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"

    if signed:
        return f"${number:+,.2f}"
    return f"${number:,.2f}"


def _colored_money(
    value: Any,
    *,
    explicit_color: Optional[str] = None,
    signed: bool = False,
) -> str:
    number = _number(value)
    if number is None:
        return "—"

    color = explicit_color
    if color is None:
        color = "bright_green" if number >= 0 else "bright_red"

    return f"[{color}]{_fmt_money(number, signed=signed)}[/{color}]"


def _colored_pct(value: Any, *, lower_is_better: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"

    if lower_is_better:
        color = "bright_green" if number <= 0.10 else "bright_red"
    else:
        color = "bright_green" if number >= 0 else "bright_red"

    return f"[{color}]{number:.2%}[/{color}]"


def _status_text(status: str, color: str = "grey62") -> Text:
    return Text(status, style=color)


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

class TrainingDashboard:
    """
    Human-readable live view of the training process.

    Design goals:
      1. Answer "is training alive?" immediately.
      2. Separate training statistics from portfolio statistics.
      3. Make risk and failures obvious.
      4. Keep live market information compact.
      5. Avoid duplicate or ambiguous labels.

    The training process remains completely decoupled from rendering. The
    dashboard only reads JSONL files through MetricsReader.
    """

    def __init__(
        self,
        metrics_path: str,
        mode: DisplayMode = DisplayMode.AUTO,
        history_window: int = 200,
        console: Optional[Console] = None,
        tick_metrics_path: Optional[str] = None,
        tick_backup_count: int = MetricsWriter._DEFAULT_TICK_BACKUP_COUNT,
    ):
        resolved_tick_path = (
            tick_metrics_path
            or MetricsWriter._derive_tick_path(metrics_path)
        )

        self.reader = MetricsReader(
            metrics_path,
            tick_path=resolved_tick_path,
            tick_backup_count=tick_backup_count,
        )

        self.mode = resolve_display_mode(mode)
        self.history_window = max(10, int(history_window))

        self._notebook_capable = _HAS_IPYTHON and _get_ipython() is not None
        self._widget_capable = self._notebook_capable and _HAS_IPYWIDGETS
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

        if console is not None:
            self.console = console
        elif self._is_tty:
            self.console = Console()
        else:
            # Notebook/headless output has no reliable terminal width.
            self.console = Console(width=160)

        self._live: Optional[Live] = None
        self._output_widget = None

        if self._widget_capable:
            self._output_widget = _ipywidgets.Output()
        elif self.mode == DisplayMode.LOCAL and self._is_tty and not self._notebook_capable:
            self._live = Live(
                console=self.console,
                auto_refresh=False,
                transient=False,
            )

        self._displayed_widget = False
        self._headless_lines_printed = 0

        self._start_time = time.time()
        self._last_step: Optional[int] = None
        self._frame_count = 0

        # Used only to show actual changes between frames.
        self._previous_prices: Dict[str, float] = {}
        self._previous_net_worth: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle / rendering backend
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._output_widget is not None and not self._displayed_widget:
            _ipy_display(self._output_widget)
            self._displayed_widget = True

        if self._live is not None:
            self._live.__enter__()

    def stop(self) -> None:
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None

    def __enter__(self) -> "TrainingDashboard":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def render_once(self) -> None:
        history = self.reader.tail(self.history_window)
        if not history:
            return

        latest_tick = self._latest_record(history, "tick")
        latest_rollout = self._latest_rollout(history)

        # A rollout-only log is still valid. This keeps the dashboard useful
        # with older metrics files.
        latest_tick = latest_tick or latest_rollout
        latest_rollout = latest_rollout or latest_tick

        if latest_tick is None:
            return

        step = latest_tick.get("step")
        is_new_step = step != self._last_step

        if is_new_step:
            self._frame_count += 1

        self._last_step = step

        renderable = self._build_renderable(
            tick_record=latest_tick,
            rollout_record=latest_rollout,
            history=history,
            now=time.time(),
            is_new_step=is_new_step,
        )

        self._render(renderable)

    @staticmethod
    def _latest_record(
        history: List[Dict[str, Any]],
        record_type: str,
    ) -> Optional[Dict[str, Any]]:
        return next(
            (
                record
                for record in reversed(history)
                if record.get("record_type") == record_type
            ),
            None,
        )

    @staticmethod
    def _latest_rollout(
        history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return next(
            (
                record
                for record in reversed(history)
                if record.get("record_type", "rollout") != "tick"
            ),
            None,
        )

    def _render(self, renderable: Any) -> None:
        if self._output_widget is not None:
            self.start()
            with self._output_widget:
                self._output_widget.clear_output(wait=True)
                self.console.print(renderable)
            return

        if self._live is not None:
            self._live.update(renderable)
            self._live.refresh()
            return

        if self._notebook_capable:
            html = self._render_html(renderable)
            _ipy_clear_output(wait=True)
            _ipy_display(_IPyHTML(html))
            return

        self._render_headless_inplace(renderable)

    def _render_html(self, renderable: Any) -> str:
        buffer = io.StringIO()
        temp_console = Console(
            file=buffer,
            record=True,
            width=100,
            force_terminal=False,
        )
        temp_console.print(renderable)
        return temp_console.export_html(
            inline_styles=True,
            code_format='<pre style="white-space:pre-wrap;font-family:monospace">{code}</pre>',
        )

    def _render_headless_inplace(self, renderable: Any) -> None:
        buffer = io.StringIO()
        temp_console = Console(
            file=buffer,
            force_terminal=True,
            width=self.console.width or 100,
        )
        temp_console.print(renderable)

        text = buffer.getvalue()
        line_count = text.count("\n")

        if self._headless_lines_printed:
            sys.stdout.write(f"\x1b[{self._headless_lines_printed}A")
            sys.stdout.write("\x1b[J")

        sys.stdout.write(text)
        sys.stdout.flush()
        self._headless_lines_printed = line_count

    def run_polling_loop(
        self,
        poll_interval_seconds: float = 2.0,
        max_iterations: Optional[int] = None,
    ) -> None:
        iteration = 0

        try:
            self.start()

            while max_iterations is None or iteration < max_iterations:
                self.render_once()
                time.sleep(max(0.1, poll_interval_seconds))
                iteration += 1
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Main layout
    # ------------------------------------------------------------------

    def _build_renderable(
        self,
        tick_record: Dict[str, Any],
        rollout_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        now: float,
        is_new_step: bool,
    ) -> Group:
        tick_history = [
            record
            for record in history
            if record.get("record_type") == "tick"
        ]

        status = self._build_status_panel(
            tick_record,
            rollout_record,
            tick_history,
            now,
            is_new_step,
        )

        performance = self._build_performance_panel(rollout_record)
        portfolio, rows = self._build_portfolio_panel(tick_record)

        market = self._build_market_panel(rows)
        trades = self._build_trade_tape(tick_history)
        environment = self._build_environment_view(rows, tick_record)

        return Group(
            status,
            performance,
            portfolio,
            market,
            trades,
            environment,
        )

    # ------------------------------------------------------------------
    # Section 1: system/training status
    # ------------------------------------------------------------------

    def _build_status_panel(
        self,
        tick: Dict[str, Any],
        rollout: Dict[str, Any],
        tick_history: List[Dict[str, Any]],
        now: float,
        is_new_step: bool,
    ) -> Panel:
        latest_wall_time = _number(tick.get("wall_time")) or now
        age = max(0, now - latest_wall_time)

        if age < 10:
            health = "LIVE"
            health_style = "bold white on green"
        elif age < 30:
            health = "STALE"
            health_style = "bold black on yellow"
        else:
            health = "STOPPED?"
            health_style = "bold white on red"

        spinner = (
            _SPINNER_FRAMES[self._frame_count % len(_SPINNER_FRAMES)]
            if is_new_step
            else "●"
        )

        throughput = self._ticks_per_second(tick_history)
        throughput_text = f"{throughput:.1f}/s" if throughput is not None else "—"

        tickers = tick.get("tickers")
        ticker_count = len(tickers) if isinstance(tickers, list) else 0

        halted = tick.get("halted")
        halted_count = (
            sum(bool(item) for item in halted)
            if isinstance(halted, list)
            else 0
        )

        lines = [
            f"[bold]TRAINING DASHBOARD[/bold]   "
            f"[{health_style}] {health} [/{health_style}]  "
            f"[cyan]{spinner}[/cyan]  "
            f"[grey62]updated {age:.0f}s ago[/grey62]",
            "",
            f"[bold]Environment step[/bold]   {_fmt_int(tick.get('step'))}",
            f"[bold]Rollout[/bold]           {_fmt_int(rollout.get('rollout', rollout.get('step')))}",
            f"[bold]Episode[/bold]           {_fmt_int(rollout.get('episode'))}",
            f"[bold]Market symbols[/bold]    {ticker_count}",
            f"[bold]Processing rate[/bold]   {throughput_text} ticks/s",
            f"[bold]Runtime[/bold]            {_fmt_uptime(now - self._start_time)}",
        ]

        if halted_count:
            lines.append(
                f"[bold]Attention[/bold]          "
                f"[white on purple4] {halted_count} HALTED [/white on purple4]"
            )
        else:
            lines.append("[bold]Attention[/bold]          none")

        return Panel(
            "\n".join(lines),
            title="1. System Status",
            border_style="cyan",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 2: model/training performance
    # ------------------------------------------------------------------

    def _build_performance_panel(
        self,
        rollout: Dict[str, Any],
    ) -> Panel:
        reward = rollout.get("reward")
        reward_ema = rollout.get("reward_ema")
        sharpe = rollout.get("sharpe")

        lines = [
            f"[bold]Reward[/bold]            {_colored_money(reward, signed=True)}",
            f"[bold]Reward EMA[/bold]        {_colored_money(reward_ema, signed=True)}",
            f"[bold]Sharpe[/bold]            {_fmt(sharpe)}",
        ]

        # Preserve other common PPO metrics when they exist, without making
        # the dashboard depend on them.
        optional_fields = (
            ("policy_loss", "Policy loss"),
            ("value_loss", "Value loss"),
            ("entropy", "Entropy"),
        )

        for key, label in optional_fields:
            if key in rollout:
                lines.append(f"[bold]{label:<20}[/bold] {_fmt(rollout.get(key))}")

        lines.append("")
        lines.append(
            "[grey62]Training metrics are taken from the most recent rollout; "
            "they do not update every environment tick.[/grey62]"
        )

        return Panel(
            "\n".join(lines),
            title="2. Model Performance",
            border_style="blue",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 3: portfolio health
    # ------------------------------------------------------------------

    def _build_portfolio_panel(
        self,
        tick: Dict[str, Any],
    ) -> Tuple[Panel, List[Dict[str, Any]]]:
        net_worth = tick.get("net_worth")
        drawdown = tick.get("drawdown")
        trades = tick.get("total_trades")
        rollout_trades = tick.get("trades_this_rollout")

        net_worth_change_style = None
        current_net_worth = _number(net_worth)

        if current_net_worth is not None and self._previous_net_worth is not None:
            if current_net_worth > self._previous_net_worth:
                net_worth_change_style = "bright_green"
            elif current_net_worth < self._previous_net_worth:
                net_worth_change_style = "bright_red"

        if current_net_worth is not None:
            self._previous_net_worth = current_net_worth

        tickers = tick.get("tickers")
        positions = tick.get("position")

        rows = self._compute_per_ticker_rows(tick, tickers, positions)

        halted = sum(row["is_halted"] for row in rows)
        bankrupt = sum(row["is_bankrupt"] for row in rows)
        active_positions = sum(
            _number(row["position"]) not in (None, 0)
            for row in rows
        )

        net_worth_text = _colored_money(
            net_worth,
            explicit_color=net_worth_change_style,
        )

        lines = [
            f"[bold]Net worth[/bold]          {net_worth_text}",
            f"[bold]Average drawdown[/bold]   {_colored_pct(drawdown, lower_is_better=True)}",
            f"[bold]Trades this rollout[/bold] {_fmt_int(rollout_trades)}",
            f"[bold]Total trades[/bold]       {_fmt_int(trades)}",
            "",
            f"[bold]Open positions[/bold]      {active_positions}",
            f"[bold]Halted symbols[/bold]      {halted}",
            f"[bold]Bankrupt symbols[/bold]    {bankrupt}",
        ]

        if bankrupt:
            lines.append("[bold red]Risk alert: one or more symbols have zero/negative net worth.[/bold red]")
        elif halted:
            lines.append("[bold yellow]Operational alert: one or more symbols are halted.[/bold yellow]")
        else:
            lines.append("[green]Portfolio health: no halted or bankrupt symbols.[/green]")

        return (
            Panel(
                "\n".join(lines),
                title="3. Portfolio Health",
                border_style="green",
                expand=False,
            ),
            rows,
        )

    # ------------------------------------------------------------------
    # Section 4: market view
    # ------------------------------------------------------------------

    def _build_market_panel(
        self,
        rows: List[Dict[str, Any]],
    ) -> Panel:
        if not rows:
            return Panel(
                "[grey62]No ticker price data is available in this record.[/grey62]",
                title="4. Market",
                border_style="grey42",
                expand=False,
            )

        tape = Text()

        for row in rows:
            ticker = str(row["ticker"])
            price = row["price"]
            arrow = row["price_arrow"]

            if price is None:
                tape.append(f" {ticker} — ", style="grey50")
                tape.append("|", style="grey35")
                continue

            price_style = row["price_color"] or "white"
            arrow_style = row["price_color"] or "grey62"

            tape.append(f" {ticker} ", style="bold white")
            tape.append(f"{price:,.2f}", style=price_style)
            tape.append(f" {arrow} ", style=arrow_style)
            tape.append("|", style="grey35")

        return Panel(
            tape,
            title=f"4. Market — {len(rows)} symbols",
            border_style="magenta",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 5: fills/trades
    # ------------------------------------------------------------------

    def _build_trade_tape(
        self,
        tick_history: List[Dict[str, Any]],
    ) -> Panel:
        events: List[str] = []

        for record in reversed(tick_history):
            if len(events) >= _MAX_TRADE_EVENTS:
                break

            tickers = record.get("tickers")
            filled = record.get("filled_qty_this_tick")
            prices = record.get("price_per_ticker")

            if not isinstance(tickers, list) or not isinstance(filled, list):
                continue

            step = record.get("step")

            for index, quantity in enumerate(filled):
                if len(events) >= _MAX_TRADE_EVENTS:
                    break

                quantity_number = _number(quantity)
                if quantity_number is None or quantity_number == 0:
                    continue

                ticker = (
                    tickers[index]
                    if index < len(tickers)
                    else "?"
                )

                price = (
                    prices[index]
                    if isinstance(prices, list) and index < len(prices)
                    else None
                )

                if quantity_number > 0:
                    side = "BUY "
                    color = "bright_green"
                else:
                    side = "SELL"
                    color = "bright_red"

                price_text = (
                    f" @ ${_number(price):,.2f}"
                    if _number(price) is not None
                    else ""
                )

                events.append(
                    f"[grey62]t{_fmt_int(step):>7}[/grey62]  "
                    f"[bold {color}]{side}[/bold {color}]  "
                    f"[bold]{str(ticker):<8}[/bold]  "
                    f"{abs(quantity_number):>8.2f} sh"
                    f"{price_text}"
                )

        if not events:
            body = "[grey62]No fills recorded in the visible history.[/grey62]"
        else:
            body = "\n".join(events)

        return Panel(
            body,
            title="5. Recent Fills",
            border_style="yellow",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 6: per-symbol state
    # ------------------------------------------------------------------

    def _build_environment_view(
        self,
        rows: List[Dict[str, Any]],
        tick_record: Dict[str, Any],
    ) -> Any:
        if not rows:
            return Panel(
                "[grey62]No per-symbol state is available in this record.[/grey62]",
                title="6. Symbol Details",
                border_style="grey42",
                expand=False,
            )

        if len(rows) <= _GRID_THRESHOLD:
            return self._build_detail_table(rows, tick_record)

        # For large portfolios, show a compact overview first, then the most
        # important rows. This keeps the dashboard readable without hiding
        # risk conditions.
        grid = self._build_compact_grid(rows)

        prioritized = sorted(
            rows,
            key=lambda row: (
                not (row["is_bankrupt"] or row["is_halted"]),
                -abs(_number(row.get("unrealized_value")) or 0.0),
            ),
        )

        return Group(
            Panel(
                grid,
                title=f"6. Symbol Overview — {len(rows)} symbols",
                border_style="grey50",
                expand=False,
            ),
            self._build_detail_table(
                prioritized,
                tick_record,
                max_rows=_MAX_PORTFOLIO_ROWS,
                title="Priority Symbol Details",
            ),
        )

    def _compute_per_ticker_rows(
        self,
        tick_record: Dict[str, Any],
        tickers: Any,
        position: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(tickers, list) or not isinstance(position, list):
            return []

        count = len(tickers)

        net_worth = _as_list(
            tick_record.get("net_worth_per_ticker"),
            count,
        )
        prices = _as_list(
            tick_record.get("price_per_ticker"),
            count,
        )
        unrealized = _as_list(
            tick_record.get("unrealized_pnl"),
            count,
        )
        drawdown = _as_list(
            tick_record.get("drawdown_per_ticker"),
            count,
        )
        trades_rollout = _as_list(
            tick_record.get("trades_per_ticker_this_rollout"),
            count,
        )
        trades_total = _as_list(
            tick_record.get("total_trades_per_ticker"),
            count,
        )
        filled = _as_list(
            tick_record.get("filled_qty_this_tick"),
            count,
        )
        halted = _as_list(
            tick_record.get("halted"),
            count,
        )

        rows: List[Dict[str, Any]] = []

        for index, ticker in enumerate(tickers):
            price = _number(prices[index])
            previous_price = self._previous_prices.get(str(ticker))

            if price is None:
                price_color = None
                price_arrow = ""
            elif previous_price is None:
                price_color = None
                price_arrow = "•"
            elif price > previous_price:
                price_color = "bright_green"
                price_arrow = "▲"
            elif price < previous_price:
                price_color = "bright_red"
                price_arrow = "▼"
            else:
                price_color = None
                price_arrow = "•"

            # Store once per frame. This fixes the old double-update problem
            # where the market tape could lose the direction arrow.
            if price is not None:
                self._previous_prices[str(ticker)] = price

            nw = _number(net_worth[index])
            pnl = _number(unrealized[index])
            is_halted = bool(halted[index]) if halted[index] is not None else False
            is_bankrupt = nw is not None and nw <= 0

            if is_bankrupt:
                status = "[bold white on red] BANKRUPT [/bold white on red]"
                row_style = "on red"
            elif is_halted:
                status = "[bold white on purple4] HALTED [/bold white on purple4]"
                row_style = "on purple4"
            else:
                status = "[green]OK[/green]"
                row_style = None

            rows.append(
                {
                    "ticker": ticker,
                    "price": price,
                    "price_color": price_color,
                    "price_arrow": price_arrow,
                    "position": position[index],
                    "net_worth": nw,
                    "unrealized_value": pnl,
                    "drawdown": _number(drawdown[index]),
                    "trades_rollout": trades_rollout[index],
                    "trades_total": trades_total[index],
                    "filled": filled[index],
                    "is_halted": is_halted,
                    "is_bankrupt": is_bankrupt,
                    "status_text": status,
                    "row_style": row_style,
                }
            )

        return rows

    def _build_detail_table(
        self,
        rows: List[Dict[str, Any]],
        tick_record: Dict[str, Any],
        max_rows: Optional[int] = None,
        title: Optional[str] = None,
    ) -> Table:
        total = len(rows)
        visible = rows if max_rows is None else rows[:max_rows]
        hidden = max(0, total - len(visible))

        table = Table(
            title=(
                title
                or f"Symbol Details — step {_fmt_int(tick_record.get('step'))}"
            ),
            expand=False,
        )

        table.add_column("Symbol")
        table.add_column("Price", justify="right")
        table.add_column("Position", justify="right")
        table.add_column("Net Worth", justify="right")
        table.add_column("Unrealized PnL", justify="right")
        table.add_column("Drawdown", justify="right")
        table.add_column("Trades", justify="right")
        table.add_column("Status")

        for row in visible:
            pnl = row["unrealized_value"]
            pnl_text = _colored_money(pnl, signed=True)

            drawdown = row["drawdown"]
            drawdown_text = (
                _colored_pct(drawdown, lower_is_better=True)
                if drawdown is not None
                else "—"
            )

            table.add_row(
                str(row["ticker"]),
                (
                    f"{row['price']:,.2f} {row['price_arrow']}"
                    if row["price"] is not None
                    else "—"
                ),
                _fmt(row["position"]),
                _fmt_money(row["net_worth"]),
                pnl_text,
                drawdown_text,
                _fmt_int(row["trades_total"]),
                row["status_text"],
                style=row["row_style"],
            )

        if hidden:
            table.caption = (
                f"{hidden} additional symbols omitted from the detailed view; "
                "the compact overview above still includes them."
            )

        return table

    def _build_compact_grid(
        self,
        rows: List[Dict[str, Any]],
    ) -> Table:
        column_width = 22
        columns = max(
            4,
            min(12, (self.console.width or 100) // column_width),
        )

        grid = Table.grid(padding=(0, 1))

        for _ in range(columns):
            grid.add_column()

        cells: List[str] = []

        for row in rows:
            if row["is_bankrupt"]:
                marker = "[red]●[/red]"
            elif row["is_halted"]:
                marker = "[purple4]●[/purple4]"
            else:
                marker = "[grey42]●[/grey42]"

            price = (
                f"{row['price']:,.2f}"
                if row["price"] is not None
                else "—"
            )

            arrow = row["price_arrow"]
            arrow_style = row["price_color"] or "grey62"

            cells.append(
                f"{marker} [bold]{str(row['ticker']):<7}[/bold]\n"
                f"  {price} [{arrow_style}]{arrow}[/{arrow_style}]"
                f"  pos {_fmt(row['position'])}"
            )

            if len(cells) == columns:
                grid.add_row(*cells)
                cells = []

        if cells:
            cells.extend([""] * (columns - len(cells)))
            grid.add_row(*cells)

        return grid

    @staticmethod
    def _ticks_per_second(
        tick_history: List[Dict[str, Any]],
    ) -> Optional[float]:
        if len(tick_history) < 2:
            return None

        first = tick_history[0]
        last = tick_history[-1]

        first_step = _number(first.get("step"))
        last_step = _number(last.get("step"))
        first_time = _number(first.get("wall_time"))
        last_time = _number(last.get("wall_time"))

        if None in (first_step, last_step, first_time, last_time):
            return None

        step_span = last_step - first_step
        time_span = last_time - first_time

        if step_span <= 0 or time_span <= 0:
            return None

        return step_span / time_span


# --------------------------------------------------------------------------
# Backwards-compatible formatting helper
# --------------------------------------------------------------------------

def _fmt_colored(
    value: Any,
    good_is: Optional[str] = "positive",
    color: Optional[str] = None,
    pct: bool = False,
    dollar: bool = False,
    flash: bool = False,
) -> str:
    """
    Compatibility helper retained for callers that imported it.

    The dashboard itself now uses the clearer formatting helpers above.
    """
    if value is None:
        return "—"

    number = _number(value)
    if number is None:
        return str(value)

    text = (
        f"${number:,.2f}"
        if dollar
        else f"{number:.2%}"
        if pct
        else f"{number:.4f}"
    )

    resolved_color = color

    if resolved_color is None:
        if good_is == "positive":
            resolved_color = "bright_green" if number >= 0 else "bright_red"
        elif good_is == "small_pct":
            resolved_color = "bright_green" if number <= 0.10 else "bright_red"

    if resolved_color is None:
        return text

    style = f"bold reverse {resolved_color}" if flash else resolved_color
    return f"[{style}]{text}[/{style}]"
