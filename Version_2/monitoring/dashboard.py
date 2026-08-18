"""
monitoring/dashboard.py

Metrics/visualization layer, built Kaggle-safe from the start.

v1's bug wasn't Rich itself -- it was Rich's Live(auto_refresh=True), which
runs a background thread doing in-place ANSI cursor redraws that raced
against Kaggle's own stdout buffering and corrupted the display. Live() is
still only ever constructed with auto_refresh=False here, refreshed by an
explicit .refresh() tied to the same cadence as everything else.

v2 threaded-notebook bug (fixed): IPython's display(display_id=...) /
update_display() pairing is NOT guaranteed to target the right output cell
when called from a background polling thread. The fix: when ipywidgets is
available, render into a persistent `ipywidgets.Output` widget and
.clear_output(wait=True) inside it every frame. Falls back to
IPython.display.clear_output(wait=True) + display() outside a notebook,
and to Rich's Live() on a real TTY.

v3 (layout redesign): the old layout buried the things you actually care
about under walls of numbers -- ~100 per-ticker rows, an unreadable market
tape, reward displayed as dollars, a dead "Sharpe" line, and an entropy
line that never rendered (training emits entropy_discrete/continuous, not
entropy). The new layout is six clear sections:
    1. Run Status   -- health, step/rollout/episode, throughput, data age
    2. Alerts       -- halted / bankrupt / deep-drawdown streams, or "none"
    3. Training     -- reward, EMA, entropy (both heads), losses, KL, grad
    4. Portfolio    -- aggregate net worth + context, spotlight top/bottom
    5. Market       -- compact grid of all symbols + movers + up/down counts
    6. Recent Fills -- last few fills, oldest -> newest

Decoupling (unchanged): the training loop NEVER renders anything and NEVER
imports Rich. It only calls MetricsWriter.log(step, **metrics), appending
one JSON line to a flat file. Whether a dashboard is watching that file,
crashed, or was never started has zero effect on training.

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
_MAX_TRADE_EVENTS = 8          # fills shown in section 6
_SPOTLIGHT_ROWS = 10           # top/bottom rows in the portfolio spotlight
_GRID_THRESHOLD = 20           # tickers beyond this get the compact grid
_MOVERS_COUNT = 3              # biggest movers shown in the market section
_DD_ALERT_FRAC = 0.25          # per-stream drawdown that raises an alert
_HEALTH_LIVE_S = 10
_HEALTH_STALE_S = 30


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


def _fmt_signed(value: Any, decimals: int = 6) -> str:
    """Signed plain number -- for reward/EMA-style scalars, NOT money."""
    if value is None:
        return "—"
    number = _number(value)
    if number is None:
        return str(value)
    return f"{number:+,.{decimals}f}"


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

    Design goals (v3):
      1. Answer "is training alive?" in one glance (Run Status panel).
      2. Make risk impossible to miss (dedicated Alerts panel, not a badge
         buried in a table).
      3. Show aggregates and spotlights, not 100 rows of numbers.
      4. Keep market information compact (grid + movers, no giant tape).
      5. Never display a scalar as the wrong kind of thing (reward is a
         dimensionless number, not dollars).

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
    # Main layout -- six clear sections, aggregates over walls of numbers
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

        rows = self._compute_per_ticker_rows(tick_record)

        return Group(
            self._build_status_panel(tick_record, rollout_record, tick_history, now, is_new_step, rows),
            self._build_alerts_panel(rows),
            self._build_training_panel(rollout_record),
            self._build_portfolio_panel(tick_record, rows),
            self._build_market_panel(rows),
            self._build_fills_panel(tick_history),
        )

    # ------------------------------------------------------------------
    # Section 1: run status -- is training alive, and at what pace
    # ------------------------------------------------------------------

    def _build_status_panel(
        self,
        tick: Dict[str, Any],
        rollout: Dict[str, Any],
        tick_history: List[Dict[str, Any]],
        now: float,
        is_new_step: bool,
        rows: List[Dict[str, Any]],
    ) -> Panel:
        latest_wall_time = _number(tick.get("wall_time")) or now
        age = max(0, now - latest_wall_time)

        if age < _HEALTH_LIVE_S:
            health, health_style = "LIVE", "bold white on green"
        elif age < _HEALTH_STALE_S:
            health, health_style = "STALE", "bold black on yellow"
        else:
            health, health_style = "STOPPED?", "bold white on red"

        spinner = (
            _SPINNER_FRAMES[self._frame_count % len(_SPINNER_FRAMES)]
            if is_new_step
            else "●"
        )

        throughput = self._ticks_per_second(tick_history)
        throughput_text = f"{throughput:.1f}/s" if throughput is not None else "—"

        tickers = tick.get("tickers")
        ticker_count = len(tickers) if isinstance(tickers, list) else 0

        halted_count = sum(1 for row in rows if row["is_halted"])
        bankrupt_count = sum(1 for row in rows if row["is_bankrupt"])

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_column(justify="left")
        grid.add_column(justify="right")

        grid.add_row(
            "[bold]Step[/bold]", _fmt_int(tick.get("step")),
            "[bold]Rollout[/bold]", _fmt_int(rollout.get("rollout", rollout.get("step"))),
        )
        grid.add_row(
            "[bold]Episode[/bold]", _fmt_int(rollout.get("episode")),
            "[bold]Symbols[/bold]", str(ticker_count),
        )
        grid.add_row(
            "[bold]Rate[/bold]", throughput_text,
            "[bold]Data age[/bold]", f"{age:.0f}s",
        )
        grid.add_row(
            "[bold]Runtime[/bold]", _fmt_uptime(now - self._start_time),
            "[bold]Uptime[/bold]", _fmt_uptime(age),
        )

        alert_line = ""
        if halted_count or bankrupt_count:
            bits = []
            if halted_count:
                bits.append(f"[bold white on purple4] {halted_count} HALTED [/bold white on purple4]")
            if bankrupt_count:
                bits.append(f"[bold white on red] {bankrupt_count} BANKRUPT [/bold white on red]")
            alert_line = "  " + "  ".join(bits)
        else:
            alert_line = "  [green]no risk alerts[/green]"

        body = Group(
            Text.from_markup(
                f"[bold]TRAINING DASHBOARD[/bold]   "
                f"[{health_style}] {health} [/{health_style}]  "
                f"[cyan]{spinner}[/cyan]  "
                f"[grey62]updated {age:.0f}s ago[/grey62]"
            ),
            grid,
            Text.from_markup(alert_line),
        )

        return Panel(
            body,
            title="1. Run Status",
            border_style="cyan",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 2: alerts -- risk conditions, named, or a clean bill of health
    # ------------------------------------------------------------------

    def _build_alerts_panel(self, rows: List[Dict[str, Any]]) -> Panel:
        halted = [row["ticker"] for row in rows if row["is_halted"]]
        bankrupt = [row["ticker"] for row in rows if row["is_bankrupt"]]
        deep_dd = [
            row["ticker"] for row in rows
            if not row["is_bankrupt"] and row["drawdown"] is not None
            and row["drawdown"] > _DD_ALERT_FRAC
        ]

        if not halted and not bankrupt and not deep_dd:
            return Panel(
                "[green]✓ No alerts — all streams healthy.[/green]",
                title="2. Alerts",
                border_style="green",
                expand=False,
            )

        lines: List[str] = []
        if halted:
            lines.append(f"[bold white on purple4] HALTED [/bold white on purple4]  {', '.join(map(str, halted))}")
        if bankrupt:
            lines.append(f"[bold white on red] BANKRUPT [/bold white on red]  {', '.join(map(str, bankrupt))}")
        if deep_dd:
            dd_rows = [
                f"{row['ticker']} [red]{_number(row['drawdown']):.1%}[/red]"
                for row in rows
                if row["ticker"] in deep_dd
            ]
            lines.append(f"[bold yellow]DEEP DRAWDOWN >{_DD_ALERT_FRAC:.0%}[/bold yellow]  {'  '.join(dd_rows)}")

        return Panel(
            "\n".join(lines),
            title="2. Alerts",
            border_style="red",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 3: training -- model numbers, correctly typed
    # ------------------------------------------------------------------

    def _build_training_panel(self, rollout: Dict[str, Any]) -> Panel:
        n_tickers = len(rollout.get("tickers") or [])
        kelly_zero = rollout.get("kelly_zero_count")
        kelly_warm = rollout.get("kelly_warm_count")
        # Red once any stream is locked (fractional_kelly == 0.0 for a warm
        # stream) -- see risk/kelly_sizing.py's diagnostics() docstring: that
        # 0.0 is permanent for the rest of the run once it happens, so this
        # count should only ever climb, never fall, within one run.
        kelly_zero_text = (
            "—" if kelly_zero is None
            else f"[bold red]{kelly_zero}/{n_tickers}[/bold red]" if kelly_zero > 0
            else f"{kelly_zero}/{n_tickers}"
        )
        kelly_warm_text = "—" if kelly_warm is None else f"{kelly_warm}/{n_tickers}"

        fields = [
            ("Reward", _fmt_signed(rollout.get("reward")), None),
            ("Reward EMA", _fmt_signed(rollout.get("reward_ema")), None),
            ("Entropy (direction)", _fmt(rollout.get("entropy_discrete")), None),
            ("Entropy (size/offset)", _fmt(rollout.get("entropy_continuous")), None),
            ("Policy loss", _fmt(rollout.get("policy_loss")), None),
            ("Value loss", _fmt(rollout.get("value_loss")), None),
            ("Clip fraction", _fmt(rollout.get("clip_frac")), None),
            ("Approx KL", _fmt(rollout.get("approx_kl")), None),
            ("Grad norm", _fmt(rollout.get("grad_norm")), None),
            ("Kelly locked (zero)", kelly_zero_text, None),
            ("Kelly warm", kelly_warm_text, None),
        ]

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_column(justify="left")
        grid.add_column(justify="right")

        row_pairs = [fields[i:i + 2] for i in range(0, len(fields), 2)]
        for pair in row_pairs:
            cells: List[str] = []
            for label, value, _color in pair:
                cells += [f"[bold]{label}[/bold]", value]
            grid.add_row(*cells)

        body = Group(
            grid,
            Text.from_markup("[grey62]reward is a differential-Sharpe signal (dimensionless), "
                             "not dollar PnL[/grey62]"),
        )

        return Panel(
            body,
            title="3. Training",
            border_style="blue",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 4: portfolio -- aggregate + context + top/bottom spotlight
    # ------------------------------------------------------------------

    def _build_portfolio_panel(
        self,
        tick: Dict[str, Any],
        rows: List[Dict[str, Any]],
    ) -> Panel:
        net_worth = tick.get("net_worth")
        n_streams = len(rows)

        net_worth_change_style = None
        current_net_worth = _number(net_worth)

        if current_net_worth is not None and self._previous_net_worth is not None:
            if current_net_worth > self._previous_net_worth:
                net_worth_change_style = "bright_green"
            elif current_net_worth < self._previous_net_worth:
                net_worth_change_style = "bright_red"

        if current_net_worth is not None:
            self._previous_net_worth = current_net_worth

        drawdown = tick.get("drawdown")
        trades_rollout = tick.get("trades_this_rollout")
        trades_total = tick.get("total_trades")

        open_positions = sum(
            _number(row["position"]) not in (None, 0)
            for row in rows
        )
        halted = sum(1 for row in rows if row["is_halted"])
        bankrupt = sum(1 for row in rows if row["is_bankrupt"])

        net_worth_text = _colored_money(
            net_worth,
            explicit_color=net_worth_change_style,
        )

        avg_per_stream = (
            current_net_worth / n_streams
            if current_net_worth is not None and n_streams
            else None
        )

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_row(
            "[bold]Net worth (Σ streams)[/bold]", net_worth_text,
            "[bold]Avg per stream[/bold]", _fmt_money(avg_per_stream),
        )
        grid.add_row(
            "[bold]Avg drawdown[/bold]", _colored_pct(drawdown, lower_is_better=True),
            "[bold]Trades this rollout[/bold]", _fmt_int(trades_rollout),
        )
        grid.add_row(
            "[bold]Total trades[/bold]", _fmt_int(trades_total),
            "[bold]Open positions[/bold]", str(open_positions),
        )
        grid.add_row(
            "[bold]Halted[/bold]", str(halted),
            "[bold]Bankrupt[/bold]", str(bankrupt),
        )

        components: List[Any] = [
            grid,
            Text.from_markup("[grey62]Aggregate = sum across all streams; each stream is an "
                             "independent simulated book. Watch direction, not the absolute number.[/grey62]"),
        ]
        spotlight = self._build_spotlight_table(rows)
        if spotlight is not None:
            components.append(spotlight)

        return Panel(
            Group(*components),
            title="4. Portfolio",
            border_style="green",
            expand=False,
        )

    def _build_spotlight_table(self, rows: List[Dict[str, Any]]) -> Optional[Table]:
        """Top and bottom `_SPOTLIGHT_ROWS/2` streams by unrealized PnL."""
        ranked = sorted(
            (row for row in rows if row["unrealized_value"] is not None),
            key=lambda row: row["unrealized_value"],
            reverse=True,
        )
        if not ranked:
            return None

        half = _SPOTLIGHT_ROWS // 2
        if len(ranked) <= _SPOTLIGHT_ROWS:
            spotlight_rows = ranked
        else:
            spotlight_rows = ranked[:half] + ranked[-half:]

        table = Table(title="Spotlight (unrealized PnL)", expand=False)
        table.add_column("Symbol")
        table.add_column("Price", justify="right")
        table.add_column("Pos", justify="right")
        table.add_column("Unrealized", justify="right")
        table.add_column("Drawdown", justify="right")
        table.add_column("Status")

        for row in spotlight_rows:
            status = (
                "[white on red]BANKRUPT[/white on red]"
                if row["is_bankrupt"]
                else "[white on purple4]HALTED[/white on purple4]"
                if row["is_halted"]
                else "[green]OK[/green]"
            )
            table.add_row(
                str(row["ticker"]),
                f"{row['price']:,.2f}" if row["price"] is not None else "—",
                _fmt(row["position"]),
                _colored_money(row["unrealized_value"], signed=True),
                (
                    _colored_pct(row["drawdown"], lower_is_better=True)
                    if row["drawdown"] is not None
                    else "—"
                ),
                status,
            )

        if len(ranked) > _SPOTLIGHT_ROWS:
            table.caption = (
                f"{len(ranked) - _SPOTLIGHT_ROWS} middle streams omitted "
                "(see the market grid for all of them)"
            )
        return table

    # ------------------------------------------------------------------
    # Section 5: market -- compact grid, movers, up/down counts
    # ------------------------------------------------------------------

    def _build_market_panel(self, rows: List[Dict[str, Any]]) -> Panel:
        if not rows:
            return Panel(
                "[grey62]No ticker price data is available in this record.[/grey62]",
                title="5. Market",
                border_style="grey42",
                expand=False,
            )

        grid = self._build_compact_grid(rows)

        movers = sorted(
            (row for row in rows
             if row["change_pct"] is not None and abs(row["change_pct"]) > 1e-9),
            key=lambda row: abs(row["change_pct"]),
            reverse=True,
        )[:_MOVERS_COUNT]

        up = sum(1 for row in rows if row["change_pct"] is not None and row["change_pct"] > 0)
        down = sum(1 for row in rows if row["change_pct"] is not None and row["change_pct"] < 0)
        flat = sum(1 for row in rows if row["change_pct"] is not None and row["change_pct"] == 0)
        unknown = len(rows) - up - down - flat

        summary = f"[bold green]{up} ▲[/bold green]  [bold red]{down} ▼[/bold red]  [grey62]{flat} •[/grey62]"
        if unknown:
            summary += f"  [grey42]{unknown} ?[/grey42]"

        movers_text = Text()
        if movers:
            movers_text.append("Movers  ")
            for i, row in enumerate(movers):
                color = "bright_green" if row["change_pct"] > 0 else "bright_red"
                arrow = "▲" if row["change_pct"] > 0 else "▼"
                movers_text.append(f"{arrow} {row['ticker']} ", style=color)
                movers_text.append(f"{row['change_pct']:+.2%}", style=color)
                if i < len(movers) - 1:
                    movers_text.append("   ")
        else:
            movers_text.append("no meaningful price moves this tick", style="grey62")

        body = Group(
            Text.from_markup(summary),
            grid,
            movers_text,
        )

        return Panel(
            body,
            title=f"5. Market — {len(rows)} symbols",
            border_style="magenta",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Section 6: fills/trades
    # ------------------------------------------------------------------

    def _build_fills_panel(
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
            title="6. Recent Fills",
            border_style="yellow",
            expand=False,
        )

    # ------------------------------------------------------------------
    # Row model
    # ------------------------------------------------------------------

    def _compute_per_ticker_rows(
        self,
        tick_record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tickers = tick_record.get("tickers")
        position = tick_record.get("position")
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
        halted = _as_list(
            tick_record.get("halted"),
            count,
        )

        rows: List[Dict[str, Any]] = []

        for index, ticker in enumerate(tickers):
            price = _number(prices[index])
            previous_price = self._previous_prices.get(str(ticker))

            change_pct: Optional[float] = None
            if price is not None and previous_price is not None and previous_price > 0:
                change_pct = (price - previous_price) / previous_price

            if price is not None:
                self._previous_prices[str(ticker)] = price

            nw = _number(net_worth[index])
            pnl = _number(unrealized[index])
            is_halted = bool(halted[index]) if halted[index] is not None else False
            is_bankrupt = nw is not None and nw <= 0

            rows.append({
                "ticker": ticker,
                "price": price,
                "change_pct": change_pct,
                "position": position[index],
                "net_worth": nw,
                "unrealized_value": pnl,
                "drawdown": _number(drawdown[index]),
                "is_halted": is_halted,
                "is_bankrupt": is_bankrupt,
            })

        return rows

    def _build_compact_grid(self, rows: List[Dict[str, Any]]) -> Table:
        column_width = 28
        columns = max(
            2,
            min(6, (self.console.width or 100) // column_width),
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

            change = row["change_pct"]
            if change is None:
                arrow, color = "", "grey62"
            elif change > 0:
                arrow, color = "▲", "bright_green"
            elif change < 0:
                arrow, color = "▼", "bright_red"
            else:
                arrow, color = "•", "grey62"

            pos = row["position"]
            try:
                pos_f = float(pos)
                pos_text = f"p:{pos_f:+.2f}"
            except (TypeError, ValueError):
                pos_text = "p:—"

            cells.append(
                f"{marker} [bold]{str(row['ticker']):<6}[/bold] "
                f"{price}[{color}]{arrow}[/{color}] {pos_text}"
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
