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

v3 (information architecture): the old layout buried the things you
actually care about under walls of numbers -- ~100 per-ticker rows, an
unreadable market tape, reward displayed as dollars, a dead "Sharpe" line,
and an entropy line that never rendered (training emits
entropy_discrete/continuous, not entropy).

v4 (visual identity): v3's six numbered sections were right about *what* to
show and wrong about how it read -- a stack of boxes that looked like a
debug dump. v4 keeps every number and re-skins it as a trading desk:
    MASTHEAD   -- brand, LIVE/STALE/NO DATA pill, session clock, pace
    TAPE       -- scrolling marquee of every symbol, price and tick change
    RISK       -- one band; "ALL CLEAR" when clean, unmissable when not
    THE BOOK   -- headline net liquidating value, equity sparkline,
                  aggregate stats, top/bottom leaderboard by unrealized PnL
    TELEMETRY  -- reward + trend sparkline, gauges for entropy/clip/KL,
                  losses, grad norm, Kelly locked/warm
    BLOTTER    -- recent fills as an execution blotter, newest first
    THE FLOOR  -- every symbol in a compact grid + breadth bar + movers
Layout is responsive: above _TWO_COLUMN_MIN_WIDTH the book sits beside
telemetry/blotter, below it everything stacks into one column.

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

from rich import box
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
# Terminal identity -- palette, glyphs, constants
# --------------------------------------------------------------------------
#
# v4 (visual identity): the information architecture of v3 was right, the
# presentation was a stack of six numbered boxes that read like a debug dump.
# v4 keeps every number and every guarantee (no Live(auto_refresh=True), no
# rendering imports in the training loop, JSONL only) and re-skins it as a
# trading desk: a masthead, a scrolling tape, a two-column body, a heat grid
# and an execution blotter, in one amber/cyan-on-black palette.

_C_AMBER = "#ffb000"        # primary brand / headline numbers
_C_AMBER_DIM = "#9a6b00"    # borders, rules, separators
_C_CYAN = "#38d6ff"         # secondary accent
_C_UP = "#00e07a"
_C_DOWN = "#ff4d5e"
_C_TEXT = "#dbe1ea"
_C_MUTE = "#6d7784"
_C_PANEL = "#0c0f14"        # panel background
_C_BAR = "#161b23"          # masthead / tape background

_BOX = box.SQUARE

_SPINNER_FRAMES = ["◜", "◝", "◞", "◟"]
_PULSE_FRAMES = ["●", "◉", "◎", "◉"]
_SPARK_CHARS = "▁▂▃▄▅▆▇█"

_MAX_TRADE_EVENTS = 9          # fills shown in the blotter
_SPOTLIGHT_ROWS = 10           # top/bottom rows in the P&L leaderboard
_MOVERS_COUNT = 4              # biggest movers called out under the heat grid
_DD_ALERT_FRAC = 0.25          # per-stream drawdown that raises an alert
_HEALTH_LIVE_S = 10
_HEALTH_STALE_S = 30
_TAPE_SCROLL_CHARS = 2         # tape characters advanced per rendered frame
_TWO_COLUMN_MIN_WIDTH = 150    # below this the body stacks into one column


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
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_money(value: Any, signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if signed:
        return f"${number:+,.2f}"
    return f"${number:,.2f}"


def _fmt_money_compact(value: Any) -> str:
    """Desk shorthand: $1.24B / $1.24M / $912.4K / $84.20."""
    number = _number(value)
    if number is None:
        return "—"
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:,.2f}M"
    if magnitude >= 1e4:
        return f"{sign}${magnitude / 1e3:,.1f}K"
    return f"{sign}${magnitude:,.2f}"


def _fmt_shares(value: Any) -> str:
    """Share counts are quantities, not rates -- two decimals, always signed."""
    number = _number(value)
    if number is None:
        return "—"
    if number == 0:
        return "·"
    return f"{number:+,.2f}"


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
        color = _C_UP if number >= 0 else _C_DOWN
    return f"[{color}]{_fmt_money(number, signed=signed)}[/{color}]"


def _colored_pct(value: Any, *, lower_is_better: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    if lower_is_better:
        color = _C_UP if number <= 0.10 else _C_DOWN
    else:
        color = _C_UP if number >= 0 else _C_DOWN
    return f"[{color}]{number:.2%}[/{color}]"


def _status_text(status: str, color: str = _C_MUTE) -> Text:
    return Text(status, style=color)


def _spaced(label: str, gap: str = " ") -> str:
    """`RISK` -> `R I S K`. Cheap way to make a header read as a masthead."""
    return gap.join(label)


def _sparkline(values: List[Any], width: int = 48) -> str:
    """Unicode block sparkline, bucket-averaged down to `width` columns."""
    series = [v for v in (_number(value) for value in values) if v is not None]
    if len(series) < 2:
        return ""

    width = max(4, int(width))

    if len(series) > width:
        bucket = len(series) / width
        resampled: List[float] = []
        for index in range(width):
            start = int(index * bucket)
            end = max(int((index + 1) * bucket), start + 1)
            chunk = series[start:end]
            resampled.append(sum(chunk) / len(chunk))
        series = resampled

    low = min(series)
    high = max(series)
    span = high - low

    if span <= 0:
        return _SPARK_CHARS[3] * len(series)

    last_index = len(_SPARK_CHARS) - 1
    return "".join(
        _SPARK_CHARS[min(last_index, int((value - low) / span * last_index + 0.5))]
        for value in series
    )


def _gauge(fraction: Optional[float], width: int = 12) -> str:
    """Solid/hollow bar for a 0..1 quantity."""
    if fraction is None:
        return "░" * width
    fraction = max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def _runs_to_text(chars: str, styles: List[str]) -> Text:
    """Collapse a per-character style list into a Text of styled runs."""
    text = Text()
    if not chars:
        return text
    start = 0
    for index in range(1, len(chars) + 1):
        if index == len(chars) or styles[index] != styles[start]:
            text.append(chars[start:index], style=styles[start])
            start = index
    return text


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

class TrainingDashboard:
    """
    Live trading-desk view of the training process.

    Design goals:
      1. Answer "is training alive?" before you have finished looking -- the
         status pill, the pulse, and a tape that visibly moves.
      2. Make risk impossible to miss (a dedicated risk band, not a badge
         buried in a table).
      3. Show aggregates, trends and spotlights, not 100 rows of numbers.
      4. Read like a trading terminal: one palette, one box style, headline
         numbers larger than the labels that name them.
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
        desk_name: str = "AUTONOMOUS TRADING DESK",
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
        self.desk_name = desk_name

        self._notebook_capable = _HAS_IPYTHON and _get_ipython() is not None
        self._widget_capable = self._notebook_capable and _HAS_IPYWIDGETS
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

        if console is not None:
            self.console = console
        elif self._is_tty:
            self.console = Console()
        else:
            # Notebook/headless output has no reliable terminal width and no
            # detectable color system -- force truecolor so the hex palette
            # above survives into Jupyter's ANSI-rendered stream output.
            self.console = Console(
                width=160,
                force_terminal=True,
                color_system="truecolor",
            )

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

    @property
    def width(self) -> int:
        return int(self.console.width or 120)

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

        # Every frame advances the animation counters -- the tape must keep
        # moving between steps, otherwise a slow trainer looks frozen.
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
            width=self.width,
            force_terminal=False,
            color_system="truecolor",
        )
        temp_console.print(renderable)
        return temp_console.export_html(
            inline_styles=True,
            code_format=(
                '<pre style="white-space:pre;overflow-x:auto;background:#05070a;'
                'padding:14px 16px;border-radius:6px;'
                "font-family:'SFMono-Regular','Menlo','Consolas',monospace;"
                'font-size:12px;line-height:1.25">{code}</pre>'
            ),
        )

    def _render_headless_inplace(self, renderable: Any) -> None:
        buffer = io.StringIO()
        temp_console = Console(
            file=buffer,
            force_terminal=True,
            color_system="truecolor",
            width=self.width,
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
        rollout_history = [
            record
            for record in history
            if record.get("record_type", "rollout") != "tick"
        ]

        rows = self._compute_per_ticker_rows(tick_record)

        book = self._build_book_panel(tick_record, tick_history, rows)
        telemetry = self._build_telemetry_panel(rollout_record, rollout_history)
        blotter = self._build_blotter_panel(tick_history)

        if self.width < _TWO_COLUMN_MIN_WIDTH:
            # Narrow terminals get one column -- side-by-side panels below
            # this width truncate the blotter's own columns into ellipses.
            body: Any = Group(book, telemetry, blotter)
        else:
            body = Table.grid(padding=(0, 1), expand=True)
            body.add_column(ratio=57)
            body.add_column(ratio=43)
            body.add_row(book, Group(telemetry, blotter))

        return Group(
            self._build_masthead(
                tick_record, rollout_record, tick_history, now, is_new_step,
            ),
            self._build_tape(rows),
            self._build_risk_band(rows),
            body,
            self._build_heat_panel(rows),
            self._build_footer(tick_record, now),
        )

    # ------------------------------------------------------------------
    # Masthead -- identity, health, pace
    # ------------------------------------------------------------------

    def _build_masthead(
        self,
        tick: Dict[str, Any],
        rollout: Dict[str, Any],
        tick_history: List[Dict[str, Any]],
        now: float,
        is_new_step: bool,
    ) -> Panel:
        latest_wall_time = _number(tick.get("wall_time")) or now
        age = max(0, now - latest_wall_time)

        if age < _HEALTH_LIVE_S:
            health, health_style = "LIVE", f"bold #04120b on {_C_UP}"
        elif age < _HEALTH_STALE_S:
            health, health_style = "STALE", f"bold #1a1200 on {_C_AMBER}"
        else:
            health, health_style = "NO DATA", f"bold #ffffff on {_C_DOWN}"

        pulse = (
            _PULSE_FRAMES[self._frame_count % len(_PULSE_FRAMES)]
            if is_new_step
            else "○"
        )
        spinner = _SPINNER_FRAMES[self._frame_count % len(_SPINNER_FRAMES)]
        pulse_color = _C_UP if age < _HEALTH_LIVE_S else _C_DOWN

        clock = time.strftime("%H:%M:%S", time.gmtime(now))

        header = Table.grid(expand=True, padding=(0, 1))
        header.add_column(justify="left")
        header.add_column(justify="center")
        header.add_column(justify="right")
        header.add_row(
            Text.from_markup(
                f"[bold {_C_AMBER}]█▛[/] [bold {_C_AMBER}]{_spaced('QUANTDESK')}[/]"
                f"  [{_C_AMBER_DIM}]│[/]  [{_C_CYAN}]{self.desk_name}[/]"
            ),
            Text.from_markup(
                f"[{health_style}] {health} [/]  "
                f"[{pulse_color}]{pulse}[/] [{_C_MUTE}]{spinner} tick +{age:.0f}s[/]"
            ),
            Text.from_markup(
                f"[{_C_MUTE}]SESSION[/] [bold {_C_TEXT}]{_fmt_uptime(now - self._start_time)}[/]"
                f"  [{_C_AMBER_DIM}]│[/]  [{_C_MUTE}]UTC[/] [bold {_C_TEXT}]{clock}[/]"
            ),
        )

        throughput = self._ticks_per_second(tick_history)
        throughput_text = f"{throughput:,.1f}/s" if throughput is not None else "—"

        tickers = tick.get("tickers")
        ticker_count = len(tickers) if isinstance(tickers, list) else 0

        stats = Table.grid(expand=True, padding=(0, 1))
        for _ in range(5):
            stats.add_column(justify="left", ratio=1)
        stats.add_row(
            self._stat("STEP", _fmt_int(tick.get("step")), _C_TEXT),
            self._stat("ROLLOUT", _fmt_int(rollout.get("rollout", rollout.get("step"))), _C_TEXT),
            self._stat("EPISODE", _fmt_int(rollout.get("episode")), _C_TEXT),
            self._stat("BOOKS", str(ticker_count), _C_CYAN),
            self._stat("THROUGHPUT", throughput_text, _C_CYAN),
        )

        return Panel(
            Group(header, Text(""), stats),
            box=_BOX,
            border_style=_C_AMBER_DIM,
            style=f"on {_C_BAR}",
            padding=(0, 1),
        )

    @staticmethod
    def _stat(label: str, value: str, color: str) -> Text:
        """Label above value, value larger in weight -- the desk's unit cell."""
        return Text.from_markup(f"[{_C_MUTE}]{label}[/]\n[bold {color}]{value}[/]")

    # ------------------------------------------------------------------
    # Tape -- the thing that makes it feel alive
    # ------------------------------------------------------------------

    def _build_tape(self, rows: List[Dict[str, Any]]) -> Panel:
        width = max(20, self.width - 4)

        if not rows:
            return Panel(
                Text("awaiting market data", style=_C_MUTE),
                box=_BOX,
                border_style=_C_AMBER_DIM,
                style=f"on {_C_BAR}",
                padding=(0, 1),
            )

        chars: List[str] = []
        styles: List[str] = []

        def push(fragment: str, style: str) -> None:
            chars.extend(fragment)
            styles.extend([style] * len(fragment))

        for row in rows:
            change = row["change_pct"]
            if change is None:
                arrow, color = "·", _C_MUTE
            elif change > 0:
                arrow, color = "▲", _C_UP
            elif change < 0:
                arrow, color = "▼", _C_DOWN
            else:
                arrow, color = "·", _C_MUTE

            price = f"{row['price']:,.2f}" if row["price"] is not None else "—"
            move = f"{change:+.2%}" if change is not None else "—"

            push(f"{row['ticker']} ", f"bold {_C_TEXT}")
            push(f"{price} ", _C_TEXT)
            push(f"{arrow}{move}", color)
            push("   ◆   ", _C_AMBER_DIM)

        if not chars:
            return Panel(
                Text("awaiting market data", style=_C_MUTE),
                box=_BOX,
                border_style=_C_AMBER_DIM,
                style=f"on {_C_BAR}",
                padding=(0, 1),
            )

        offset = (self._frame_count * _TAPE_SCROLL_CHARS) % len(chars)
        repeats = (width // len(chars)) + 2
        chars = chars * repeats
        styles = styles * repeats

        tape = _runs_to_text(
            "".join(chars[offset:offset + width]),
            styles[offset:offset + width],
        )

        return Panel(
            tape,
            box=_BOX,
            border_style=_C_AMBER_DIM,
            style=f"on {_C_BAR}",
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Risk band -- quiet when clean, unmissable when not
    # ------------------------------------------------------------------

    def _build_risk_band(self, rows: List[Dict[str, Any]]) -> Panel:
        halted = [row["ticker"] for row in rows if row["is_halted"]]
        bankrupt = [row["ticker"] for row in rows if row["is_bankrupt"]]
        deep_dd = [
            row for row in rows
            if not row["is_bankrupt"] and row["drawdown"] is not None
            and row["drawdown"] > _DD_ALERT_FRAC
        ]

        title = f"[bold {_C_AMBER}]▌ {_spaced('RISK')}[/]"

        if not halted and not bankrupt and not deep_dd:
            return Panel(
                Text.from_markup(
                    f"[{_C_UP}]✓ ALL CLEAR[/]  [{_C_MUTE}]no halts, no blown books, "
                    f"nothing beyond {_DD_ALERT_FRAC:.0%} drawdown[/]"
                ),
                title=title,
                title_align="left",
                box=_BOX,
                border_style="#1f4d36",
                style=f"on {_C_PANEL}",
                padding=(0, 1),
            )

        lines: List[str] = []
        if bankrupt:
            lines.append(
                f"[bold #ffffff on {_C_DOWN}] BLOWN [/]  "
                f"[bold {_C_DOWN}]{', '.join(map(str, bankrupt))}[/]"
            )
        if halted:
            lines.append(
                f"[bold #1a1200 on {_C_AMBER}] HALTED [/]  "
                f"[bold {_C_AMBER}]{', '.join(map(str, halted))}[/]"
            )
        if deep_dd:
            rendered = "  ".join(
                f"[bold {_C_TEXT}]{row['ticker']}[/] [{_C_DOWN}]{row['drawdown']:.1%}[/]"
                for row in sorted(deep_dd, key=lambda r: -r["drawdown"])[:12]
            )
            lines.append(
                f"[bold #ffffff on {_C_DOWN}] DRAWDOWN >{_DD_ALERT_FRAC:.0%} [/]  {rendered}"
            )

        return Panel(
            Text.from_markup("\n".join(lines)),
            title=title,
            title_align="left",
            box=_BOX,
            border_style=_C_DOWN,
            style=f"on {_C_PANEL}",
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # The book -- headline equity, trend, leaderboard
    # ------------------------------------------------------------------

    def _build_book_panel(
        self,
        tick: Dict[str, Any],
        tick_history: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
    ) -> Panel:
        net_worth = _number(tick.get("net_worth"))
        n_streams = len(rows)

        delta = None
        if net_worth is not None and self._previous_net_worth is not None:
            delta = net_worth - self._previous_net_worth

        equity_series = [record.get("net_worth") for record in tick_history]
        opening = next(
            (value for value in (_number(v) for v in equity_series) if value is not None),
            None,
        )
        session_change = (
            (net_worth - opening) / opening
            if net_worth is not None and opening not in (None, 0)
            else None
        )

        if delta is None or delta == 0:
            headline_color, tick_mark = _C_AMBER, "·"
        elif delta > 0:
            headline_color, tick_mark = _C_UP, "▲"
        else:
            headline_color, tick_mark = _C_DOWN, "▼"

        if net_worth is not None:
            self._previous_net_worth = net_worth

        spark = _sparkline(equity_series, max(18, int(self.width * 0.57) - 32))
        spark_color = (
            _C_MUTE if session_change is None
            else _C_UP if session_change >= 0
            else _C_DOWN
        )

        delta_text = _fmt_money(delta, signed=True) if delta is not None else "—"
        session_text = f"{session_change:+.2%}" if session_change is not None else "—"

        headline = Table.grid(padding=(0, 3))
        headline.add_column(justify="left")
        headline.add_column(justify="left")
        headline.add_row(
            Text.from_markup(
                f"[{_C_MUTE}]NET LIQUIDATING VALUE[/]\n"
                f"[bold {headline_color}]{_fmt_money(net_worth)}[/] "
                f"[{headline_color}]{tick_mark}[/]"
            ),
            Text.from_markup(
                f"[{_C_MUTE}]LAST TICK[/]      [{_C_MUTE}]SESSION[/]\n"
                f"[bold {headline_color}]{delta_text:>13}[/]  "
                f"[bold {spark_color}]{session_text:>8}[/]"
            ),
        )

        drawdown = _number(tick.get("drawdown"))
        open_positions = sum(
            _number(row["position"]) not in (None, 0)
            for row in rows
        )
        avg_per_stream = (
            net_worth / n_streams
            if net_worth is not None and n_streams
            else None
        )

        stats = Table.grid(expand=True, padding=(0, 1))
        for _ in range(4):
            stats.add_column(justify="left", ratio=1)
        stats.add_row(
            self._stat("AVG / BOOK", _fmt_money_compact(avg_per_stream), _C_TEXT),
            self._stat(
                "AVG DRAWDOWN",
                f"{drawdown:.2%}" if drawdown is not None else "—",
                _C_DOWN if (drawdown or 0) > 0.10 else _C_TEXT,
            ),
            self._stat("OPEN POS", f"{open_positions}/{n_streams}", _C_CYAN),
            self._stat("FILLS · TOTAL", _fmt_int(tick.get("total_trades")), _C_TEXT),
        )
        stats.add_row(Text(""), Text(""), Text(""), Text(""))
        stats.add_row(
            self._stat("FILLS · ROLLOUT", _fmt_int(tick.get("trades_this_rollout")), _C_TEXT),
            self._stat("HALTED", str(sum(1 for row in rows if row["is_halted"])), _C_AMBER),
            self._stat("BLOWN", str(sum(1 for row in rows if row["is_bankrupt"])), _C_DOWN),
            self._stat("BOOKS", str(n_streams), _C_TEXT),
        )

        components: List[Any] = [headline]

        if spark:
            components.append(
                Text.from_markup(f"[{_C_MUTE}]EQUITY [/][{spark_color}]{spark}[/]")
            )

        components.append(Text(""))
        components.append(stats)

        leaderboard = self._build_spotlight_table(rows)
        if leaderboard is not None:
            components.append(Text(""))
            components.append(leaderboard)

        return Panel(
            Group(*components),
            title=f"[bold {_C_AMBER}]▌ {_spaced('THE BOOK')}[/]",
            title_align="left",
            subtitle=(
                f"[{_C_MUTE}]Σ over {n_streams} independent simulated books[/]"
                if self.width >= 100
                else f"[{_C_MUTE}]Σ over {n_streams} books[/]"
            ),
            subtitle_align="right",
            box=_BOX,
            border_style=_C_AMBER_DIM,
            style=f"on {_C_PANEL}",
            padding=(0, 1),
        )

    def _build_spotlight_table(self, rows: List[Dict[str, Any]]) -> Optional[Table]:
        """Top and bottom `_SPOTLIGHT_ROWS/2` books by unrealized PnL."""
        ranked = sorted(
            (row for row in rows if row["unrealized_value"] is not None),
            key=lambda row: row["unrealized_value"],
            reverse=True,
        )
        if not ranked:
            return None

        half = _SPOTLIGHT_ROWS // 2
        if len(ranked) <= _SPOTLIGHT_ROWS:
            spotlight_rows: List[Optional[Dict[str, Any]]] = list(ranked)
        else:
            spotlight_rows = ranked[:half] + [None] + ranked[-half:]

        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            pad_edge=False,
            padding=(0, 1),
            header_style=_C_MUTE,
            border_style="#232a34",
        )
        table.add_column("SYMBOL", style=f"bold {_C_TEXT}")
        table.add_column("LAST", justify="right")
        table.add_column("POS", justify="right")
        table.add_column("UNREALIZED", justify="right")
        table.add_column("DD", justify="right")
        table.add_column("", justify="center", width=1)

        for row in spotlight_rows:
            if row is None:
                omitted = len(ranked) - _SPOTLIGHT_ROWS
                table.add_row(
                    Text.from_markup(f"[{_C_MUTE}]⋯[/]"),
                    "",
                    "",
                    Text.from_markup(f"[{_C_MUTE}]{omitted} books between[/]"),
                    "",
                    "",
                )
                continue

            flag = (
                f"[{_C_DOWN}]✖[/]" if row["is_bankrupt"]
                else f"[{_C_AMBER}]‖[/]" if row["is_halted"]
                else f"[{_C_UP}]✓[/]"
            )
            table.add_row(
                str(row["ticker"]),
                f"{row['price']:,.2f}" if row["price"] is not None else "—",
                _fmt_shares(row["position"]),
                _colored_money(row["unrealized_value"], signed=True),
                (
                    _colored_pct(row["drawdown"], lower_is_better=True)
                    if row["drawdown"] is not None
                    else "—"
                ),
                Text.from_markup(flag),
            )

        return table

    # ------------------------------------------------------------------
    # Telemetry -- the model's vitals
    # ------------------------------------------------------------------

    def _build_telemetry_panel(
        self,
        rollout: Dict[str, Any],
        rollout_history: List[Dict[str, Any]],
    ) -> Panel:
        n_tickers = len(rollout.get("tickers") or [])
        kelly_zero = rollout.get("kelly_zero_count")
        kelly_warm = rollout.get("kelly_warm_count")
        # Red once any stream is locked (fractional_kelly == 0.0 for a warm
        # stream) -- see risk/kelly_sizing.py's diagnostics() docstring: that
        # 0.0 is permanent for the rest of the run once it happens, so this
        # count should only ever climb, never fall, within one run.
        kelly_zero_text = (
            "—" if kelly_zero is None
            else f"[bold {_C_DOWN}]{kelly_zero}/{n_tickers}[/]" if kelly_zero > 0
            else f"[bold {_C_TEXT}]{kelly_zero}/{n_tickers}[/]"
        )
        kelly_warm_text = (
            "—" if kelly_warm is None
            else f"[bold {_C_TEXT}]{kelly_warm}/{n_tickers}[/]"
        )

        reward = _number(rollout.get("reward"))
        reward_ema = _number(rollout.get("reward_ema"))
        reward_color = _C_MUTE if reward is None else _C_UP if reward >= 0 else _C_DOWN
        ema_color = _C_MUTE if reward_ema is None else _C_UP if reward_ema >= 0 else _C_DOWN

        reward_spark = _sparkline(
            [record.get("reward") for record in rollout_history],
            max(14, int(self.width * 0.43) - 26),
        )

        components: List[Any] = [
            Text.from_markup(
                f"[{_C_MUTE}]REWARD  differential Sharpe[/]\n"
                f"[bold {reward_color}]{_fmt_signed(reward, 5)}[/]   "
                f"[{_C_MUTE}]EMA[/] [bold {ema_color}]{_fmt_signed(reward_ema, 5)}[/]"
            )
        ]

        if reward_spark:
            components.append(
                Text.from_markup(f"[{_C_MUTE}]TREND  [/][{ema_color}]{reward_spark}[/]")
            )
        components.append(Text(""))

        # Bounded-ish quantities get a gauge against a nominal ceiling; the
        # ceiling is a display scale only, never a threshold the model sees.
        gauges = [
            ("ENTROPY · dir", rollout.get("entropy_discrete"), 1.6),
            ("ENTROPY · size", rollout.get("entropy_continuous"), 3.0),
            ("CLIP FRACTION", rollout.get("clip_frac"), 0.40),
            ("APPROX KL", rollout.get("approx_kl"), 0.05),
        ]

        gauge_grid = Table.grid(expand=True, padding=(0, 1))
        gauge_grid.add_column(justify="left", width=16)
        gauge_grid.add_column(justify="left", width=12)
        gauge_grid.add_column(justify="right", width=10)
        gauge_grid.add_column(ratio=1)   # spacer: absorbs the panel's slack

        for label, raw, ceiling in gauges:
            value = _number(raw)
            fraction = None if value is None else max(0.0, min(1.0, value / ceiling))
            if value is None:
                color = _C_MUTE
            elif fraction is not None and fraction > 0.85:
                color = _C_DOWN
            else:
                color = _C_CYAN
            gauge_grid.add_row(
                Text.from_markup(f"[{_C_MUTE}]{label}[/]"),
                Text.from_markup(f"[{color}]{_gauge(fraction, 12)}[/]"),
                Text.from_markup(f"[bold {_C_TEXT}]{_fmt(raw)}[/]"),
                "",
            )

        components.append(gauge_grid)
        components.append(Text(""))

        numeric = Table.grid(expand=True, padding=(0, 1))
        numeric.add_column(justify="left", width=20)
        numeric.add_column(justify="left", width=21)
        numeric.add_column(ratio=1)      # spacer: keeps the pairs left-packed
        numeric.add_row(
            self._stat("POLICY LOSS", _fmt(rollout.get("policy_loss")), _C_TEXT),
            self._stat("VALUE LOSS", _fmt(rollout.get("value_loss")), _C_TEXT),
            "",
        )
        numeric.add_row(Text(""), Text(""), "")
        numeric.add_row(
            self._stat("GRAD NORM", _fmt(rollout.get("grad_norm")), _C_TEXT),
            Text.from_markup(
                f"[{_C_MUTE}]KELLY  locked · warm[/]\n"
                f"{kelly_zero_text}  [{_C_MUTE}]·[/]  {kelly_warm_text}"
            ),
            "",
        )
        components.append(numeric)

        return Panel(
            Group(*components),
            title=f"[bold {_C_AMBER}]▌ {_spaced('TELEMETRY')}[/]",
            title_align="left",
            subtitle=(
                f"[{_C_MUTE}]reward is dimensionless, not dollar PnL[/]"
                if self.width >= 100
                else f"[{_C_MUTE}]dimensionless[/]"
            ),
            subtitle_align="right",
            box=_BOX,
            border_style=_C_AMBER_DIM,
            style=f"on {_C_PANEL}",
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # The floor -- every symbol at a glance
    # ------------------------------------------------------------------

    def _build_heat_panel(self, rows: List[Dict[str, Any]]) -> Panel:
        title = f"[bold {_C_AMBER}]▌ {_spaced('THE FLOOR')}[/]"

        if not rows:
            return Panel(
                Text("no ticker price data in this record", style=_C_MUTE),
                title=title,
                title_align="left",
                box=_BOX,
                border_style=_C_AMBER_DIM,
                style=f"on {_C_PANEL}",
                padding=(0, 1),
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

        breadth_width = 24
        up_cells = int(round(up / max(1, up + down) * breadth_width)) if (up + down) else 0
        breadth = (
            f"[{_C_UP}]{'█' * up_cells}[/]"
            f"[{_C_DOWN}]{'█' * (breadth_width - up_cells)}[/]"
            if (up + down) else f"[{_C_MUTE}]{'░' * breadth_width}[/]"
        )

        summary = Text.from_markup(
            f"[bold {_C_UP}]{up:>3} ▲[/]   [bold {_C_DOWN}]{down:>3} ▼[/]   "
            f"[{_C_MUTE}]{flat:>3} ·[/]"
            + (f"   [{_C_MUTE}]{unknown:>3} ?[/]" if unknown else "")
            + f"    {breadth}  [{_C_MUTE}]breadth[/]"
        )

        movers_text = Text()
        if movers:
            movers_text.append("MOVERS   ", style=_C_MUTE)
            for index, row in enumerate(movers):
                color = _C_UP if row["change_pct"] > 0 else _C_DOWN
                arrow = "▲" if row["change_pct"] > 0 else "▼"
                movers_text.append(f"{arrow} {row['ticker']} ", style=f"bold {color}")
                movers_text.append(f"{row['change_pct']:+.2%}", style=color)
                if index < len(movers) - 1:
                    movers_text.append("   ◆   ", style=_C_AMBER_DIM)
        else:
            movers_text.append("flat tape — no meaningful moves this tick", style=_C_MUTE)

        return Panel(
            Group(summary, Text(""), grid, Text(""), movers_text),
            title=title,
            title_align="left",
            subtitle=f"[{_C_MUTE}]{len(rows)} symbols[/]",
            subtitle_align="right",
            box=_BOX,
            border_style=_C_AMBER_DIM,
            style=f"on {_C_PANEL}",
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Blotter -- what actually got executed
    # ------------------------------------------------------------------

    def _build_blotter_panel(
        self,
        tick_history: List[Dict[str, Any]],
    ) -> Panel:
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=False,
            pad_edge=False,
            padding=(0, 1),
            header_style=_C_MUTE,
            border_style="#232a34",
        )
        table.add_column("STEP", justify="right", style=_C_MUTE)
        table.add_column("", justify="left", width=1)
        table.add_column("SIDE", width=4)
        table.add_column("SYMBOL", style=f"bold {_C_TEXT}")
        table.add_column("QTY", justify="right")
        table.add_column("PRICE", justify="right")
        table.add_column("NOTIONAL", justify="right")

        count = 0

        for record in reversed(tick_history):
            if count >= _MAX_TRADE_EVENTS:
                break

            tickers = record.get("tickers")
            filled = record.get("filled_qty_this_tick")
            prices = record.get("price_per_ticker")

            if not isinstance(tickers, list) or not isinstance(filled, list):
                continue

            step = record.get("step")

            for index, quantity in enumerate(filled):
                if count >= _MAX_TRADE_EVENTS:
                    break

                quantity_number = _number(quantity)
                if quantity_number is None or quantity_number == 0:
                    continue

                ticker = tickers[index] if index < len(tickers) else "?"
                price = (
                    _number(prices[index])
                    if isinstance(prices, list) and index < len(prices)
                    else None
                )

                side, color, arrow = (
                    ("BUY", _C_UP, "▲") if quantity_number > 0
                    else ("SELL", _C_DOWN, "▼")
                )
                notional = abs(quantity_number) * price if price is not None else None

                table.add_row(
                    _fmt_int(step),
                    Text.from_markup(f"[{color}]{arrow}[/]"),
                    Text.from_markup(f"[bold {color}]{side}[/]"),
                    str(ticker),
                    f"{abs(quantity_number):,.2f}",
                    f"${price:,.2f}" if price is not None else "—",
                    _fmt_money_compact(notional),
                )
                count += 1

        body: Any = table
        if count == 0:
            body = Text("no fills in the visible history", style=_C_MUTE)

        return Panel(
            body,
            title=f"[bold {_C_AMBER}]▌ {_spaced('BLOTTER')}[/]",
            title_align="left",
            subtitle=f"[{_C_MUTE}]most recent first[/]",
            subtitle_align="right",
            box=_BOX,
            border_style=_C_AMBER_DIM,
            style=f"on {_C_PANEL}",
            padding=(0, 1),
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def _build_footer(self, tick: Dict[str, Any], now: float) -> Text:
        age = max(0, now - (_number(tick.get("wall_time")) or now))
        separator = f"   [{_C_AMBER_DIM}]│[/]   "

        parts = [
            f"[{_C_MUTE}]SIMULATED BOOKS · NO CAPITAL AT RISK[/]",
            f"[{_C_UP}]✓ live[/]  [{_C_AMBER}]‖ halted[/]  [{_C_DOWN}]✖ blown[/]",
        ]
        if self.width >= 130:
            parts.append(
                f"[{_C_MUTE}]frame {self._frame_count} · data +{age:.0f}s · "
                f"window {self.history_window}[/]"
            )

        return Text.from_markup(separator.join(parts))

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

        net_worth = _as_list(tick_record.get("net_worth_per_ticker"), count)
        prices = _as_list(tick_record.get("price_per_ticker"), count)
        unrealized = _as_list(tick_record.get("unrealized_pnl"), count)
        drawdown = _as_list(tick_record.get("drawdown_per_ticker"), count)
        halted = _as_list(tick_record.get("halted"), count)

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
        column_width = 26
        columns = max(2, min(6, (self.width - 4) // column_width))

        grid = Table.grid(padding=(0, 1), expand=True)
        for _ in range(columns):
            grid.add_column(ratio=1)

        cells: List[Text] = []

        for row in rows:
            change = row["change_pct"]
            if change is None:
                arrow, color = " ", _C_MUTE
            elif change > 0:
                arrow, color = "▲", _C_UP
            elif change < 0:
                arrow, color = "▼", _C_DOWN
            else:
                arrow, color = "·", _C_MUTE

            if row["is_bankrupt"]:
                marker, marker_color = "✖", _C_DOWN
            elif row["is_halted"]:
                marker, marker_color = "‖", _C_AMBER
            else:
                marker, marker_color = "▏", color

            price = f"{row['price']:,.2f}" if row["price"] is not None else "—"

            position = _number(row["position"])
            if position is None or position == 0:
                position_text, position_color = "·", _C_MUTE
            else:
                position_text = f"{position:+.2f}"
                position_color = _C_UP if position > 0 else _C_DOWN

            cells.append(
                Text.from_markup(
                    f"[{marker_color}]{marker}[/]"
                    f"[bold {_C_TEXT}]{str(row['ticker'])[:6]:<6}[/] "
                    f"[{_C_TEXT}]{price:>8}[/]"
                    f"[{color}]{arrow}[/]"
                    f"[{position_color}]{position_text:>7}[/]"
                )
            )

            if len(cells) == columns:
                grid.add_row(*cells)
                cells = []

        if cells:
            cells.extend([Text("")] * (columns - len(cells)))
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
