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
from typing import Any, Dict, List, Optional

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


# --------------------------------------------------------------------------
# Metrics reader
# --------------------------------------------------------------------------

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
        cfg_mode = DisplayMode(cfg_display_mode)
    except ValueError:
        cfg_mode = DisplayMode.AUTO
    return resolve_display_mode(cfg_mode)


_SPINNER_FRAMES = ["\u28f7", "\u28ef", "\u28df", "\u287f", "\u28bf", "\u28fb", "\u28fd", "\u28fe"]  # braille spinner


def _fmt_uptime(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

class TrainingDashboard:
    """
    Renders structured metrics written by MetricsWriter.

    Rendering path, chosen once at construction:
        notebook + ipywidgets installed:  a persistent Output widget,
                                           .clear_output(wait=True) each
                                           frame -- the robust in-place
                                           update for a background-thread
                                           poller (see module docstring).
        notebook, no ipywidgets:          IPython.display.clear_output +
                                           display(), same-cell in-place.
        real terminal (isatty), no notebook: Rich Live(auto_refresh=False).
        headless, no tty, no notebook:    manual ANSI cursor-up rewrite --
                                           genuinely redraws in place
                                           instead of appending a new frame
                                           to scrollback every call.
    """

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

        self._notebook_capable = _HAS_IPYTHON and _get_ipython() is not None
        self._widget_capable = self._notebook_capable and _HAS_IPYWIDGETS
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

        self._live: Optional[Live] = None
        self._output_widget = None
        if self._widget_capable:
            self._output_widget = _ipywidgets.Output()
        elif self.mode == DisplayMode.LOCAL and self._is_tty and not self._notebook_capable:
            self._live = Live(console=self.console, auto_refresh=False, transient=False)

        self._displayed_widget = False
        self._headless_lines_printed = 0  # for the manual ANSI rewrite path

        self._start_time = time.time()
        self._last_render_time: Optional[float] = None
        self._last_step: Optional[int] = None
        self._frame_count = 0

        # trend state, all keyed by ticker (or None for the portfolio total)
        self._prev_net_worth: Optional[float] = None
        self._prev_net_worth_per_ticker: Dict[str, float] = {}
        self._prev_trades_per_ticker: Dict[str, int] = {}

    def start(self) -> None:
        if self._output_widget is not None and not self._displayed_widget:
            _ipy_display(self._output_widget)
            self._displayed_widget = True
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

        # "is this actually new data, or the same last line as before" --
        # drives both the spinner (only animates on real progress) and the
        # rollouts/sec throughput estimate below.
        step = latest.get("step")
        now = time.time()
        is_new_step = step != self._last_step
        if is_new_step:
            self._frame_count += 1
        self._last_step = step
        self._last_render_time = now

        renderable = self._build_renderable(latest, history, now, is_new_step)

        if self._output_widget is not None:
            self.start()
            with self._output_widget:
                self._output_widget.clear_output(wait=True)
                self.console.print(renderable)
        elif self._live is not None:
            self._live.update(renderable)
            self._live.refresh()
        elif self._notebook_capable:
            html = self._render_html(renderable)
            _ipy_clear_output(wait=True)
            _ipy_display(_IPyHTML(html))
        else:
            self._render_headless_inplace(renderable)

    def _render_html(self, renderable) -> str:
        buf = io.StringIO()
        tmp_console = Console(file=buf, record=True, width=100, force_terminal=False)
        tmp_console.print(renderable)
        return tmp_console.export_html(
            inline_styles=True,
            code_format='<pre style="white-space:pre-wrap;font-family:monospace">{code}</pre>',
        )

    def _render_headless_inplace(self, renderable) -> None:
        """
        No tty, no notebook -- there's no ANSI Live()-style redraw available
        without risking the exact race v1 hit, but a PLAIN scrolling print
        (the old behavior) reads as "frozen, just reprinting itself" even
        when it's actually updating, because nothing visually distinguishes
        one frame from the next in a long log. This moves the cursor back
        up over the previous frame's lines and overwrites them in place --
        a single-threaded, synchronous ANSI move (cursor-up + clear-line),
        not Live()'s background-thread redraw, so it doesn't reintroduce the
        v1 race.
        """
        buf = io.StringIO()
        tmp_console = Console(file=buf, force_terminal=True, width=self.console.width or 100)
        tmp_console.print(renderable)
        text = buf.getvalue()
        n_lines = text.count("\n")

        if self._headless_lines_printed > 0:
            sys.stdout.write(f"\x1b[{self._headless_lines_printed}A")  # cursor up
            sys.stdout.write("\x1b[J")  # clear from cursor to end of screen
        sys.stdout.write(text)
        sys.stdout.flush()
        self._headless_lines_printed = n_lines

    def run_polling_loop(self, poll_interval_seconds: float = 2.0, max_iterations: Optional[int] = None) -> None:
        i = 0
        try:
            self.start()
            while max_iterations is None or i < max_iterations:
                self.render_once()
                time.sleep(poll_interval_seconds)
                i += 1
        finally:
            self.stop()

    # ----------------------------------------------------------------

    def _build_renderable(
        self, latest: Dict[str, Any], history: List[Dict[str, Any]], now: float, is_new_step: bool
    ) -> Group:
        reward = latest.get("reward")
        reward_ema = latest.get("reward_ema")
        drawdown = latest.get("drawdown")
        net_worth = latest.get("net_worth")

        net_worth_color = None
        if net_worth is not None and self._prev_net_worth is not None:
            if net_worth > self._prev_net_worth:
                net_worth_color = "green"
            elif net_worth < self._prev_net_worth:
                net_worth_color = "red"
        if net_worth is not None:
            self._prev_net_worth = net_worth

        # throughput -- rollouts/sec over the visible history window, a
        # cheap "is this actually moving" signal independent of any single
        # metric's own trend
        rollouts_per_sec = None
        if len(history) >= 2:
            span_steps = (history[-1].get("step") or 0) - (history[0].get("step") or 0)
            span_time = (history[-1].get("wall_time") or now) - (history[0].get("wall_time") or now)
            if span_time > 0 and span_steps > 0:
                rollouts_per_sec = span_steps / span_time

        spinner = _SPINNER_FRAMES[self._frame_count % len(_SPINNER_FRAMES)] if is_new_step else "\u25cf"
        spinner_color = "green" if is_new_step else "yellow"
        staleness = now - (latest.get("wall_time") or now)
        live_note = "" if staleness < 30 else f"  [red](no new data for {int(staleness)}s -- check the training process)[/red]"

        panel_lines = [
            f"[{spinner_color}]{spinner}[/{spinner_color}] live  |  uptime {_fmt_uptime(now - self._start_time)}"
            f"  |  {rollouts_per_sec:.2f} rollouts/s" if rollouts_per_sec is not None else
            f"[{spinner_color}]{spinner}[/{spinner_color}] live  |  uptime {_fmt_uptime(now - self._start_time)}",
            f"step:              {latest.get('step')}{live_note}",
            f"episode:           {latest.get('episode')}",
            f"reward:            {_fmt_colored(reward, good_is='positive')}",
            f"reward (EMA):      {_fmt_colored(reward_ema, good_is='positive')}",
            f"net worth (total): {_fmt_colored(net_worth, good_is=None, color=net_worth_color, dollar=True)}",
            f"sharpe:            {_fmt(latest.get('sharpe'))}",
            f"drawdown (avg):    {_fmt_colored(drawdown, good_is='small_pct', pct=True)}",
            f"trades (rollout):  {_fmt_int(latest.get('trades_this_rollout'))}",
            f"trades (total):    {_fmt_int(latest.get('total_trades'))}",
        ]
        header = Panel("\n".join(panel_lines), title="Training Status", expand=False)

        table = Table(title=f"Per-Environment State  (frame #{self._frame_count})")
        table.add_column("Ticker")
        table.add_column("Position", justify="right")
        table.add_column("Net Worth", justify="right")
        table.add_column("Unrealized PnL", justify="right")
        table.add_column("Drawdown", justify="right")
        table.add_column("Trades (rollout)", justify="right")
        table.add_column("Trades (total)", justify="right")

        tickers = latest.get("tickers")
        position = latest.get("position")

        if isinstance(tickers, list) and isinstance(position, list):
            n = len(tickers)
            net_worth_per_ticker = _as_list(latest.get("net_worth_per_ticker"), n)
            unrealized = _as_list(latest.get("unrealized_pnl"), n)
            drawdown_per_ticker = _as_list(latest.get("drawdown_per_ticker"), n)
            trades_rollout_per_ticker = _as_list(latest.get("trades_per_ticker_this_rollout"), n)
            trades_total_per_ticker = _as_list(latest.get("total_trades_per_ticker"), n)

            for i, ticker in enumerate(tickers):
                nw = net_worth_per_ticker[i]
                nw_color = None
                if nw is not None:
                    prev = self._prev_net_worth_per_ticker.get(ticker)
                    if prev is not None:
                        if nw > prev:
                            nw_color = "green"
                        elif nw < prev:
                            nw_color = "red"
                    self._prev_net_worth_per_ticker[ticker] = nw
                nw_text = _fmt_colored(nw, good_is=None, color=nw_color, dollar=True)
                if nw_color == "green":
                    nw_text = "\u25b2 " + nw_text
                elif nw_color == "red":
                    nw_text = "\u25bc " + nw_text

                trades_this = trades_rollout_per_ticker[i]
                trades_total = trades_total_per_ticker[i]
                # highlight (bold cyan) any env that actually traded THIS
                # rollout, so per-env activity is visible at a glance
                # instead of having to compare numbers frame to frame.
                just_traded = trades_this is not None and trades_this not in (0, "0")
                trades_this_text = f"[bold cyan]{_fmt_int(trades_this)}[/bold cyan]" if just_traded else _fmt_int(trades_this)

                table.add_row(
                    str(ticker),
                    _fmt(position[i]),
                    nw_text,
                    _fmt_colored(unrealized[i], good_is="positive"),
                    _fmt_colored(drawdown_per_ticker[i], good_is="small_pct", pct=True),
                    trades_this_text,
                    _fmt_int(trades_total),
                )
        else:
            table.add_row("(no per-ticker breakdown in this record)", "", "", "", "", "", "")

        return Group(header, table)


def _as_list(value: Any, n: int) -> List[Any]:
    return value if isinstance(value, list) and len(value) == n else [None] * n


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
    """
    good_is="positive"  -> green if value >= 0, red if value < 0
    good_is="small_pct" -> green if value <= 0.10 (10%), red otherwise
    good_is=None         -> no automatic rule; only an explicit `color` applies
    `color`, when given, always wins over the good_is rule.
    """
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