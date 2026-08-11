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

    def log(self, step: int, fsync: bool = True, **metrics: Any) -> None:
        """
        fsync=True (default) is what every rollout-level log() call should
        keep using -- one call per ~256 env-steps, fsync cost is
        negligible. Tick-level logging (once per env-step -- see
        train.py's per-tick MetricsWriter usage) calls this with
        fsync=False: flush() still happens (so a concurrent reader sees the
        line immediately -- MetricsReader.tail() does a plain file read,
        not an OS-buffered one), but the disk-sync syscall is skipped,
        since paying an fsync 256x per rollout instead of once is a real,
        avoidable cost for data that's inherently disposable (a lost tick
        record on a crash is nothing like a lost checkpoint).
        """
        record = {"step": step, "wall_time": time.time(), **metrics}
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        if self.flush_every_call:
            self._fh.flush()
            if fsync:
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
        """
        Reads only the last ~n lines, not the whole file. The previous
        implementation did `f.readlines()` on the ENTIRE file every single
        call -- fine for a short run, but with tick-level logging now
        writing far more lines than the old rollout-only cadence, the file
        keeps growing for the whole training run, and re-reading all of it
        every poll gets progressively slower as it grows -- exactly the
        kind of bug that looks like "updates aren't frequent enough" and
        gets WORSE the longer training runs, not better. This reads
        backward from the end of the file in chunks until it has at least
        n newlines, which keeps the cost roughly constant regardless of
        total file size.
        """
        if not os.path.exists(self.path):
            return []

        chunk_size = 65536
        file_size = os.path.getsize(self.path)
        if file_size == 0:
            return []

        with open(self.path, "rb") as f:
            data = b""
            pos = file_size
            newline_count = 0
            # +1 target: we need n complete lines, which means n+1 newlines
            # scanned from EOF in the worst case (a trailing partial/no
            # final newline) -- overshoot by one chunk rather than
            # under-read and silently return fewer than n records.
            while pos > 0 and newline_count <= n:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
                newline_count = data.count(b"\n")

        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()[-n:]

        records: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # a torn read of a line split mid-write (either the file's
                # true last line still being flushed, or a chunk boundary
                # landing mid-line) -- skip it, it'll read fine next poll.
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
_TAPE_MAX_ROWS = 12  # Live Trade Tape -- how many recent real fills to keep on screen
_GRID_LAYOUT_THRESHOLD = 20  # above this many tickers, switch from a tall list to a scroll-free grid


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

        self._notebook_capable = _HAS_IPYTHON and _get_ipython() is not None
        self._widget_capable = self._notebook_capable and _HAS_IPYWIDGETS
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

        if console is not None:
            self.console = console
        elif self._is_tty:
            # Real terminal -- let Rich auto-detect actual columns.
            self.console = Console()
        else:
            # No real terminal to measure (notebook/widget rendering, or a
            # piped/redirected stdout) -- Rich's own fallback here is a
            # narrow 80-column default, which starves _build_compact_grid()
            # down to its 4-column floor regardless of how much actual
            # horizontal space the notebook output area has. 160 gives the
            # grid room to actually use multiple columns in the common case
            # this dashboard runs in (Kaggle/Jupyter, not a real tty).
            self.console = Console(width=160)

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
        self._prev_price_per_ticker: Dict[str, float] = {}
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
        latest = history[-1]  # newest record of ANY type -- used only for staleness/liveness

        # Tick records now dominate the log (one per env-step vs. one per
        # 256-step rollout), so "the last line" and "the last rollout
        # summary" are usually different records. Find each explicitly
        # rather than assuming history[-1] is a rollout record like the
        # single-granularity version of this file did. record_type is
        # absent on any log written before this revision -- treated as
        # "rollout" for backward compatibility with old metrics files.
        latest_tick = next((r for r in reversed(history) if r.get("record_type") == "tick"), None)
        latest_rollout = next((r for r in reversed(history) if r.get("record_type", "rollout") != "tick"), None)
        if latest_tick is None:
            latest_tick = latest_rollout
        if latest_rollout is None:
            latest_rollout = latest_tick
        if latest_tick is None:
            return

        # "is this actually new data" drives the spinner + throughput --
        # keyed off the tick stream now (global_tick), since that's what
        # actually updates every single poll once training is tick-logging.
        step = latest_tick.get("step")
        now = time.time()
        is_new_step = step != self._last_step
        if is_new_step:
            self._frame_count += 1
        self._last_step = step
        self._last_render_time = now

        renderable = self._build_renderable(latest_tick, latest_rollout, history, now, is_new_step)

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
        self,
        tick_record: Dict[str, Any],
        rollout_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        now: float,
        is_new_step: bool,
    ) -> Group:
        # PPO training stats (reward, sharpe, policy/value loss) only exist
        # at rollout granularity -- read those from rollout_record. Live
        # position/net-worth/drawdown/trades read from tick_record, which
        # updates every single env-step, not once per 256.
        reward = rollout_record.get("reward")
        reward_ema = rollout_record.get("reward_ema")
        drawdown = tick_record.get("drawdown")
        net_worth = tick_record.get("net_worth")

        net_worth_color = None
        if net_worth is not None and self._prev_net_worth is not None:
            if net_worth > self._prev_net_worth:
                net_worth_color = "bright_green"
            elif net_worth < self._prev_net_worth:
                net_worth_color = "bright_red"
        if net_worth is not None:
            self._prev_net_worth = net_worth

        # throughput -- ticks/sec over the visible history window (now that
        # "step" means real env-ticks, this is real market-bar throughput,
        # not "rollouts/sec")
        ticks_per_sec = None
        tick_history = [r for r in history if r.get("record_type") == "tick"]
        if len(tick_history) >= 2:
            span_steps = (tick_history[-1].get("step") or 0) - (tick_history[0].get("step") or 0)
            span_time = (tick_history[-1].get("wall_time") or now) - (tick_history[0].get("wall_time") or now)
            if span_time > 0 and span_steps > 0:
                ticks_per_sec = span_steps / span_time

        spinner = _SPINNER_FRAMES[self._frame_count % len(_SPINNER_FRAMES)] if is_new_step else "\u25cf"
        spinner_color = "bright_green" if is_new_step else "grey62"  # dim grey, not yellow -- see this file's dark-mode color notes
        staleness = now - (tick_record.get("wall_time") or now)
        live_note = "" if staleness < 30 else f"  [red](no new data for {int(staleness)}s -- check the training process)[/red]"

        throughput_text = f"  |  {ticks_per_sec:.1f} ticks/s" if ticks_per_sec is not None else ""
        halted_list = tick_record.get("halted")
        n_halted = sum(1 for h in halted_list if h) if isinstance(halted_list, list) else 0
        halted_note = f"  [bold white on purple4] {n_halted} HALTED [/bold white on purple4] " if n_halted > 0 else ""

        panel_lines = [
            f"[{spinner_color}]{spinner}[/{spinner_color}] live  |  uptime {_fmt_uptime(now - self._start_time)}{throughput_text}{halted_note}",
            f"tick (env-steps):  {tick_record.get('step')}{live_note}",
            f"rollout:           {rollout_record.get('rollout', rollout_record.get('step'))}",
            f"episode (passes):  {rollout_record.get('episode')}",
            f"reward:            {_fmt_colored(reward, good_is='positive')}   (as of last rollout)",
            f"reward (EMA):      {_fmt_colored(reward_ema, good_is='positive')}",
            f"net worth (total): {_fmt_colored(net_worth, good_is=None, color=net_worth_color, dollar=True, flash=True)}",
            f"sharpe:            {_fmt(rollout_record.get('sharpe'))}",
            f"drawdown (avg):    {_fmt_colored(drawdown, good_is='small_pct', pct=True)}",
            f"trades (rollout):  {_fmt_int(tick_record.get('trades_this_rollout'))}   (live)",
            f"trades (total):    {_fmt_int(tick_record.get('total_trades'))}",
        ]
        header = Panel("\n".join(panel_lines), title="Training Status", expand=False)

        tickers = tick_record.get("tickers")
        position = tick_record.get("position")
        per_ticker_rows = self._compute_per_ticker_rows(tick_record, tickers, position)

        tape_panel = Panel(self._build_ticker_tape(tick_record), title="Live Prices", expand=False)

        n = len(per_ticker_rows)
        if n == 0:
            table = Table(title="Per-Environment State")
            table.add_column("(no per-ticker breakdown in this record)")
            table.add_row("")
            return Group(header, tape_panel, self._build_trade_tape(tick_history), table)

        if n <= _GRID_LAYOUT_THRESHOLD:
            # Small ticker counts: the full detailed table, unchanged from
            # before -- every column, one row per ticker, no scrolling
            # problem at this size.
            body = self._build_detail_table(per_ticker_rows, tick_record)
        else:
            # Large ticker counts (see _GRID_LAYOUT_THRESHOLD): a scroll-free
            # multi-column board instead of one tall list -- fits on screen
            # regardless of n by wrapping into as many rows as needed, at a
            # fixed column count sized to the console width. Full per-column
            # detail (unrealized PnL, drawdown, trade counts) isn't shown
            # per-tile at this density; a compact tile has ticker/price/
            # position/status only. Anything flagged (halted/bankrupt) ALSO
            # gets a full-detail row in the separate table below it, so
            # nothing critical is lost to the compact view.
            grid = self._build_compact_grid(per_ticker_rows)
            flagged = [r for r in per_ticker_rows if r["is_halted"] or r["is_bankrupt"]]
            pieces = [grid]
            if flagged:
                pieces.append(self._build_detail_table(flagged, tick_record, title_prefix="Flagged "))
            body = Group(*pieces)

        return Group(header, tape_panel, self._build_trade_tape(tick_history), body)

    def _compute_per_ticker_rows(
        self, tick_record: Dict[str, Any], tickers: Any, position: Any
    ) -> List[Dict[str, Any]]:
        """
        One computation pass, shared by both the detailed table and the
        compact grid, so trend-flash state (self._prev_price_per_ticker /
        self._prev_net_worth_per_ticker) only gets updated ONCE per ticker
        per frame regardless of which layout ends up rendering it -- doing
        this twice (once per layout) would double-advance the "previous
        value" trackers and break the flash/arrow logic.
        """
        if not (isinstance(tickers, list) and isinstance(position, list)):
            return []
        n = len(tickers)
        net_worth_per_ticker = _as_list(tick_record.get("net_worth_per_ticker"), n)
        price_per_ticker = _as_list(tick_record.get("price_per_ticker"), n)
        unrealized = _as_list(tick_record.get("unrealized_pnl"), n)
        drawdown_per_ticker = _as_list(tick_record.get("drawdown_per_ticker"), n)
        trades_rollout_per_ticker = _as_list(tick_record.get("trades_per_ticker_this_rollout"), n)
        trades_total_per_ticker = _as_list(tick_record.get("total_trades_per_ticker"), n)
        filled_this_tick = _as_list(tick_record.get("filled_qty_this_tick"), n)
        halted_per_ticker = _as_list(tick_record.get("halted"), n)

        rows: List[Dict[str, Any]] = []
        for i, ticker in enumerate(tickers):
            price = price_per_ticker[i]
            price_color = None
            if price is not None:
                prev_price = self._prev_price_per_ticker.get(ticker)
                if prev_price is not None:
                    if price > prev_price:
                        price_color = "bright_green"
                    elif price < prev_price:
                        price_color = "bright_red"
                self._prev_price_per_ticker[ticker] = price
            price_text = _fmt_colored(price, good_is=None, color=price_color, dollar=True, flash=True)
            price_arrow = "\u25b2" if price_color == "bright_green" else ("\u25bc" if price_color == "bright_red" else "")

            nw = net_worth_per_ticker[i]
            nw_color = None
            if nw is not None:
                prev = self._prev_net_worth_per_ticker.get(ticker)
                if prev is not None:
                    if nw > prev:
                        nw_color = "bright_green"
                    elif nw < prev:
                        nw_color = "bright_red"
                self._prev_net_worth_per_ticker[ticker] = nw
            nw_text = _fmt_colored(nw, good_is=None, color=nw_color, dollar=True, flash=True)
            if nw_color == "bright_green":
                nw_text = "\u25b2 " + nw_text
            elif nw_color == "bright_red":
                nw_text = "\u25bc " + nw_text

            trades_this = trades_rollout_per_ticker[i]
            trades_total = trades_total_per_ticker[i]
            fq = filled_this_tick[i]
            just_traded = fq is not None and fq != 0
            trades_this_text = f"[bold cyan]{_fmt_int(trades_this)}[/bold cyan]" if just_traded else _fmt_int(trades_this)

            is_halted = bool(halted_per_ticker[i]) if halted_per_ticker[i] is not None else False
            is_bankrupt = nw is not None and nw <= 0
            if is_bankrupt:
                status_text = "[bold white on bright_red]BANKRUPT[/bold white on bright_red]"
                row_style = "on bright_red"
            elif is_halted:
                status_text = "[bold white on purple4]HALTED[/bold white on purple4]"
                row_style = "on purple4"
            else:
                status_text = ""
                row_style = None

            rows.append({
                "ticker": ticker,
                "price": price, "price_text": price_text, "price_color": price_color, "price_arrow": price_arrow,
                "position": position[i], "nw_text": nw_text, "nw_color": nw_color,
                "unrealized_text": _fmt_colored(unrealized[i], good_is="positive"),
                "drawdown_text": _fmt_colored(drawdown_per_ticker[i], good_is="small_pct", pct=True),
                "trades_this_text": trades_this_text, "just_traded": just_traded,
                "trades_total": _fmt_int(trades_total),
                "status_text": status_text, "row_style": row_style,
                "is_halted": is_halted, "is_bankrupt": is_bankrupt,
            })
        return rows

    def _build_detail_table(
        self, rows: List[Dict[str, Any]], tick_record: Dict[str, Any], title_prefix: str = ""
    ) -> Table:
        """Full-column table, one row per ticker -- used directly for small ticker counts, and for the
        flagged-only subset shown alongside the compact grid at large ticker counts."""
        title = f"{title_prefix}Per-Environment State  (tick #{tick_record.get('step')}, frame #{self._frame_count})"
        table = Table(title=title)
        table.add_column("Ticker")
        table.add_column("Price", justify="right")
        table.add_column("Position", justify="right")
        table.add_column("Net Worth", justify="right")
        table.add_column("Unrealized PnL", justify="right")
        table.add_column("Drawdown", justify="right")
        table.add_column("Trades (rollout)", justify="right")
        table.add_column("Trades (total)", justify="right")
        table.add_column("Status")
        for r in rows:
            table.add_row(
                str(r["ticker"]), r["price_text"], _fmt(r["position"]), r["nw_text"],
                r["unrealized_text"], r["drawdown_text"], r["trades_this_text"], r["trades_total"],
                r["status_text"], style=r["row_style"],
            )
        return table

    def _build_compact_grid(self, rows: List[Dict[str, Any]]) -> Table:
        """
        Scroll-free multi-column board for large ticker counts (see
        _GRID_LAYOUT_THRESHOLD) -- wraps N tickers into as many columns as
        the console width allows, instead of one N-row list a person has to
        scroll through. Each tile: ticker, real price with its flash/arrow,
        position, and a one-character status dot (colored, not just
        text -- readable at a glance even in a small tile). This is
        deliberately less detailed per-ticker than the full table -- see
        _build_renderable()'s docstring comment on why flagged
        (halted/bankrupt) envs still get a full-detail row elsewhere.
        """
        col_width = 20
        n_cols = max(4, min(12, (self.console.width or 100) // col_width))

        grid = Table.grid(padding=(0, 1))
        for _ in range(n_cols):
            grid.add_column()

        def tile(r: Dict[str, Any]) -> str:
            if r["is_bankrupt"]:
                dot = "[bright_red]\u25cf[/bright_red]"
            elif r["is_halted"]:
                dot = "[purple4]\u25cf[/purple4]"
            else:
                dot = "[grey42]\u25cf[/grey42]"
            arrow = r["price_arrow"]
            arrow_markup = (
                f"[bright_green]{arrow}[/bright_green]" if r["price_color"] == "bright_green"
                else f"[bright_red]{arrow}[/bright_red]" if r["price_color"] == "bright_red"
                else " "
            )
            price_str = f"{r['price']:,.2f}" if r["price"] is not None else "-"
            return f"{dot} [bold]{r['ticker']:<6}[/bold]\n  {price_str}{arrow_markup}  pos {_fmt(r['position'])}"

        row_cells: List[str] = []
        for r in rows:
            row_cells.append(tile(r))
            if len(row_cells) == n_cols:
                grid.add_row(*row_cells)
                row_cells = []
        if row_cells:
            row_cells += [""] * (n_cols - len(row_cells))
            grid.add_row(*row_cells)

        return grid

    def _build_ticker_tape(self, tick_record: Dict[str, Any]) -> Text:
        """
        A single-line, continuously-updating readout of REAL per-tick
        prices for EVERY ticker, every frame -- tick_record["price_per_ticker"],
        the exact mid price env.step() traded against on this tick (see
        train.py's tick_callback). Unlike _build_trade_tape() below (which
        only shows a row when a fill actually happened), this updates on
        literally every render_once() call regardless of whether anything
        traded -- a real market ticker moves even when you personally
        aren't trading it, and that continuous real movement (not fake
        flicker) is what gives the "watching a live market" feel.

        If this tick_record predates the price_per_ticker field (an old
        metrics.jsonl from before that field existed), this prints an
        explicit fallback line rather than drawing a tape with fabricated
        numbers -- same convention used everywhere else in this file.
        """
        tickers = tick_record.get("tickers")
        prices = tick_record.get("price_per_ticker")

        if not isinstance(tickers, list) or not isinstance(prices, list) or len(tickers) != len(prices):
            return Text("(no price feed in this record -- resume with the updated train.py to populate it)",
                        style="grey50")

        tape = Text()
        for i, ticker in enumerate(tickers):
            price = prices[i]
            if price is None:
                tape.append(f" {ticker} -- ", style="grey50")
                continue

            prev = self._prev_price_per_ticker.get(ticker)
            if prev is None:
                arrow, color = "\u2022", "grey62"
            elif price > prev:
                arrow, color = "\u25b2", "bright_green"
            elif price < prev:
                arrow, color = "\u25bc", "bright_red"
            else:
                arrow, color = "\u2022", "grey62"
            self._prev_price_per_ticker[ticker] = price

            tape.append(f" {ticker} ", style="bold white")
            tape.append(f"{price:,.2f} ", style=color)
            tape.append(f"{arrow} ", style=color)
            tape.append("|", style="grey35")

        return tape

    def _build_trade_tape(self, tick_history: List[Dict[str, Any]]) -> Panel:
        """
        A scrolling blotter of REAL individual fills, newest first -- every
        entry here is read directly off a real tick record's
        filled_qty_this_tick / price_per_ticker (see train.py's
        tick_callback: price_per_ticker is the actual mid price env.step()
        marked that fill against, not a display-only estimate). Nothing in
        this panel is synthesized, randomized, or interpolated -- the
        "flicker" comes entirely from genuinely new fills appearing as
        training actually produces them, at whatever rate that really
        happens, not from an animation faking activity that isn't there.

        Scans tick_history (already fetched for this frame, no extra I/O)
        for any record with a nonzero fill, most recent last-in-history
        first. Capped at the last _TAPE_MAX_ROWS fills found, so this stays
        cheap even with a large history_window.
        """
        events: List[str] = []
        for record in reversed(tick_history):
            if len(events) >= _TAPE_MAX_ROWS:
                break
            tickers = record.get("tickers")
            filled = record.get("filled_qty_this_tick")
            prices = record.get("price_per_ticker")
            if not (isinstance(tickers, list) and isinstance(filled, list)):
                continue
            step = record.get("step")
            for i, qty in enumerate(filled):
                if len(events) >= _TAPE_MAX_ROWS:
                    break
                if not qty:
                    continue
                ticker = tickers[i] if i < len(tickers) else "?"
                price = prices[i] if isinstance(prices, list) and i < len(prices) else None
                side = "BUY " if qty > 0 else "SELL"
                color = "bright_green" if qty > 0 else "bright_red"
                price_text = f"@ ${price:,.2f}" if price is not None else ""
                events.append(
                    f"[grey62]t{step:>7}[/grey62]  [bold {color}]{side}[/bold {color}]  "
                    f"[bold]{ticker:<6}[/bold]  {abs(qty):>8.2f} sh  {price_text}"
                )

        if not events:
            body = "[grey62](no fills yet)[/grey62]"
        else:
            body = "\n".join(events)

        return Panel(body, title="Live Trade Tape", expand=False, border_style="grey42")



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
    flash: bool = False,
) -> str:
    """
    good_is="positive"  -> green if value >= 0, red if value < 0
    good_is="small_pct" -> green if value <= 0.10 (10%), red otherwise
    good_is=None         -> no automatic rule; only an explicit `color` applies
    `color`, when given, always wins over the good_is rule.

    flash=True adds a reverse-video (inverted fg/bg) style on top of the
    resolved color -- meant ONLY for values whose color already means "this
    changed since last frame" (price, net worth -- see their call sites,
    where color is None unless an actual tick-to-tick delta was detected).
    Since redraws happen every poll, a value that changed gets exactly one
    inverted frame before reverting to plain colored text on the next
    render -- a real flash, not a persistent highlight, and only ever on
    values that are genuinely moving. Do NOT set flash=True for
    magnitude/sign-based coloring (e.g. unrealized PnL's good_is="positive")
    -- that color reflects current sign, not "just changed," and would
    flash every single frame regardless of whether anything moved.
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
            resolved_color = "bright_green" if v >= 0 else "bright_red"
        elif good_is == "small_pct":
            resolved_color = "bright_green" if v <= 0.10 else "bright_red"

    if resolved_color is None:
        return text
    style = f"bold reverse {resolved_color}" if flash else resolved_color
    return f"[{style}]{text}[/{style}]"