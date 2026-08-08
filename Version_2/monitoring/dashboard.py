"""
monitoring/dashboard.py

Metrics/visualization layer, built Kaggle-safe from the start.

v1's bug wasn't Rich itself -- it was Rich's Live(auto_refresh=True), which
runs a background thread doing in-place ANSI cursor redraws. On Kaggle,
that thread's redraws raced against the notebook's own stdout buffering and
corrupted the display. The fix here applies EVERYWHERE, not just on
Kaggle: Live() is only ever constructed with auto_refresh=False, and
refreshed by an explicit .refresh() call tied to the same cadence as
everything else. That removes the race at its source; it just happened to
only be visible on Kaggle because Kaggle's buffering was the thing that
turned a latent race into a corrupted screen.

Decoupling (this is the other half of "Kaggle-safe"):
    The training loop NEVER renders anything and NEVER imports Rich. It
    only calls MetricsWriter.log(step, **metrics) every N steps, which
    appends one JSON line to a flat file and does nothing else. Whether a
    dashboard is watching that file, crashed, or was never started has zero
    effect on training -- "is training crashing" and "is the dashboard
    crashing" are two different processes' problems, extending the same
    spirit as Berserker's log_redirect.py (isolate the thing that must keep
    running from the thing that's just for humans to look at).

JSONL over SQLite, on purpose: SQLite's file-locking semantics are a known
footgun on containerized/network-mounted filesystems (Kaggle's working
directory among them) when a writer and a reader touch the same file
concurrently -- exactly the class of "unrelated infra behavior corrupts the
run" bug this module exists to avoid. JSONL appends have no locking to get
wrong: `tail -f` works, `pandas.read_json(path, lines=True)` works, and a
half-written last line just gets skipped by the reader (see
MetricsReader.tail()).

Mode selection precedence (see resolve_mode()): an explicit --kaggle /
--local CLI flag > training/config.py's RunConfig.display_mode > env-var
auto-detection. This is what "the same training code path works in both
environments without branching logic scattered elsewhere" means in
practice: the training loop always just does `dashboard.render_once()` (or
runs the standalone polling loop); which mode that resolves to is decided
once, here, not re-checked all over the codebase.
"""

import enum
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


# --------------------------------------------------------------------------
# Metrics writer -- the training loop calls THIS, never the dashboard directly.
# --------------------------------------------------------------------------

class MetricsWriter:
    """
    Append-only structured metrics log (JSONL). One line per log() call,
    each line a self-contained JSON object -- no schema migration to think
    about later, no locking to get wrong.
    """

    def __init__(self, path: str, flush_every_call: bool = True):
        self.path = path
        self.flush_every_call = flush_every_call
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._fh = open(path, "a", buffering=1)  # line-buffered

    def log(self, step: int, **metrics: Any) -> None:
        """
        step: the training step/rollout index this record belongs to.
        **metrics: whatever the training loop has -- this project's
            convention is episode, reward, sharpe, drawdown, position, but
            nothing here enforces a fixed schema.
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
    """Handles the non-JSON-serializable values a training loop hands us (numpy/torch scalars, tensors)."""
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
# Metrics reader -- the dashboard's ONLY way of seeing training state. It
# never touches the training loop's in-memory objects.
# --------------------------------------------------------------------------

class MetricsReader:
    def __init__(self, path: str):
        self.path = path

    def tail(self, n: int) -> List[Dict[str, Any]]:
        """
        Returns the last n parsed records, oldest first. A plain linear
        read -- fine for a log meant to be polled every few seconds, not a
        hot path. Returns [] if the file doesn't exist yet (e.g. the
        dashboard started polling before training wrote anything).
        """
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
                # a torn read of the last, still-being-written line -- skip
                # it rather than crash the dashboard; it'll be complete
                # (and read fine) next poll.
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
    """
    Real env vars Kaggle sets, in both notebook and script-run contexts.
    Deliberately not sniffing sys.stdout.isatty() -- a non-interactive
    stdout doesn't by itself mean "Kaggle" (a local run redirected to a log
    file isn't interactive either), and we don't want auto-detection to
    misfire there.
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ


def resolve_display_mode(mode: DisplayMode) -> DisplayMode:
    if mode != DisplayMode.AUTO:
        return mode
    return DisplayMode.KAGGLE if _looks_like_kaggle() else DisplayMode.LOCAL


def resolve_mode(cfg_display_mode: str = "auto", argv: Optional[List[str]] = None) -> DisplayMode:
    """
    Precedence: explicit CLI flag > training/config.py's
    RunConfig.display_mode > env-var auto-detection. `argv` defaults to
    sys.argv[1:]; passed explicitly here mainly so this is testable without
    mutating sys.argv.
    """
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


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

class TrainingDashboard:
    """
    Renders structured metrics written by MetricsWriter. Two ways to drive
    it:
        (a) from the SAME process, calling render_once() every N training
            steps (cheap: it's just a file read + a print/refresh).
        (b) from a SEPARATE process/script with run_polling_loop(), which
            only ever touches metrics_path -- the strongest version of the
            "dashboard crashing can't touch training" guarantee, since
            there isn't even a shared Python process to crash.

    Mode differences are cosmetic ONLY (the auto_refresh=False fix in
    __init__ applies to both):
        LOCAL:  Live() in-place redraw -- the same table updates in place,
                same as v1's intended experience.
        KAGGLE: no Live() at all -- render_once() reprints a fresh
                Table/Panel every call, appended to scrollback instead of
                overwriting. Noisier output, but there's no redraw
                machinery left to race against, even in principle.
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

        self._live: Optional[Live] = None
        if self.mode == DisplayMode.LOCAL:
            # auto_refresh=False is the actual bug fix -- see module and
            # class docstrings. Never set this True.
            self._live = Live(console=self.console, auto_refresh=False, transient=False)

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
        """
        Reads the latest metrics off disk and renders one frame. Call this
        every N training steps (see
        training/config.py's RunConfig.dashboard_refresh_every_n_steps), or
        let run_polling_loop() call it on a timer instead.
        """
        history = self.reader.tail(self.history_window)
        if not history:
            return
        latest = history[-1]
        renderable = self._build_renderable(latest, history)

        if self.mode == DisplayMode.LOCAL:
            self._live.update(renderable)
            self._live.refresh()  # explicit -- never rely on Live's own timer
        else:
            # KAGGLE (and any future non-LOCAL mode): plain static print,
            # appended to scrollback, never redrawn in place.
            self.console.print(renderable)

    def run_polling_loop(self, poll_interval_seconds: float = 2.0, max_iterations: Optional[int] = None) -> None:
        """
        Standalone polling mode: run this from a separate process/script
        that only watches metrics_path. Blocks, sleeping
        poll_interval_seconds between reads, forever unless max_iterations
        is set (mainly for tests).
        """
        i = 0
        try:
            self.start()
            while max_iterations is None or i < max_iterations:
                self.render_once()
                time.sleep(poll_interval_seconds)
                i += 1
        finally:
            self.stop()

    def _build_renderable(self, latest: Dict[str, Any], history: List[Dict[str, Any]]) -> Group:
        panel_lines = [
            f"step:     {latest.get('step')}",
            f"episode:  {latest.get('episode')}",
            f"reward:   {_fmt(latest.get('reward'))}",
            f"sharpe:   {_fmt(latest.get('sharpe'))}",
            f"drawdown: {_fmt(latest.get('drawdown'), pct=True)}",
        ]
        header = Panel("\n".join(panel_lines), title="Training Status", expand=False)

        table = Table(title="Position State (latest)")
        table.add_column("Ticker")
        table.add_column("Position", justify="right")
        table.add_column("Unrealized PnL", justify="right")

        tickers = latest.get("tickers")
        position = latest.get("position")

        if isinstance(tickers, list) and isinstance(position, list):
            unrealized = latest.get("unrealized_pnl")
            unrealized = unrealized if isinstance(unrealized, list) else [None] * len(tickers)
            for ticker, pos, upnl in zip(tickers, position, unrealized):
                table.add_row(str(ticker), _fmt(pos), _fmt(upnl))
        else:
            table.add_row("(no per-ticker breakdown in this record)", "", "")

        return Group(header, table)


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:.2%}" if pct else f"{v:.4f}"