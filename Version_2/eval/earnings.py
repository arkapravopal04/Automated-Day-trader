"""
eval/earnings.py -- map a scheduled-earnings calendar onto the overnight book's
periods, so the book can stand flat into a print.

THE MAPPING IS THE WHOLE FILE, AND IT IS EASY TO GET BACKWARDS.

The overnight book's period dated session `D` is ENTERED at D's close and EXITED
at the next session's open. So the window a position is exposed across is
`D close -> D+1 open`, and a release lands inside it when:

    AMC on date d   the print is published after d's close, and the stock gaps
                    at the NEXT open. The exposed period is the one dated d.
                    -> exclude the last session with date <= d

    BMO on date d   the print is published before d's open, and the stock gaps
                    at d's OWN open. The exposed period is the one entered the
                    session BEFORE.
                    -> exclude the last session with date < d

Invert those two and the control excludes the safe session while holding the
dangerous one -- it would do nothing, while appearing in every log to be on, and
the conclusion drawn would be that calendars do not help. `assert_mapping()`
below is the regression test for exactly that, pinned to releases whose side is
independently known.

`unknown` timing excludes BOTH adjacent periods. Over-exclusion costs breadth;
under-exclusion is the failure above.

Adjacency is by TRADING SESSION, never by calendar day. A Monday-BMO print
excludes the preceding Friday, and the Martin Luther King holiday before MS's
2024-01-16 release excludes 2024-01-12 -- neither of which `d - 1 day` gets
right. A release dated on a non-trading day resolves to the surrounding
sessions by the same rule, so a Saturday 8-K does not silently vanish.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

TIMINGS = ("bmo", "amc", "unknown")


def load_calendar(path):
    """csv -> DataFrame[ticker, date(datetime64), timing]. Validated, not trusted."""
    if not os.path.exists(path):
        raise SystemExit(
            f"earnings calendar not found: {path}\n"
            "Build it once with:  python eval/fetch_earnings_calendar.py"
        )
    df = pd.read_csv(path)
    missing = {"ticker", "date", "timing"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s): {sorted(missing)}")
    bad = set(df["timing"].unique()) - set(TIMINGS)
    if bad:
        raise SystemExit(f"{path} has unrecognised timing value(s): {sorted(bad)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[["ticker", "date", "timing"]]


def session_dates(index, day_id):
    """The ET calendar date of each session, ordered by session id.

    `day_id` is `pd.factorize` over the normalised ET timestamp on a sorted
    index, so session id g is the g-th session and this is a plain lookup --
    but it is built by taking the FIRST bar of each id rather than assuming
    that, so a reordering upstream would surface as a non-monotonic array and
    trip the assertion instead of silently shifting every exclusion by a day.
    """
    ny = pd.DatetimeIndex(index).tz_convert("America/New_York").normalize().tz_localize(None)
    day_id = np.asarray(day_id)
    first = np.zeros(int(day_id.max()) + 1, dtype=np.int64)
    seen = np.zeros(first.shape[0], dtype=bool)
    for k in range(len(day_id)):
        g = day_id[k]
        if not seen[g]:
            first[g], seen[g] = k, True
    if not seen.all():
        raise SystemExit("day_id has gaps; the panel's session index is not contiguous")
    out = pd.DatetimeIndex(ny[first])
    if not out.is_monotonic_increasing:
        raise SystemExit("session dates are not increasing; the panel index is not sorted")
    return out


def exclusion_mask(index, tickers, day_id, calendar, report=True):
    """-> ([T, N] bool, stats). True where the name must be FLAT for that period.

    The mask is written across every bar of the excluded session, not only its
    decision bar. In overnight mode the schedule trades one period per session,
    so the two are the same set of trades; writing the whole session keeps the
    mask meaningful if it is ever read at another cadence.
    """
    sess = session_dates(index, day_id)
    sess_np = sess.values                       # sorted datetime64[ns]
    day_id = np.asarray(day_id)
    T, N = len(day_id), len(tickers)
    G = len(sess)

    col = {t: j for j, t in enumerate(tickers)}
    by_sess = np.zeros((G, N), dtype=bool)

    n_used = n_before = n_after = n_unknown_sym = 0
    per_timing = {t: 0 for t in TIMINGS}

    for tk, d, timing in calendar.itertuples(index=False):
        j = col.get(tk)
        if j is None:
            n_unknown_sym += 1
            continue
        dv = np.datetime64(pd.Timestamp(d).to_datetime64())
        hits = []
        if timing in ("amc", "unknown"):
            hits.append(int(np.searchsorted(sess_np, dv, side="right")) - 1)
        if timing in ("bmo", "unknown"):
            hits.append(int(np.searchsorted(sess_np, dv, side="left")) - 1)
        placed = False
        for g in hits:
            if g < 0:
                n_before += 1
                continue
            if g >= G:
                n_after += 1
                continue
            by_sess[g, j] = True
            placed = True
        if placed:
            n_used += 1
            per_timing[timing] += 1

    mask = by_sess[day_id]                       # [T, N]

    stats = {
        "events_in_calendar": int(len(calendar)),
        "events_applied": int(n_used),
        "events_by_timing": {k: int(v) for k, v in per_timing.items()},
        "events_symbol_not_in_panel": int(n_unknown_sym),
        "events_before_panel": int(n_before),
        "events_after_panel": int(n_after),
        "excluded_name_sessions": int(by_sess.sum()),
        "sessions": int(G),
        "names_with_no_events": sorted(
            t for j, t in enumerate(tickers) if not by_sess[:, j].any()
        ),
    }
    if report:
        print(f"[earnings] {stats['events_applied']:,} of "
              f"{stats['events_in_calendar']:,} calendar events land inside the "
              f"panel ({per_timing['bmo']:,} bmo, {per_timing['amc']:,} amc, "
              f"{per_timing['unknown']:,} unknown)")
        print(f"[earnings] {stats['excluded_name_sessions']:,} (name, session) "
              f"pairs held flat out of {G * N:,} "
              f"({100.0 * stats['excluded_name_sessions'] / max(G * N, 1):.2f}%)")
        gaps = stats["names_with_no_events"]
        if gaps:
            print(f"[earnings] NO CALENDAR ROWS for {len(gaps)} of {N} names -- the "
                  f"exclusion cannot act on them: {', '.join(gaps)}")
    return mask, stats


def apply_to_edge(edge, mask):
    """Blank the edge where the calendar says stand flat.

    NaN rather than zero: `book_weights` selects on `isfinite(edge)`, so a NaN
    is 'this name is not a candidate', while a zero is 'this name has no edge'
    and would still be demeaned against. The two differ in the offsetting leg.
    """
    return np.where(mask, np.nan, edge).astype(edge.dtype)


# ---------------------------------------------------------------------------
# The regression test for the inversion this file exists to prevent
# ---------------------------------------------------------------------------

# (ticker, release date, timing, the session that must be excluded). Every row
# is a release whose side is known independently of Yahoo: the six AMC names are
# fold 2's largest loss cells, dated in AGENTS.md against the gap that followed;
# MS 2024-01-16 is a BMO release the Monday after a market holiday, which is the
# case plain 'd - 1 day' arithmetic gets wrong.
KNOWN = (
    ("PANW",  "2024-02-20", "amc", "2024-02-20"),
    ("GOOGL", "2023-10-24", "amc", "2023-10-24"),
    ("NFLX",  "2024-01-23", "amc", "2024-01-23"),
    ("FDX",   "2023-12-19", "amc", "2023-12-19"),
    ("TSLA",  "2024-01-24", "amc", "2024-01-24"),
    ("ADBE",  "2024-03-14", "amc", "2024-03-14"),
    ("MS",    "2024-01-16", "bmo", "2024-01-12"),
    ("JPM",   "2024-01-12", "bmo", "2024-01-11"),
    ("KO",    "2024-02-13", "bmo", "2024-02-12"),
)


def assert_mapping(index, tickers, day_id, calendar):
    """Raise unless every KNOWN release excludes the session it actually gapped.

    Called on every run that uses the calendar. It costs one pass over nine rows
    and it is the only thing standing between 'the control is on' and 'the
    control is on and pointed the right way'.
    """
    sess = session_dates(index, day_id)
    mask, _ = exclusion_mask(index, tickers, day_id, calendar, report=False)
    col = {t: j for j, t in enumerate(tickers)}
    day_id = np.asarray(day_id)

    checked, failures = 0, []
    for tk, d, timing, want in KNOWN:
        j = col.get(tk)
        if j is None:
            continue
        have = calendar[(calendar.ticker == tk) &
                        (calendar.date == pd.Timestamp(d))]
        if not len(have):
            failures.append(f"{tk} {d}: absent from the calendar")
            continue
        got_timing = have.timing.iloc[0]
        if got_timing != timing:
            failures.append(f"{tk} {d}: calendar says {got_timing}, known {timing}")
            continue
        g = int(np.searchsorted(sess.values,
                                np.datetime64(pd.Timestamp(want).to_datetime64())))
        if g >= len(sess) or sess[g] != pd.Timestamp(want):
            continue                            # the session is outside the panel
        rows = np.flatnonzero(day_id == g)
        if not mask[rows, j].all():
            failures.append(f"{tk} {d} ({timing}): session {want} NOT excluded")
        checked += 1

    if failures:
        raise SystemExit("[earnings] MAPPING IS WRONG -- refusing to run:\n  "
                         + "\n  ".join(failures))
    print(f"[earnings] mapping check: {checked} known releases each exclude the "
          f"session they gapped")
    return checked
