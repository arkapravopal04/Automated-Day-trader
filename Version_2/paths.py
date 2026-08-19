"""
Shared environment & path resolution for the trading data pipeline.
All pipeline scripts (fetch_alpaca.py, preprocess.py, dataset.py, data_diagnostics.py)
import from this module so that "where is my data" is defined in exactly one place.

Behavior:
- Local machine: everything lives under <this_folder>/data/{parquet,processed}
- Kaggle:        everything lives under /kaggle/working/data/{parquet,processed}
                  (writable, but wiped when the session ends unless you save it
                  as Kaggle Dataset output). If you've attached a pre-fetched
                  cache as a Kaggle *input* Dataset, this module will auto-detect
                  it and bootstrap /kaggle/working/data on first run, so you don't
                  re-download 3 years of history every session.

Kaggle input cache layout: attach a Kaggle input Dataset containing a
top-level (or one-level-nested) folder named "data" with "parquet" and
"processed" subfolders inside it -- e.g.:

    /kaggle/input/<your-dataset-slug>/data/parquet/AAPL.parquet, ...
    /kaggle/input/<your-dataset-slug>/data/processed/AAPL_features.parquet, ...
    /kaggle/input/<your-dataset-slug>/data/processed/metadata.json

find_kaggle_input_dir() below searches both the dataset root AND one level
of subfolders (so "data/parquet" and "data/processed" are both found), and
is tolerant of the "paraquet" misspelling in case that's how the folder
actually got named on upload.

Override anything explicitly with environment variables:
    TRADING_DATA_ROOT   -> overrides the base "data" directory entirely
    TRADING_RAW_DIR      -> overrides just the raw parquet directory
    TRADING_PROCESSED_DIR -> overrides just the processed features directory
"""

import os
import shutil
from typing import Dict, List, Optional

# Folder-name aliases to search for on Kaggle input datasets, keyed by the
# canonical name used everywhere else in this module. Add an alias here
# (rather than renaming your uploaded folder) if your Kaggle dataset uses a
# different spelling -- e.g. "paraquet" as a typo for "parquet".
SUBFOLDER_ALIASES: Dict[str, List[str]] = {
    "parquet": ["parquet", "paraquet"],
    "processed": ["processed"],
}

# Extensions bootstrap_dir() will copy from a matched Kaggle input folder.
# Kept as an allowlist (rather than copying everything) so stray files in
# the uploaded dataset (.DS_Store, notebook checkpoints, etc.) don't end up
# cluttering the working cache.
_CACHE_FILE_EXTENSIONS = (".parquet", ".json")


def is_kaggle() -> bool:
    """True if running inside a Kaggle Notebook/Script session."""
    if os.name == "nt":
        # Windows can't be Kaggle; os.path.exists('/kaggle/working') would
        # resolve to C:\\kaggle\\working and false-positive on any stray dir.
        return False
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.environ.get("KAGGLE_URL_BASE") is not None
        or os.path.exists("/kaggle/working")
    )


def _script_dir() -> str:
    """Directory containing this paths.py file (works locally when all
    pipeline scripts live side by side in the repo)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def get_base_dir() -> str:
    if os.environ.get("TRADING_DATA_ROOT"):
        return os.environ["TRADING_DATA_ROOT"]
    if is_kaggle():
        return "/kaggle/working/data"
    return os.path.join(_script_dir(), "data")


def _candidate_names(target_subfolder: str) -> List[str]:
    """Resolves a canonical subfolder name (e.g. 'parquet') to every alias to search for."""
    return SUBFOLDER_ALIASES.get(target_subfolder, [target_subfolder])


def find_kaggle_input_dir(target_subfolder: str) -> Optional[str]:
    """
    Search read-only /kaggle/input/*/ datasets for a folder matching
    `target_subfolder` (e.g. 'parquet' or 'processed'), or any of its
    aliases (see SUBFOLDER_ALIASES). Returns the first match, or None.
    Used to bootstrap a previously-uploaded cache dataset.

    Checks, per attached input dataset:
        1. Direct:  /kaggle/input/<dataset>/<alias>
        2. Nested:  /kaggle/input/<dataset>/<anything>/<alias>
                    (this is what matches your "data/parquet",
                    "data/processed" layout -- <anything> == "data")
    """
    input_root = "/kaggle/input"
    if not os.path.isdir(input_root):
        return None

    names = _candidate_names(target_subfolder)

    for dataset_name in sorted(os.listdir(input_root)):
        dataset_path = os.path.join(input_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for name in names:
            direct = os.path.join(dataset_path, name)
            if os.path.isdir(direct) and os.listdir(direct):
                return direct

        for sub in os.listdir(dataset_path):
            sub_path = os.path.join(dataset_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for name in names:
                nested = os.path.join(sub_path, name)
                if os.path.isdir(nested) and os.listdir(nested):
                    return nested

    return None


def _copy_cache_tree(source: str, target_dir: str) -> int:
    """
    Recursively copies every file under `source` matching
    _CACHE_FILE_EXTENSIONS into `target_dir`, preserving relative
    subdirectory structure. Returns the number of files copied.
    """
    copied = 0
    for root, _dirs, files in os.walk(source):
        rel_root = os.path.relpath(root, source)
        dest_root = target_dir if rel_root == "." else os.path.join(target_dir, rel_root)
        for fname in files:
            if not fname.endswith(_CACHE_FILE_EXTENSIONS):
                continue
            os.makedirs(dest_root, exist_ok=True)
            shutil.copy2(os.path.join(root, fname), os.path.join(dest_root, fname))
            copied += 1
    return copied


def bootstrap_dir(target_dir: str, subfolder_name: str, quiet: bool = False) -> None:
    """
    If `target_dir` doesn't exist yet or is empty, and we're on Kaggle,
    try to seed it by copying from an attached input Dataset with a
    matching subfolder name (e.g. your uploaded "data/parquet" or
    "data/processed" folder -- see module docstring). No-op locally, and
    a no-op if target_dir already has files.
    """
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return
    if not is_kaggle():
        return
    source = find_kaggle_input_dir(subfolder_name)
    if source is None:
        return

    os.makedirs(target_dir, exist_ok=True)
    copied = _copy_cache_tree(source, target_dir)
    if not quiet:
        print(f"[paths] Bootstrapped {copied} file(s) into {target_dir} from Kaggle input cache '{source}'.")


def is_cache_ready(require_metadata: bool = True) -> bool:
    """
    Cheap check for "is there already usable data on disk right now" --
    call this from fetch_alpaca.py / preprocess.py as a guard clause so a
    Kaggle session with an attached cache dataset skips re-fetching or
    re-preprocessing entirely instead of just skipping the download step.

    Checks:
        - RAW_DIR contains at least one .parquet file
        - PROCESSED_DIR contains metadata.json (if require_metadata) and at
          least one *_features.parquet file

    This only inspects the CURRENT contents of RAW_DIR/PROCESSED_DIR (i.e.
    after bootstrap_dir() has already run at import time below) -- it does
    not itself search /kaggle/input.
    """
    raw_ok = os.path.isdir(RAW_DIR) and any(f.endswith(".parquet") for f in os.listdir(RAW_DIR))

    if not os.path.isdir(PROCESSED_DIR):
        return False
    processed_files = os.listdir(PROCESSED_DIR)
    processed_ok = any(f.endswith("_features.parquet") for f in processed_files)
    if require_metadata:
        processed_ok = processed_ok and "metadata.json" in processed_files

    return raw_ok and processed_ok


BASE_DIR = get_base_dir()

RAW_DIR = os.environ.get("TRADING_RAW_DIR") or os.path.join(BASE_DIR, "parquet")
PROCESSED_DIR = os.environ.get("TRADING_PROCESSED_DIR") or os.path.join(BASE_DIR, "processed")

# Train/val/test split as FRACTIONS of each ticker's available history, rather
# than fixed calendar dates. This way the split scales automatically no matter
# how many years of history you've fetched (3, 6, whatever). Test = remainder.
# Override via env vars if you want a different split, e.g. more test data:
#   TRADING_TRAIN_FRAC=0.7  TRADING_VAL_FRAC=0.15   (leaves 0.15 for test)
TRAIN_FRAC = float(os.environ.get("TRADING_TRAIN_FRAC", "0.8"))
VAL_FRAC = float(os.environ.get("TRADING_VAL_FRAC", "0.1"))
if TRAIN_FRAC + VAL_FRAC >= 1.0:
    raise ValueError(
        f"TRADING_TRAIN_FRAC ({TRAIN_FRAC}) + TRADING_VAL_FRAC ({VAL_FRAC}) must leave "
        f"room for a non-empty test split (currently sum to {TRAIN_FRAC + VAL_FRAC})."
    )

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Directional-bias mitigation (see directional_bias_fix notes) ---
# Fraction of episodes that get price-mirrored (returns sign-flipped) so the
# agent can't exploit "long = free money" on an inherently bullish 6-year
# equity dataset. Decided INDEPENDENTLY per ticker/stream each episode (not
# one global flag for the whole batch), so a single episode can have some
# tickers mirrored and others not. 0.5 = each stream has a 50/50 chance.
# Set to 0.0 to disable entirely.
MIRROR_PROB = float(os.environ.get("TRADING_MIRROR_PROB", "0.5"))

# Rolling-window opponent-aware diversity bonus: penalizes a persistently
# one-sided direction history over the last DIVERSITY_WINDOW steps.
# DIVERSITY_COEF=0.0 disables it.
DIVERSITY_WINDOW = int(os.environ.get("TRADING_DIVERSITY_WINDOW", "50"))
DIVERSITY_COEF = float(os.environ.get("TRADING_DIVERSITY_COEF", "0.015"))

# --- Overtrading surcharge & platform fee ---
# Extra adverse slippage applied once a stream has traded more than
# OVERTRADE_FREE_TRADES times within the last OVERTRADE_WINDOW bars (5-min
# bars by default, so 12 bars ~= 1 hour). Discourages churning the position
# every step just to farm the vol-normalized step reward. Set
# OVERTRADE_SURCHARGE_BPS=0.0 to disable.
OVERTRADE_WINDOW = int(os.environ.get("TRADING_OVERTRADE_WINDOW", "12"))
OVERTRADE_FREE_TRADES = int(os.environ.get("TRADING_OVERTRADE_FREE_TRADES", "3"))
OVERTRADE_SURCHARGE_BPS = float(os.environ.get("TRADING_OVERTRADE_SURCHARGE_BPS", "3.0"))

# Flat $ ticket fee charged by the broker/platform on any non-zero fill,
# independent of size (separate from commission_bps/commission_per_share,
# which model exchange/broker commission proper). Set to 0.0 to disable.
# Alpaca (which live/broker_client.py targets) is commission-free on US
# equities, so the correct default is 0.0. This is NOT the Session 1 advice
# that was retracted: back then the $1 ticket was masking the dust-order bug,
# and zeroing it would have hidden a real defect. That bug is fixed, and the
# 151-rollout run of 2026-08-19 measured what the ticket actually costs now:
#
#   305,966 trades, $375,288 total loss  ->  $1.227 loss per trade
#   of which the flat ticket is           ->  $1.000  = 81.5% of ALL losses
#
# Everything else (spread, impact, adverse selection) sums to $0.227/trade.
# Set TRADING_PLATFORM_FEE_PER_TRADE=1.0 to model an IBKR-shaped venue.
PLATFORM_FEE_PER_TRADE = float(os.environ.get("TRADING_PLATFORM_FEE_PER_TRADE", "0.0"))

# --- Liquidity-proxy scaling: single-venue (IEX) volume -> consolidated ---
# HISTORICAL CONTEXT -- see "RESOLVED" at the bottom of this block first.
# fetch_alpaca.py's DEFAULT_DATA_FEED used to be "iex", so every bar's `volume`
# column is IEX-only prints, NOT consolidated tape volume. IEX is roughly
# 2-3% of total US equity volume, and measuring this dataset against
# real-world figures confirms it: AAPL 1.17M/day here vs ~45M actual,
# BAC 1.18M vs ~35M, ABBV 186k vs ~6M -- a consistent ~30-40x shortfall.
#
# That matters because vec_trading_env.py feeds this volume to
# execution_sim.py as `bar_liquidity_proxy`, where participation =
# order_size / volume drives a SQRT market-impact term. Understating volume
# ~35x overstates participation ~35x, and because sqrt(x) >> x for small x
# the resulting slippage is wildly overstated: a 1.33-share order against a
# 1,736-share median bar paid 0.277% one-way (0.55% round trip) against a
# 0.056% median 5-minute price move -- i.e. every trade was structurally
# unprofitable before the policy made any decision at all.
#
# Alpaca routes retail orders across venues, not to IEX alone, so
# consolidated volume is the correct liquidity proxy for what a real fill
# would see. Scaling here (rather than lowering impact_coef to compensate)
# keeps the impact MODEL honest and fixes the input that was actually
# wrong.
#
# RESOLVED 2026-08-19: the fudge is no longer needed. The free Alpaca tier
# does serve full historical SIP -- only the trailing ~15 minutes is
# embargoed, which the fetch now steps around (see fetch_alpaca.py's
# SIP_LAG_MINUTES). DEFAULT_DATA_FEED is therefore "sip" and bar volume is
# already consolidated, so the correct scale is 1.0.
#
# The 35x estimate was also wrong in a way worth recording: measured on one
# year of AAPL regular-hours bars, SIP median volume is 318,938 against
# IEX's 14,008 -- a ratio of ~23x, not 35x. Scaling IEX by 35 therefore
# OVERSTATED volume by ~1.5x, understating participation and UNDERcharging
# sqrt-impact. (A single earlier spot-check suggested ~73x; that was one
# unrepresentative bar, which is exactly why per-bar samples are a bad way
# to calibrate a constant like this.)
#
# Leave this at 1.0 whenever the cache was fetched on SIP. It is only
# meaningful for an IEX-fetched cache, and mixing the two in one cache is a
# silent corruption -- fetch_alpaca.py shouts loudly if a ticker falls back.
VOLUME_SCALE = float(os.environ.get("TRADING_VOLUME_SCALE", "1.0"))

# On Kaggle, opportunistically seed from an attached input dataset the first
# time these directories are touched. Cheap no-op everywhere else.
if is_kaggle():
    bootstrap_dir(RAW_DIR, "parquet")
    bootstrap_dir(PROCESSED_DIR, "processed")
    if is_cache_ready():
        print("[paths] Kaggle input cache detected and bootstrapped -- "
              "fetch_alpaca.py/preprocess.py can skip regenerating this data.")


if __name__ == "__main__":
    print(f"Environment : {'Kaggle' if is_kaggle() else 'Local'}")
    print(f"BASE_DIR    : {BASE_DIR}")
    print(f"RAW_DIR     : {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"Cache ready : {is_cache_ready()}")