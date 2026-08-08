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

Override anything explicitly with environment variables:
    TRADING_DATA_ROOT   -> overrides the base "data" directory entirely
    TRADING_RAW_DIR      -> overrides just the raw parquet directory
    TRADING_PROCESSED_DIR -> overrides just the processed features directory
"""

import os
import shutil


def is_kaggle() -> bool:
    """True if running inside a Kaggle Notebook/Script session."""
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


def find_kaggle_input_dir(target_subfolder: str):
    """
    Search read-only /kaggle/input/*/ datasets for a folder matching
    `target_subfolder` (e.g. 'parquet' or 'processed'). Returns the first
    match, or None. Used to bootstrap a previously-uploaded cache dataset.
    """
    input_root = "/kaggle/input"
    if not os.path.isdir(input_root):
        return None
    for dataset_name in sorted(os.listdir(input_root)):
        dataset_path = os.path.join(input_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        # Direct match: /kaggle/input/<dataset>/<target_subfolder>
        direct = os.path.join(dataset_path, target_subfolder)
        if os.path.isdir(direct) and os.listdir(direct):
            return direct
        # One level deeper: /kaggle/input/<dataset>/<anything>/<target_subfolder>
        for sub in os.listdir(dataset_path):
            nested = os.path.join(dataset_path, sub, target_subfolder)
            if os.path.isdir(nested) and os.listdir(nested):
                return nested
    return None


def bootstrap_dir(target_dir: str, subfolder_name: str, quiet: bool = False) -> None:
    """
    If `target_dir` doesn't exist yet or is empty, and we're on Kaggle,
    try to seed it by copying from an attached input Dataset with a
    matching subfolder name (e.g. your uploaded alpaca_cache dataset).
    No-op locally, and a no-op if target_dir already has files.
    """
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return
    if not is_kaggle():
        return
    source = find_kaggle_input_dir(subfolder_name)
    if source is None:
        return
    os.makedirs(target_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(source):
        if fname.endswith((".parquet", ".json")):
            shutil.copy2(os.path.join(source, fname), os.path.join(target_dir, fname))
            copied += 1
    if not quiet:
        print(f"[paths] Bootstrapped {copied} file(s) into {target_dir} from Kaggle input dataset '{source}'.")


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
DIVERSITY_COEF = float(os.environ.get("TRADING_DIVERSITY_COEF", "0.05"))

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
PLATFORM_FEE_PER_TRADE = float(os.environ.get("TRADING_PLATFORM_FEE_PER_TRADE", "1.0"))

# On Kaggle, opportunistically seed from an attached input dataset the first
# time these directories are touched. Cheap no-op everywhere else.
if is_kaggle():
    bootstrap_dir(RAW_DIR, "parquet")
    bootstrap_dir(PROCESSED_DIR, "processed")


if __name__ == "__main__":
    print(f"Environment : {'Kaggle' if is_kaggle() else 'Local'}")
    print(f"BASE_DIR    : {BASE_DIR}")
    print(f"RAW_DIR     : {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")