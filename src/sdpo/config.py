"""Project-wide constants. Values come from the design spec; change them there first."""

from pathlib import Path

SEED = 42

# Universe and dates
N_UNIVERSE = 30
UNIVERSE_WINDOW = ("2010-01-01", "2011-12-31")
PRICE_START = "2010-01-01"
PRICE_END = "2020-07-01"
TRAIN_START = "2010-01-01"
TEST_YEARS = range(2016, 2021)

# Features
SEQ_LEN = 12
FEATURES = ["sent_mean", "sent_std", "log_news", "sent_change", "ret_1w", "ret_4w", "vol_4w"]

# Portfolio
MAX_WEIGHT = 0.10
COST_BPS = 10
BL_DELTA = 2.5
BL_TAU = 0.05
COV_WINDOW = 52
OMEGA_WINDOW = 26
TILT_K = 0.5
LAMBDA_EQ_GRID = (0.1, 1.0, 10.0)

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
