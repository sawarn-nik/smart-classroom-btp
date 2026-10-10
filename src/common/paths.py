from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

LFW_DIR = RAW / "lfw" / "lfw_home" / "lfw_funneled"
LFW_PAIRS = RAW / "lfw" / "lfw_home" / "pairs.txt"
INSIGHTFACE_ROOT = CKPT / "insightface"

for p in (PROCESSED, METRICS, FIGURES, TABLES):
    p.mkdir(parents=True, exist_ok=True)
