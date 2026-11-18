# Paths, constants and plotting style of the analysis.

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, ".."))

RESULTS_EXT = os.environ.get("BAG_RESULTS_EXT", os.path.join(ROOT, "results_ext"))
CACHE_DIR = os.path.join(_HERE, "cache")
OUT_DIR = os.path.join(_HERE, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CSV_DIR = os.path.join(OUT_DIR, "csv")

LEARNERS = ["LR", "KNN", "SVR", "KRR", "RT", "HINNPerf"]
METRICS = ["MRE", "MAE", "MBRE", "MIBRE"]
BORDA_METRICS = ["MAE", "MRE", "MBRE", "MIBRE"]
RULES = ["MEAN", "MEDIAN", "IRWM"]
K_ALL = list(range(1, 11))
K_ENS = list(range(2, 11))
NUM_RUNS = 30

DATASETS = ["apache", "bdbc", "kanzi", "x264", "hipacc", "hsmgp", "dune", "lrzip"]
DATASET_LABELS = {d: d.upper().replace("BDBC", "BDB-C") for d in DATASETS}
DATASET_REGISTRY = {
    "apache": [9, 18, 27, 36, 45],
    "bdbc": [18, 36, 54, 72, 90],
    "kanzi": [31, 62, 93, 124, 155],
    "x264": [16, 32, 48, 64, 80],
    "lrzip": [127, 295, 386, 485, 907],
    "dune": [224, 692, 1000, 1365, 1612],
    "hipacc": [261, 528, 736, 1281, 2631],
    "hsmgp": [77, 173, 384, 480, 864],
}

SK_KWARGS = {"a12_threshold": 0.60, "conf": 0.01, "seed": 42}
N_JOBS = max(1, (os.cpu_count() or 2) - 2)

MODEL_COLORS = {
    "LR": "#1f77b4",
    "SVR": "#ff7f0e",
    "RT": "#2ca02c",
    "KNN": "#9467bd",
    "KRR": "#8c564b",
    "HINNPerf": "#7f7f7f",
}
MODEL_MARKERS = {"LR": "o", "KNN": "s", "SVR": "^", "KRR": "D", "RT": "v", "HINNPerf": "P"}
MODEL_LINES = {"LR": "-", "KNN": "--", "SVR": "-.", "KRR": ":", "RT": (0, (6, 2)), "HINNPerf": (0, (3, 1, 1, 1, 1, 1))}

CONFIGS = ["Bag-OOB", "Bag-k10-MEDIAN"]

CM = 1 / 2.54
FULL_W = 12.2 * CM
FONT_SIZE = 8
PNG_DPI = 200
