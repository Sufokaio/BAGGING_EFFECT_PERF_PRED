# Shared settings and helpers.

import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_ORIG = os.environ.get("BAG_RESULTS_ORIG", os.path.join(ROOT, "results_original"))
RESULTS_EXT = os.environ.get("BAG_RESULTS_EXT", os.path.join(ROOT, "results_ext"))

INIT_SEED = 25
NUM_RUNS = 30
MAX_N = 10
METRICS = ["MAE", "MRE", "MBRE", "MIBRE"]
RULES = ["mean", "median", "irwm"]

LEARNERS = ["SVR", "RT", "KRR", "LR", "KNN", "HINNPerf"]
COPIED_SINGLES = ["SVR", "RT", "KRR", "HINNPerf"]
NEW_SINGLES = ["LR", "KNN"]

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


def size_level(dataset, s):
    return DATASET_REGISTRY[dataset].index(int(s)) + 1


def scenarios(datasets=None, size=None, levels=None):
    out = []
    for ds, sizes in DATASET_REGISTRY.items():
        if datasets is not None and ds not in datasets:
            continue
        for i, s in enumerate(sizes):
            if size is not None and int(s) != int(size):
                continue
            if levels is not None and (i + 1) not in levels:
                continue
            out.append((ds, int(s)))
    return sorted(out, key=lambda x: (size_level(*x), x[1], x[0]))


def scen_dir(dataset, s, root=None):
    return os.path.join(root or RESULTS_EXT, dataset, str(int(s)))


def run_split(n_total, num_train, run):
    np.random.seed(INIT_SEED + run)
    indices = np.random.permutation(n_total)
    return indices[:num_train], indices[num_train:]


def load_top1_params(dataset, s, model_name, root=None):
    path = os.path.join(scen_dir(dataset, s, root), f"{model_name}_single_metrics.json")
    with open(path, "r") as f:
        return json.load(f)[0]["Params"]


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mre = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    mbre = np.mean(np.abs(y_true - y_pred) / (np.minimum(y_true, y_pred) + 1e-8))
    mibre = np.mean(np.abs(y_true - y_pred) / (np.maximum(y_true, y_pred) + 1e-8))
    return {"MAE": mae, "MRE": mre, "MBRE": mbre, "MIBRE": mibre}


def log_line(filename, message, root=None):
    root = root or RESULTS_EXT
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(root, filename), "a") as f:
        f.write(f"[{stamp}] {message}\n")


def log_time(message, root=None):
    log_line("times_ext.txt", message, root)
