# Loads the results of results_ext into long-format tables.

import json
import os
from dataclasses import dataclass, field

import pandas as pd

import config as cfg

RULE_FILE = {"MEAN": "mean", "MEDIAN": "median", "IRWM": "irwm"}


def all_scenarios():
    out = []
    for ds in cfg.DATASETS:
        for i, s in enumerate(cfg.DATASET_REGISTRY[ds]):
            out.append((ds, s, i + 1))
    return out


def _sdir(root, ds, s):
    return os.path.join(root, ds, str(s))


def has_ext(learner, ds, s):
    d = _sdir(cfg.RESULTS_EXT, ds, s)
    return (os.path.exists(os.path.join(d, f"{learner}_bagging_N10_irwm.json"))
            and os.path.exists(os.path.join(d, f"{learner}_bagging_oob_selection.parquet"))
            and os.path.exists(os.path.join(d, f"{learner}_bagging_diversity.parquet")))


def _metrics_rows(path, base):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        obj = obj[0]
    rows = []
    for m in cfg.METRICS:
        for run, v in enumerate(obj["Metrics"][m]):
            rows.append(dict(base, run=run, metric=m, value=float(v)))
    return rows


def build_cache(force=False):
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    paths = {n: os.path.join(cfg.CACHE_DIR, f"{n}.parquet")
             for n in ["singles", "bagging", "oob", "diversity", "coverage"]}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return paths

    singles, bag, oob, div, cov = [], [], [], [], []
    for ds, s, lvl in all_scenarios():
        d_ext = _sdir(cfg.RESULTS_EXT, ds, s)
        for L in cfg.LEARNERS:
            key = dict(learner=L, dataset=ds, sample_size=s, level=lvl)
            sp = os.path.join(d_ext, f"{L}_single_metrics.json")
            if os.path.exists(sp):
                singles += _metrics_rows(sp, key)
            ext = has_ext(L, ds, s)
            cov.append(dict(key, ext=ext))
            if ext:
                for k in cfg.K_ALL:
                    for rule, suf in RULE_FILE.items():
                        bag += _metrics_rows(os.path.join(d_ext, f"{L}_bagging_N{k}_{suf}.json"),
                                             dict(key, k=k, rule=rule))
                for name, acc in [("oob_selection", oob), ("diversity", div)]:
                    t = pd.read_parquet(os.path.join(d_ext, f"{L}_bagging_{name}.parquet"))
                    for c, v in key.items():
                        t[c] = v
                    acc.append(t)

    pd.DataFrame(singles).to_parquet(paths["singles"], index=False)
    pd.DataFrame(bag).to_parquet(paths["bagging"], index=False)
    oob_df = pd.concat(oob, ignore_index=True) if oob else pd.DataFrame()
    if len(oob_df):
        oob_df["rule"] = oob_df["rule"].str.upper()
    oob_df.to_parquet(paths["oob"], index=False)
    (pd.concat(div, ignore_index=True) if div else pd.DataFrame()).to_parquet(paths["diversity"], index=False)
    pd.DataFrame(cov).to_parquet(paths["coverage"], index=False)
    return paths


@dataclass
class Scope:
    label: str
    scenarios: list
    ks: list
    rules: list
    note: str = ""


@dataclass
class Data:
    mode: str
    singles: pd.DataFrame
    bagging: pd.DataFrame
    oob: pd.DataFrame
    diversity: pd.DataFrame
    coverage: pd.DataFrame
    scopes: dict = field(default_factory=dict)


def detect_mode(cov):
    if not cov["ext"].all():
        missing = cov.loc[~cov["ext"], ["learner", "dataset", "sample_size"]].values.tolist()
        raise RuntimeError(f"Results missing for {len(missing)} learner x scenario pairs: {missing[:10]} ...")
    return "A"


def load(force=False):
    paths = build_cache(force)
    rd = {n: pd.read_parquet(p) for n, p in paths.items()}
    mode = detect_mode(rd["coverage"])
    data = Data(mode, rd["singles"], rd["bagging"], rd["oob"], rd["diversity"], rd["coverage"])
    scen_all = [(ds, s) for ds, s, _ in all_scenarios()]
    data.scopes["all"] = Scope("all40", scen_all, cfg.K_ENS, cfg.RULES,
                               "40 scenarios (8 systems x 5 sample sizes), 30 runs each.")
    return data


def in_scope(df, scope):
    keys = pd.DataFrame(scope.scenarios, columns=["dataset", "sample_size"])
    out = df.merge(keys, on=["dataset", "sample_size"])
    if "k" in out.columns:
        out = out[out["k"].isin(scope.ks)]
    if "rule" in out.columns:
        out = out[out["rule"].isin(scope.rules)]
    return out
