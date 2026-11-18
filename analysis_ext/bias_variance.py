# Bias-variance decomposition over runs.

import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as cfg

sys.path.insert(0, cfg.ROOT)
from ext_common import run_split

MIN_RUNS = 2


def _mre(y, p, single=False):
    if single:
        return np.mean(np.abs((y - p) / y)) * 100
    return np.mean(np.abs(y - p) / (np.abs(y) + 1e-8)) * 100


def _json(path):
    with open(path) as f:
        j = json.load(f)
    return j[0] if isinstance(j, list) else j


def _decompose(pred, cid, y_cfg, valid):
    m = np.bincount(cid, minlength=len(y_cfg)).astype(float)
    mean = np.bincount(cid, weights=pred, minlength=len(y_cfg)) / np.where(m > 0, m, 1)
    dev = pred - mean[cid]
    var = np.bincount(cid, weights=dev ** 2, minlength=len(y_cfg)) / np.where(m > 0, m, 1)
    bias2 = (mean - y_cfg) ** 2
    pair_ok = valid[cid]
    mse_pairs = np.mean((pred[pair_ok] - y_cfg[cid][pair_ok]) ** 2)
    mse_terms = np.sum(m[valid] * (bias2[valid] + var[valid])) / np.sum(m[valid])
    return bias2, var, abs(mse_terms - mse_pairs) / max(mse_pairs, 1e-300)


def _one(ds, s, level, learner, y_all):
    n_total = len(y_all)
    sdir = os.path.join(cfg.RESULTS_EXT, ds, str(s))
    test_idx = [run_split(n_total, s, r)[1] for r in range(cfg.NUM_RUNS)]
    mem = np.load(os.path.join(sdir, f"{learner}_bagging_members.npz"))
    single = _json(os.path.join(sdir, f"{learner}_single_metrics.json"))
    bad, ulp_runs, ulp_max = [], 0, 0.0
    for r in range(cfg.NUM_RUNS):
        yt = y_all[test_idx[r]]
        ys = np.asarray(single["Runs"][r]["y_test"], dtype=float)
        if not np.array_equal(mem["y_test"][r], yt) or len(ys) != len(yt):
            bad.append(r)
        elif not np.array_equal(ys, yt):
            rel = float(np.max(np.abs(ys - yt) / np.abs(yt)))
            if rel > 1e-12:
                bad.append(r)
            ulp_runs += 1
            ulp_max = max(ulp_max, rel)
    if bad:
        raise RuntimeError(f"y_test mismatch {learner} {ds}/{s} runs {bad}")

    cid = np.concatenate(test_idx)
    y_cfg = y_all.astype(float)
    m = np.bincount(cid, minlength=n_total)
    valid = m >= MIN_RUNS
    preds = {("Single", 0): np.vstack([np.asarray(single["Runs"][r]["y_pred"], dtype=float).ravel()
                                       for r in range(cfg.NUM_RUNS)])}
    stored = {("Single", 0): np.asarray(single["Metrics"]["MRE"], dtype=float)}
    P = mem["member_test_pred"]
    for k in cfg.K_ALL:
        preds[("MEAN", k)] = P[:, :k].mean(axis=1)
        preds[("MEDIAN", k)] = np.median(P[:, :k], axis=1)
        jw = _json(os.path.join(sdir, f"{learner}_bagging_N{k}_irwm.json"))
        w = np.asarray(jw["IRWM_weights"], dtype=float)
        preds[("IRWM", k)] = np.einsum("rm,rmi->ri", w, P[:, :k])
        stored[("IRWM", k)] = np.asarray(jw["Metrics"]["MRE"], dtype=float)
        for rule in ["mean", "median"]:
            stored[(rule.upper(), k)] = np.asarray(
                _json(os.path.join(sdir, f"{learner}_bagging_N{k}_{rule}.json"))["Metrics"]["MRE"], dtype=float)

    rows, max_mre_diff = [], 0.0
    ref = None
    for (rule, k), pr in preds.items():
        mre = np.array([_mre(y_all[test_idx[r]], pr[r], single=(rule == "Single")) for r in range(cfg.NUM_RUNS)])
        diff = float(np.max(np.abs(mre - stored[(rule, k)]) / np.maximum(np.abs(stored[(rule, k)]), 1e-12)))
        max_mre_diff = max(max_mre_diff, diff)
        if diff > 1e-9:
            raise RuntimeError(f"MRE mismatch {learner} {ds}/{s} {rule} k={k}: rel diff {diff:.2e}")
        bias2, var, resid = _decompose(pr.ravel(), cid, y_cfg, valid)
        rv, rb = var[valid] / y_cfg[valid] ** 2, bias2[valid] / y_cfg[valid] ** 2
        if ref is None:
            ref = (var[valid], bias2[valid])
        rows.append(dict(learner=learner, dataset=ds, sample_size=s, level=level, rule=rule, k=k,
                         n_configs=int(valid.sum()), n_excluded=int((~valid).sum()),
                         runs_min=int(m[valid].min()), runs_median=float(np.median(m[valid])),
                         rel_var=float(np.median(rv)), rel_bias2=float(np.median(rb)),
                         rel_var_mean=float(np.mean(rv)), rel_bias2_mean=float(np.mean(rb)),
                         share_lower_var=float(np.mean(var[valid] < ref[0])),
                         share_lower_bias2=float(np.mean(bias2[valid] < ref[1])),
                         identity_residual=resid, max_mre_rel_diff=diff,
                         ytest_single_roundoff_runs=ulp_runs, ytest_single_roundoff_max_rel=ulp_max))
    return rows


def build():
    tasks = []
    for ds in cfg.DATASETS:
        y_all = pd.read_csv(os.path.join(cfg.ROOT, "data", f"{ds}.csv")).iloc[:, -1].to_numpy(dtype=float)
        for level, s in enumerate(cfg.DATASET_REGISTRY[ds], start=1):
            for L in cfg.LEARNERS:
                tasks.append((ds, s, level, L, y_all))
    out = Parallel(n_jobs=cfg.N_JOBS)(delayed(_one)(*t) for t in tasks)
    return pd.DataFrame([r for rows in out for r in rows])


def load():
    path = os.path.join(cfg.CACHE_DIR, "bias_variance.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df
