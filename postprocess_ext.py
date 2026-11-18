# Combines the bagging members (MEAN, MEDIAN, IRWM) and computes OOB metrics and diversity.

import argparse
import json
import multiprocessing
import os
import time
import traceback

import numpy as np
import pandas as pd

from ext_common import (LEARNERS, MAX_N, METRICS, RESULTS_EXT, RULES, compute_metrics, log_line,
                        scen_dir, scenarios)

MIN_OOB_ROWS = 2


def member_oob_metrics(y_train, oob_pred):
    n_members = oob_pred.shape[0]
    out = np.full((n_members, len(METRICS)), np.nan)
    n_oob = np.zeros(n_members, dtype=int)
    for m in range(n_members):
        mask = ~np.isnan(oob_pred[m])
        n_oob[m] = int(mask.sum())
        if n_oob[m] >= MIN_OOB_ROWS:
            met = compute_metrics(y_train[mask], oob_pred[m, mask].astype(np.float64))
            out[m] = [met[k] for k in METRICS]
    return out, n_oob


def _ordinal_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def irwm_weights(oob_metrics_k):
    k = oob_metrics_k.shape[0]
    if k == 1:
        return np.ones(1), 0, False
    valid = ~np.isnan(oob_metrics_k).any(axis=1)
    n_edge = int((~valid).sum())
    if valid.sum() == 0:
        return np.full(k, 1.0 / k), n_edge, True
    vi = np.where(valid)[0]
    borda = np.zeros(k)
    for j in range(len(METRICS)):
        borda[vi] += _ordinal_ranks(oob_metrics_k[vi, j])
    if n_edge:
        borda[~valid] = np.median(borda[vi])
    mae = np.where(valid, oob_metrics_k[:, METRICS.index("MAE")], np.inf)
    order = np.lexsort((mae, borda))
    ranks = np.empty(k)
    ranks[order] = np.arange(1, k + 1)
    w = 1.0 / ranks
    return w / w.sum(), n_edge, False


def combine(preds, rule, weights=None):
    if rule == "mean":
        return np.mean(preds, axis=0)
    if rule == "median":
        return np.median(preds, axis=0)
    return np.average(preds, axis=0, weights=weights)


def combine_oob(oob_k, rule, weights):
    valid = ~np.isnan(oob_k)
    covered = valid.any(axis=0)
    p = oob_k[:, covered].astype(np.float64)
    v = valid[:, covered]
    if rule == "mean":
        out = np.nanmean(p, axis=0)
    elif rule == "median":
        out = np.nanmedian(p, axis=0)
    else:
        w = np.where(v, weights[:, None], 0.0)
        out = np.nansum(np.where(v, p, 0.0) * w, axis=0) / w.sum(axis=0)
    return out, covered


def _pairwise_mean_corr(m):
    if m.shape[0] < 2:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(m)
    vals = c[np.triu_indices(m.shape[0], 1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def diversity_measures(preds, y):
    errors = preds - y
    mean_pred = preds.mean(axis=0)
    ambiguity = float(((preds - mean_pred) ** 2).mean())
    member_mse = float((errors ** 2).mean())
    return {
        "err_corr": _pairwise_mean_corr(errors),
        "pred_corr": _pairwise_mean_corr(preds),
        "ambiguity": ambiguity,
        "member_mse": member_mse,
        "ens_mse": float(((mean_pred - y) ** 2).mean()),
        "rel_ambiguity": ambiguity / member_mse if member_mse > 0 else np.nan,
    }


def outputs_for(model, ds, s):
    d = scen_dir(ds, s)
    files = [os.path.join(d, f"{model}_bagging_N{k}_{rule}.json") for k in range(1, MAX_N + 1) for rule in RULES]
    files += [os.path.join(d, f"{model}_bagging_{x}.parquet") for x in ["oob_selection", "diversity", "member_metrics"]]
    return files


def _json_result(per_run, weights=None):
    metrics = {k: [float(m[k]) for m in per_run] for k in METRICS}
    res = {"Metrics": metrics}
    res.update({f"Mean_{k}": float(np.mean(v)) for k, v in metrics.items()})
    res.update({f"Median_{k}": float(np.median(v)) for k, v in metrics.items()})
    res["Runs"] = [{} for _ in per_run]
    if weights is not None:
        res["IRWM_weights"] = [w.tolist() for w in weights]
    return res


def process(model, ds, s, force=False):
    npz = os.path.join(scen_dir(ds, s), f"{model}_bagging_members.npz")
    if not os.path.exists(npz):
        return None
    outs = outputs_for(model, ds, s)
    if not force and all(os.path.exists(f) and os.path.getmtime(f) >= os.path.getmtime(npz) for f in outs):
        return f"Skipped {model} {ds} {s}"
    t0 = time.time()
    d = np.load(npz)
    P, Y = d["member_test_pred"], d["y_test"]
    OOB, YT = d["oob_pred"], d["y_train"]
    R, M = P.shape[0], P.shape[1]

    per_run = {(k, rule): [] for k in range(1, M + 1) for rule in RULES}
    irwm_w = {k: [] for k in range(1, M + 1)}
    oob_rows, div_rows, mem_rows, edge_rows = [], [], [], []
    max_identity_rel = 0.0

    for r in range(R):
        mo, n_oob = member_oob_metrics(YT[r], OOB[r])
        for m in range(M):
            met_test = compute_metrics(Y[r], P[r, m])
            for j, key in enumerate(METRICS):
                mem_rows.append((r, m, "test", key, float(met_test[key]), len(Y[r])))
                mem_rows.append((r, m, "oob", key, float(mo[m, j]), int(n_oob[m])))
        for k in range(1, M + 1):
            w, n_edge, fallback = irwm_weights(mo[:k])
            irwm_w[k].append(w)
            if n_edge or fallback:
                edge_rows.append(f"{model}\t{ds}\t{s}\t{r}\t{k}\t{n_edge}\t{int(fallback)}")
            for rule in RULES:
                pred = combine(P[r, :k], rule, w)
                per_run[(k, rule)].append(compute_metrics(Y[r], pred))
                oob_p, covered = combine_oob(OOB[r, :k], rule, w)
                cov = float(covered.mean())
                met = compute_metrics(YT[r][covered], oob_p) if covered.any() else {key: np.nan for key in METRICS}
                for key in METRICS:
                    oob_rows.append((r, k, rule, key, float(met[key]), cov))
            dm = diversity_measures(P[r, :k], Y[r])
            mre_members = np.array([compute_metrics(Y[r], P[r, m])["MRE"] for m in range(k)])
            dm["acc_spread"] = float(mre_members.mean() / mre_members.min()) if mre_members.min() > 0 else np.nan
            ident = dm["member_mse"] - dm["ambiguity"] - dm["ens_mse"]
            dm["identity_residual"] = ident
            if dm["ens_mse"] > 0:
                max_identity_rel = max(max_identity_rel, abs(ident) / max(dm["member_mse"], 1e-300))
            div_rows.append(dict(run=r, k=k, **dm))

    sd = scen_dir(ds, s)
    for (k, rule), runs in per_run.items():
        res = _json_result(runs, irwm_w[k] if rule == "irwm" else None)
        with open(os.path.join(sd, f"{model}_bagging_N{k}_{rule}.json"), "w") as f:
            json.dump(res, f, indent=2)
    pd.DataFrame(oob_rows, columns=["run", "k", "rule", "metric", "value", "oob_coverage"]).to_parquet(
        os.path.join(sd, f"{model}_bagging_oob_selection.parquet"), index=False)
    pd.DataFrame(div_rows).to_parquet(os.path.join(sd, f"{model}_bagging_diversity.parquet"), index=False)
    pd.DataFrame(mem_rows, columns=["run", "member", "split", "metric", "value", "n_rows"]).to_parquet(
        os.path.join(sd, f"{model}_bagging_member_metrics.parquet"), index=False)

    if edge_rows:
        log_line("oob_edge_cases.log", "\n".join(edge_rows))
    log_line("postprocess.log", f"{model}\t{ds}\t{s}\tidentity_max_rel={max_identity_rel:.2e}\t"
                                f"edge_rows={len(edge_rows)}\t{time.time() - t0:.1f}s")
    return f"Done {model} {ds} {s} ({time.time() - t0:.1f}s)"


def _task(t):
    try:
        return process(*t)
    except Exception:
        traceback.print_exc()
        return f"FAILED {t}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all")
    p.add_argument("--dataset", default="all")
    p.add_argument("--processes", type=int, default=max(1, os.cpu_count() - 3))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    models = LEARNERS if a.model == "all" else a.model.split(",")
    datasets = None if a.dataset == "all" else a.dataset.split(",")
    tasks = [(m, ds, s, a.force) for ds, s in scenarios(datasets) for m in models]
    with multiprocessing.Pool(min(a.processes, len(tasks))) as pool:
        for res in pool.imap_unordered(_task, tasks, chunksize=1):
            if res:
                print(res, flush=True)


if __name__ == "__main__":
    main()
