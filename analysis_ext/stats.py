# Scott-Knott runner, Borda ranking and OOB selection.

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as cfg
from sk_impl import scott_knott


def _sk_one(key, groups):
    names = list(groups)
    if len(names) == 1:
        return key, {names[0]: 1}, False
    try:
        result = scott_knott([(n, list(v)) for n, v in groups.items()], **cfg.SK_KWARGS)
        return key, {name: int(rank) for rank, name, *_ in result}, False
    except Exception:
        return key, {n: 1 for n in names}, True


def run_sk(df, by, competitor_col="treatment", value_col="value", n_jobs=None):
    tasks = []
    for key, grp in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        groups = {name: g.sort_values("run")[value_col].to_numpy() for name, g in grp.groupby(competitor_col, sort=True)}
        tasks.append((key, groups))
    results = Parallel(n_jobs=n_jobs or cfg.N_JOBS)(delayed(_sk_one)(k, g) for k, g in tasks)
    rows, failed = [], 0
    for key, ranks, fail in results:
        failed += int(fail)
        for name, rank in ranks.items():
            rows.append(dict(zip(by, key), **{competitor_col: name, "sk_rank": rank}))
    if failed:
        print(f"  Scott-Knott failed in {failed} of {len(tasks)} groups (all competitors set to rank 1)")
    return pd.DataFrame(rows)


def borda_unique(sk, mae_means, competitor_col="treatment", by=None):
    by = by or []
    total = sk.groupby(by + [competitor_col], as_index=False)["sk_rank"].sum()
    total = total.rename(columns={"sk_rank": "borda_total"})
    total = total.merge(mae_means[by + [competitor_col, "mean_MAE"]], on=by + [competitor_col], how="left",
                        validate="1:1")
    assert total["mean_MAE"].notna().all(), "mean MAE missing for some competitors"
    total = total.sort_values(by + ["borda_total", "mean_MAE", competitor_col]).reset_index(drop=True)
    grp = total.groupby(by) if by else total.groupby(np.zeros(len(total)))
    total["borda_rank"] = grp.cumcount() + 1
    return total


def oob_borda(means, group_cols, id_cols):
    df = means.sort_values(group_cols + ["oob_MAE"] + id_cols).reset_index(drop=True)
    score = 0
    for m in cfg.BORDA_METRICS:
        df[f"oob_rank_{m}"] = df.groupby(group_cols)[f"oob_{m}"].rank(method="first")
        score = score + df[f"oob_rank_{m}"]
    df["oob_borda_score"] = score
    df = df.sort_values(group_cols + ["oob_borda_score", "oob_MAE"] + id_cols).reset_index(drop=True)
    df["oob_borda_rank"] = df.groupby(group_cols).cumcount() + 1
    return df


def rel_improvement(err_single, err_bag):
    return 100.0 * (err_single - err_bag) / err_single
