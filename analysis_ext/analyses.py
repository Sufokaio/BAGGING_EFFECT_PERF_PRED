# Tables and figures of RQ1-RQ5.

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import bias_variance as bvmod
import config as cfg
from common import (KEYS, bagging_long, config_rows, latex_table, learner_style, save_fig, scen_median, select_oob,
                    singles_long, stack, tag, write_csv, write_tex)
from loaders import in_scope
from stats import borda_unique, rel_improvement, run_sk

CONFS = cfg.CONFIGS


def _cache(name, fn):
    path = os.path.join(cfg.CACHE_DIR, f"{name}_k{cfg.K_ENS[0]}-{cfg.K_ENS[-1]}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path, index=False)
    return df


def _legend_below(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, fontsize=cfg.FONT_SIZE,
               handlelength=2.2, columnspacing=1.0)


def sk_single_vs_bagging(data, scope, metrics=cfg.METRICS):
    def build():
        df = pd.concat([singles_long(data, scope), bagging_long(data, scope)], ignore_index=True)
        df = df[df.metric.isin(metrics)]
        return run_sk(df, KEYS + ["metric"], "treatment", "value")
    return _cache(f"sk_rq1_{tag(scope, data.mode)}", build)


def mae_means(data, scope):
    df = pd.concat([singles_long(data, scope), bagging_long(data, scope)], ignore_index=True)
    df = df[df.metric == "MAE"].copy()
    by = KEYS + ["treatment"]
    m = df.groupby(by, as_index=False)["value"].mean().rename(columns={"value": "mean_MAE"})
    runs = df.groupby(by)["value"].size()
    assert (runs == cfg.NUM_RUNS).all(), runs.value_counts()
    return m


def rq1_t1(data, scope):
    sk = sk_single_vs_bagging(data, scope)
    sk["is_single"] = sk.treatment == "Single"
    mre = sk[sk.metric == "MRE"]
    borda = borda_unique(sk, mae_means(data, scope), "treatment", KEYS)
    rows = []
    for L in cfg.LEARNERS:
        m, b = mre[mre.learner == L], borda[borda.learner == L]
        n = m.groupby(["dataset", "sample_size"]).ngroups
        single_best = int(m[m.is_single & (m.sk_rank == 1)].shape[0])
        bag_best = int(m[~m.is_single & (m.sk_rank == 1)].groupby(["dataset", "sample_size"]).ngroups)
        bag_only = int(m.groupby(["dataset", "sample_size"]).apply(
            lambda g: ((g[~g.is_single].sk_rank == 1).any()) and (g[g.is_single].sk_rank.min() > 1)).sum())
        single_b1 = int(b[(b.treatment == "Single") & (b.borda_rank == 1)].shape[0])
        bag_b1 = int(b[(b.treatment != "Single") & (b.borda_rank == 1)].groupby(["dataset", "sample_size"]).ngroups)
        both_mre = single_best + bag_best - n
        both_b1 = int(b[b.borda_rank == 1].groupby(["dataset", "sample_size"]).treatment.apply(
            lambda s: (s == "Single").any() and (s != "Single").any()).sum())
        rows.append(dict(learner=L, n=n, single_best_mre=single_best, bag_best_mre=bag_best,
                         bag_only_best_mre=bag_only, single_only_best_mre=single_best - both_mre,
                         single_borda1=single_b1, bag_borda1=bag_b1,
                         single_only_borda1=single_b1 - both_b1, bag_only_borda1=bag_b1 - both_b1))
    t = pd.DataFrame(rows)
    assert (t.single_only_best_mre + t.bag_only_best_mre + (t.single_best_mre - t.single_only_best_mre) == t.n).all()
    assert (t.single_borda1 + t.bag_borda1 == t.n).all()
    write_csv(t, f"T1_best_cluster_{tag(scope, data.mode)}.csv")
    _t1_variants(t, scope, data.mode)
    t1_per_system(mre, borda, scope, data.mode)
    return t


BORDA_TIE_NOTE = (r"rank 1 is unique per scenario: ties on the Borda sum are broken by the mean MAE over the 30 "
                  r"runs (then by the model identifier).")


def _top_flags(mre, borda):
    f = mre[mre.sk_rank == 1].assign(single=lambda d: d.treatment == "Single").groupby(KEYS).single.agg(
        s_mre="any", b_mre=lambda s: (~s).any())
    g = borda[borda.borda_rank == 1].assign(single=lambda d: d.treatment == "Single").groupby(KEYS).single.agg(
        s_b1="any", b_b1=lambda s: (~s).any())
    return f.join(g, how="outer").fillna(False).astype(bool).reset_index()


def t1_per_system(mre, borda, scope, mode):
    f = _top_flags(mre, borda)
    for a, b in [("s_mre", "b_mre"), ("s_b1", "b_b1")]:
        f[a + "_only"], f[b + "_only"] = f[a] & ~f[b], f[b] & ~f[a]
    cols = ["s_mre", "s_mre_only", "b_mre", "b_mre_only", "s_b1", "s_b1_only", "b_b1", "b_b1_only"]
    assert (f.s_b1 ^ f.b_b1).all()
    t = f.groupby("dataset")[cols].sum().reindex(cfg.DATASETS)
    t.insert(0, "n", f.groupby("dataset").size().reindex(cfg.DATASETS))
    write_csv(t.reset_index(), f"T1v1_per_system_{tag(scope, mode)}.csv")

    labels = ["Single", stack("Single", "only"), stack("Any", "bagging"), stack("Bagging", "only")]
    header = [r"System & \multicolumn{4}{c}{SK best cluster (MRE)} & \multicolumn{4}{c}{Borda rank 1 (4 metrics)} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}", r" & " + " & ".join(labels * 2) + r" \\"]
    body = [[cfg.DATASET_LABELS[ds]] + [int(r[c]) for c in cols] for ds, r in t.iterrows()]
    n_sys = int(t.n.iloc[0])
    note = (f"Number of learner $\\times$ sample-size scenarios per system (of {n_sys}: 6 learners $\\times$ "
            f"{n_sys // len(cfg.LEARNERS)} sizes). " + r"Same SK and Borda "
            r"as the per-learner table (Single and its 27 bagging variants); Borda " + BORDA_TIE_NOTE + " "
            r"Single / Any bagging: at the top (SK best cluster: ties counted for both); Single only / Bagging only: at "
            r"the top and the other side not.")
    write_tex(f"T1v1_per_system_{tag(scope, mode)}.tex",
              latex_table("l" + "c" * 8, header, body, [note], colsep="5pt"))
    return t


def _t1_variants(t, scope, mode):
    n_sc = int(t.n.iloc[0])
    labels = ["Single", stack("Single", "only"), stack("Any", "bagging"), stack("Bagging", "only")]
    cm = ["single_best_mre", "single_only_best_mre", "bag_best_mre", "bag_only_best_mre"]
    cb = ["single_borda1", "single_only_borda1", "bag_borda1", "bag_only_borda1"]
    nc = len(labels)
    header = [r"Learner & \multicolumn{" + str(nc) + r"}{c}{SK best cluster (MRE)} & \multicolumn{" + str(nc)
              + r"}{c}{Borda rank 1 (4 metrics)} \\",
              r"\cmidrule(lr){2-" + str(1 + nc) + r"}\cmidrule(lr){" + str(2 + nc) + "-" + str(1 + 2 * nc) + "}",
              r" & " + " & ".join(labels * 2) + r" \\"]
    body = [[r.learner] + [int(getattr(r, c)) for c in cm + cb] for r in t.itertuples()]
    note = (f"Number of scenarios (of {n_sc} per learner). " + r"SK: Scott-Knott on the 30 per-run MRE values over Single and the 27 bagging variants "
            r"of the learner (bootstrap $t$, conf 0.01, $A_{12}\geq0.60$); Borda: sum of SK ranks over MRE, MAE, "
            r"MBRE, MIBRE; " + BORDA_TIE_NOTE + " ")
    note += (r"Single / Any bagging: Single / at least one bagging variant at the top (SK best cluster: ties "
             r"counted for both); Single only / Bagging only: at the top and the other side not. ")
    write_tex(f"T1v1_best_cluster_{tag(scope, mode)}.tex",
              latex_table("l" + "c" * (2 * nc), header, body, [note], colsep="5pt"))


def improvement_rows(data, scope):
    sing = scen_median(singles_long(data, scope))
    out = []
    for conf in CONFS:
        bag = scen_median(config_rows(data, scope, conf))
        m = sing.merge(bag, on=KEYS + ["level", "metric"], suffixes=("_single", "_bag"))
        m["config"] = conf
        out.append(m)
    m = pd.concat(out, ignore_index=True)
    m["win"] = m.value_bag < m.value_single
    m["rel_imp"] = rel_improvement(m.value_single, m.value_bag)
    return m


def k_best_counts(pool, scope, mode, rules):
    pool = pool.assign(top=lambda d: d.sk_rank == 1)
    single_top = pool[pool.treatment == "Single"]
    top = pool[pool.treatment != "Single"].copy()
    top["k"] = top.treatment.str.extract(r"Bag-k(\d+)-")[0].astype(int)
    top["rule"] = top.treatment.str.split("-").str[2]
    ks = cfg.K_ENS
    cnt = top.groupby(["learner", "rule", "k"]).top.sum().unstack("k").reindex(columns=ks)
    s_cnt = single_top.groupby("learner").top.sum()
    a_cnt = top.groupby(KEYS).top.any().groupby("learner").sum()
    cnt.insert(0, -1, [int(a_cnt[L]) for L, _ in cnt.index])
    cnt.insert(0, 0, [int(s_cnt[L]) for L, _ in cnt.index])
    write_csv(cnt.reset_index(), f"T_k_best_cluster_{tag(scope, mode)}.csv")
    n_sc = len(scope.scenarios)

    fig, axes = plt.subplots(1, len(rules), figsize=(cfg.FULL_W, 0.46 * cfg.FULL_W), sharey=True)
    for ax, rule, lab in zip(np.atleast_1d(axes), rules, "abc"):
        for L in cfg.LEARNERS:
            ax.plot(ks, 100 * cnt.loc[(L, rule), ks].to_numpy() / n_sc, label=L, **learner_style(L, 2.6))
        ax.set_title(f"({lab}) {rule}", loc="left", fontsize=cfg.FONT_SIZE)
        ax.set_xlabel("$k$")
        ax.set_xticks(ks)
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(alpha=0.3, lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("Share of scenarios\nin best cluster")
    _legend_below(fig, *np.atleast_1d(axes)[0].get_legend_handles_labels(), ncol=len(cfg.LEARNERS))
    save_fig(fig, f"F3_k_best_cluster_{tag(scope, mode)}")
    return cnt


def rq2(data, scope):
    pool = sk_single_vs_bagging(data, scope)
    return k_best_counts(pool[pool.metric == "MRE"], scope, data.mode, scope.rules)


def rule_rows(data, scope):
    sel = select_oob(data, scope, per_rule=True)[KEYS + ["k", "rule"]]
    b = bagging_long(data, scope).merge(sel, on=KEYS + ["k", "rule"])
    return b, sel


def rq3(data, scope):
    b, sel = rule_rows(data, scope)
    write_csv(sel, f"T_rules_k_{tag(scope, data.mode)}.csv")
    sk = _cache(f"sk_rq3_{tag(scope, data.mode)}", lambda: run_sk(b[b.metric == "MRE"], KEYS, "rule", "value"))
    med = scen_median(b, extra=("rule",))
    med = med[med.metric == "MRE"]
    agg = med.groupby(["rule", "learner"]).value.median().unstack().reindex(index=scope.rules, columns=cfg.LEARNERS)
    rk = sk.groupby(["rule", "learner"]).sk_rank.mean().unstack().reindex(index=scope.rules, columns=cfg.LEARNERS)
    out = pd.concat({"median_mre": agg.stack(), "mean_sk_rank": rk.stack()}, axis=1).reset_index()
    write_csv(out, f"T_rules_{tag(scope, data.mode)}.csv")
    header = [r"Rule & " + " & ".join(cfg.LEARNERS) + r" \\"]
    body = []
    for rule in scope.rules:
        best = [rk[L].min() for L in cfg.LEARNERS]
        cells = [rule]
        for j, L in enumerate(cfg.LEARNERS):
            r_txt = f"({rk.loc[rule, L]:.2f})"
            if np.isclose(rk.loc[rule, L], best[j]):
                r_txt = r"\textbf{" + r_txt + "}"
            cells.append(stack(f"{agg.loc[rule, L]:.1f}", r_txt))
        body.append(cells)
    notes = [r"Per rule, $k$ chosen per scenario by OOB Borda among that rule's ensembles. Cell: median MRE (\%) over "
             r"scenarios; in brackets the mean SK rank of the rule (SK per scenario over the three rules, 30 runs; "
             r"bold: lowest). Scope: " + scope.note]
    write_tex(f"T_rules_{tag(scope, data.mode)}.tex", latex_table("l" + "c" * len(cfg.LEARNERS), header, body, notes,
                                                                colsep="9pt"))
    return agg, rk


def rq4(data, scope):
    bv = bvmod.load().merge(pd.DataFrame(scope.scenarios, columns=["dataset", "sample_size"]),
                            on=["dataset", "sample_size"])
    bv = bv[(bv.rule == "Single") | bv.k.isin(scope.ks)]
    single = bv[bv.rule == "Single"][KEYS + ["rel_var", "rel_bias2"]]
    ens = bv[bv.rule != "Single"].merge(single, on=KEYS, suffixes=("", "_single"))
    ens["var_change"] = 100 * (ens.rel_var - ens.rel_var_single) / ens.rel_var_single
    ens["bias2_change"] = 100 * (ens.rel_bias2 - ens.rel_bias2_single) / ens.rel_bias2_single
    vk = ens.groupby(["learner", "rule", "k"])[["var_change", "bias2_change"]].median().reset_index()
    write_csv(vk, f"T4d_variance_vs_k_{tag(scope, data.mode)}.csv")

    fig, axes = plt.subplots(1, len(cfg.RULES), figsize=(cfg.FULL_W, 0.46 * cfg.FULL_W), sharey=True)
    for ax, rule, lab in zip(axes, cfg.RULES, "abc"):
        for L in cfg.LEARNERS:
            c = vk[(vk.learner == L) & (vk.rule == rule)]
            ax.plot(c.k, c.var_change, label=L, **learner_style(L, 2.6))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(cfg.K_ENS)
        ax.set_xlabel("$k$")
        ax.set_title(f"({lab}) {rule}", loc="left", fontsize=cfg.FONT_SIZE)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Relative variance change")
    axes[0].yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0%" if round(v) == 0 else ("+" if v > 0 else "−") + f"{abs(v):.0f}%"))
    _legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=len(cfg.LEARNERS))
    save_fig(fig, f"F6_bias_variance_{tag(scope, data.mode)}")
    return vk


def rq5(data, scope, m):
    div = in_scope(data.diversity, scope)
    div = div[div.k.isin(scope.ks)]
    per = div.groupby(KEYS + ["k"], as_index=False)[["err_corr", "pred_corr", "ambiguity", "member_mse", "ens_mse",
                                                     "rel_ambiguity", "acc_spread"]].mean()

    k10 = per[per.k == 10].copy()
    sing = scen_median(singles_long(data, scope))
    bag = scen_median(bagging_long(data, scope), extra=("k", "rule"))
    g = bag[(bag.rule == "MEAN") & (bag.k == 10) & (bag.metric == "MRE")].merge(
        sing[sing.metric == "MRE"], on=KEYS + ["level", "metric"], suffixes=("_bag", "_single"))
    g["gain"] = rel_improvement(g.value_single, g.value_bag)
    k10 = k10.merge(g[KEYS + ["gain"]], on=KEYS)
    mm = m[m.metric == "MRE"]
    for conf, col in [("Bag-k10-MEDIAN", "gain_k10_median"), ("Bag-OOB", "gain_oob")]:
        k10 = k10.merge(mm[mm.config == conf][KEYS + ["rel_imp"]].rename(columns={"rel_imp": col}), on=KEYS)
    write_csv(k10, f"T3_diversity_{tag(scope, data.mode)}.csv")

    measures = [("err_corr", "Error corr."), ("pred_corr", "Pred. corr."), ("rel_ambiguity", "Rel. ambiguity"),
                ("acc_spread", "Acc. spread")]
    header = [r"Learner & " + " & ".join(t for _, t in measures) + r" \\"]
    body = [[L] + [f"{np.median(k10[k10.learner == L][c]):.2f}" for c, _ in measures] for L in cfg.LEARNERS]
    notes = [r"Diversity of the $k=10$ ensemble members, median over the scenarios (per scenario: mean over the 30 "
             r"runs). Error / Pred. corr.: mean pairwise correlation of the members' errors / predictions on the test "
             r"set (1: identical members). Rel. ambiguity: how far the members spread around the ensemble prediction, "
             r"relative to their average error (Krogh--Vedelsby; 0: no diversity). Acc. spread: mean member MRE / best "
             r"member MRE (1: equally accurate members)."]
    write_tex(f"T3_diversity_{tag(scope, data.mode)}.tex", latex_table("lcccc", header, body, notes, colsep="9pt"))
    return k10
