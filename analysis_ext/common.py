# Treatments, OOB selection and table / figure writers.

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
from loaders import in_scope
from stats import oob_borda

KEYS = ["learner", "dataset", "sample_size"]


def singles_long(data, scope):
    s = in_scope(data.singles, scope).copy()
    s["treatment"], s["k"], s["rule"] = "Single", 0, "-"
    return s


def bagging_long(data, scope):
    b = in_scope(data.bagging, scope).copy()
    b["treatment"] = "Bag-k" + b["k"].astype(str) + "-" + b["rule"]
    return b


def oob_means(data, scope):
    o = in_scope(data.oob, scope)
    m = o.pivot_table(index=KEYS + ["k", "rule"], columns="metric", values="value", aggfunc="mean")
    m.columns = [f"oob_{c}" for c in m.columns]
    cov = o.groupby(KEYS + ["k", "rule"])["oob_coverage"].mean().rename("oob_coverage")
    return m.join(cov).reset_index()


def select_oob(data, scope, per_rule=False):
    means = oob_means(data, scope)
    group = KEYS + (["rule"] if per_rule else [])
    ranked = oob_borda(means, group, ["k", "rule"])
    cols = list(dict.fromkeys(group + ["k", "rule", "oob_coverage"]))
    return ranked[ranked.oob_borda_rank == 1][cols].drop_duplicates(group)


def config_sel(data, scope, conf):
    if conf == "Bag-OOB":
        return select_oob(data, scope)[KEYS + ["k", "rule"]]
    keys = pd.DataFrame([(L, ds, s) for L in cfg.LEARNERS for ds, s in scope.scenarios], columns=KEYS)
    keys["k"], keys["rule"] = 10, "MEDIAN"
    return keys


def config_rows(data, scope, conf):
    b = in_scope(data.bagging, scope).merge(config_sel(data, scope, conf), on=KEYS + ["k", "rule"]).copy()
    b["variant"] = "Bag-k" + b["k"].astype(str) + "-" + b["rule"]
    b["treatment"] = conf
    return b


def learner_style(L, ms=3.0):
    return dict(color=cfg.MODEL_COLORS[L], marker=cfg.MODEL_MARKERS[L], linestyle=cfg.MODEL_LINES[L], ms=ms,
                mfc=cfg.MODEL_COLORS[L], mec=cfg.MODEL_COLORS[L], lw=1.0)


def scen_median(df, extra=()):
    return df.groupby(KEYS + ["level", "metric"] + list(extra), as_index=False)["value"].median()


def tag(scope, mode):
    return f"{scope.label}_mode{mode}"


def write_tex(name, lines):
    os.makedirs(cfg.TAB_DIR, exist_ok=True)
    with open(os.path.join(cfg.TAB_DIR, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(df, name):
    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    df.to_csv(os.path.join(cfg.CSV_DIR, name), index=False)


def latex_table(col_spec, header_rows, body, notes=None, size=r"\footnotesize", colsep=None):
    lines = [size] + ([r"\begingroup\setlength{\tabcolsep}{" + colsep + "}"] if colsep else [])
    lines += [r"\begin{tabular}{" + col_spec + "}", r"\toprule"] + header_rows + [r"\midrule"]
    for row in body:
        lines.append(row if isinstance(row, str) else " & ".join(str(c) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    n_cols = len([c for c in col_spec if c.isalpha()])
    for note in notes or []:
        lines.append(r"\multicolumn{" + str(n_cols) + r"}{p{11.5cm}}{\scriptsize " + note + r"} \\")
    lines.append(r"\end{tabular}")
    if colsep:
        lines.append(r"\endgroup")
    return lines


def stack(top, bottom):
    return r"\begin{tabular}[t]{@{}c@{}}" + top + r"\\{" + bottom + r"}\end{tabular}"


def setup_style():
    plt.rcParams.update({
        "font.size": cfg.FONT_SIZE, "axes.titlesize": cfg.FONT_SIZE, "axes.labelsize": cfg.FONT_SIZE,
        "xtick.labelsize": cfg.FONT_SIZE, "ytick.labelsize": cfg.FONT_SIZE, "legend.fontsize": cfg.FONT_SIZE,
        "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "figure.dpi": 100, "figure.constrained_layout.use": True,
    })


def save_fig(fig, name):
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(cfg.FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(cfg.FIG_DIR, f"{name}.png"), dpi=cfg.PNG_DPI)
    plt.close(fig)
