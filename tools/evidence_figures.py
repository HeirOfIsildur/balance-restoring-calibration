"""
Publication figures of the IME paper (Figures 1-3), from the run directories.

Reads the run outputs (results_v2/partial, results_sim) and writes three
vector PDFs (plus PNG previews) to --out (default results/figures/):

  fig2_horizon_collapse.pdf   reserve error vs horizon h, real data, several
                              losses — the paper's central empirical claim
                              (calibration fixes h=1; every loss collapses at
                              large h)
  fig3_perdy_invariance.pdf   upper-triangle cross-fitted ratio (with 95%
                              delta CI) vs the ratio realised on the lower
                              triangle, flagship — the illustration of the
                              invariance condition (Remark 3.11)
  fig1_sim_horizon.pdf        reserve error vs horizon in the simulation
                              (S-A, mean over seeds, seed band) — the same
                              pattern in a known-truth world

Run from the repo root with the project interpreter:
    python tools/evidence_figures.py                      # -> results/figures/
    python tools/evidence_figures.py --out paper/ime/figures --pdf-only   # Paper B copies
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# Okabe–Ito colourblind-safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#666666"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figures")
EXTS = ("pdf", "png")


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in EXTS:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}." + " / .".join(EXTS))


def _real(config):
    d = pd.read_csv(os.path.join(ROOT, "results_v2", "partial", f"predictions__{config}.csv"))
    d["h"] = d.accident_year + d.dev_lag - 1 - 10
    return d


def horizon_err(d, col):
    g = d.groupby("h").agg(true=("true_incremental_payment", "sum"), pred=(col, "sum"))
    return g.index.values, (g.pred / g.true - 1) * 100, g.true.values


# ---------------------------------------------------------------- Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    flagship = _real("random_forest__per_dy__log1p__two_stage")
    h, _, true = horizon_err(flagship, "pred__none")
    ax2 = ax.twinx()
    ax2.bar(h, true / true.sum() * 100, width=0.7, color="0.88", zorder=0)
    ax2.set_ylabel("share of true reserve (%)", color="0.55")
    ax2.tick_params(axis="y", colors="0.55")
    ax2.set_ylim(0, 45)
    ax2.spines["top"].set_visible(False)

    series = [
        (flagship, "pred__none", C["blue"], "-", "o", "log1p forest, uncorrected"),
        (flagship, "pred__cv_global", C["blue"], "--", "o", "log1p forest, calibrated"),
        (_real("random_forest_poisson__per_dy__raw__two_stage"), "pred__none", C["green"], "-", "s", "Poisson forest"),
        (_real("lightgbm_tweedie__per_dy__raw__two_stage"), "pred__none", C["orange"], "-", "^", "LightGBM Tweedie"),
        (_real("random_forest__multi_step__log1p__two_stage"), "pred__cv_dy", C["red"], "--", "D", "multi-step log1p, per-$h$ calibrated"),
    ]
    for d, col, c, ls, mk, lab in series:
        hh, err, _ = horizon_err(d, col)
        ax.plot(hh, err, ls, color=c, marker=mk, ms=3.5, lw=1.3, label=lab, zorder=3)
    ax.axhline(0, color="0.3", lw=0.8, zorder=1)
    ax.set_xlabel("horizon $h$ (calendar year $-$ 10)")
    ax.set_ylabel("reserve error (%)")
    ax.set_xticks(range(1, 10))
    ax.set_ylim(-110, 240)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.set_title("Reserve error by horizon (MTPL data)")
    _save(fig, "fig2_horizon_collapse")


# ---------------------------------------------------------------- Figure 2
def fig2():
    cf = pd.read_csv(os.path.join(ROOT, "results_v2", "partial",
                                  "calibration__random_forest__per_dy__log1p__two_stage.csv"))
    dy = cf[cf.level == "dy"].copy()
    dy["group"] = dy.group.astype(int)
    dy = dy.sort_values("group")
    d = _real("random_forest__per_dy__log1p__two_stage")
    realised = d.groupby("dev_lag").apply(
        lambda x: x.true_incremental_payment.sum() / x.pred__none.sum(), include_groups=False)
    x = dy.group.values
    ahat = dy.alpha_cv.values
    se = np.sqrt(dy.v_cv_delta.clip(lower=0).values)
    rr = realised.reindex(x).values

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(x, ahat, yerr=1.96 * se, fmt="o", color=C["blue"], ms=4, lw=1.2,
                capsize=2.5, label=r"upper-triangle $\hat\alpha_k$ (95% CI)", zorder=3)
    ax.plot(x, rr, "s--", color=C["red"], ms=4, lw=1.2,
            label=r"realised lower-triangle ratio", zorder=3)
    ax.axhline(1, color="0.4", lw=0.8, ls=":")
    ax.axhspan(0.5, 5, color=C["green"], alpha=0.06, zorder=0)
    ax.text(x.max(), 5.15, "clip bound 5", color=C["green"], fontsize=6.5, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("development year")
    ax.set_ylabel("calibration ratio")
    ax.set_title("Estimated against realised ratio by development year (flagship, log1p forest)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xticks(x)
    _save(fig, "fig3_perdy_invariance")


# ---------------------------------------------------------------- Figure 3
def fig3():
    cfgs = [("results_sim", "random_forest__per_dy__log1p__two_stage", "pred__cv_global", C["blue"], "--", "log1p per-DY, calibrated"),
            ("results_sim", "random_forest__per_dy__log1p__two_stage", "pred__none", C["blue"], "-", "log1p per-DY, uncorrected"),
            ("results_sim", "random_forest_poisson__per_dy__raw__two_stage", "pred__none", C["green"], "-", "Poisson per-DY, uncorrected"),
            ("results_sim_ms", "random_forest__multi_step__log1p__two_stage", "pred__cv_dy", C["red"], "--", "multi-step log1p, per-$h$ calibrated")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for root_dir, cfg, arm, c, ls, lab in cfgs:
        if True:
            per_seed = []
            for sd in sorted(glob.glob(os.path.join(ROOT, root_dir, "S-A", "seed*"))):
                f = os.path.join(sd, "partial", f"predictions__{cfg}.csv")
                if not os.path.exists(f):
                    continue
                d = pd.read_csv(f, usecols=["accident_year", "dev_lag", "true_incremental_payment", arm])
                d["h"] = d.accident_year + d.dev_lag - 1 - 10
                g = d.groupby("h").agg(t=("true_incremental_payment", "sum"), p=(arm, "sum"))
                per_seed.append((g.p / g.t - 1) * 100)
            if not per_seed:
                continue
            M = pd.concat(per_seed, axis=1)
            mean, lo, hi = M.mean(axis=1), M.min(axis=1), M.max(axis=1)
            if ls == "--":
                ax.fill_between(mean.index, lo, hi, color=c, alpha=0.10, zorder=1)
            ax.plot(mean.index, mean.values, ls, color=c, marker="o", ms=3, lw=1.2,
                    label=lab, zorder=3)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("reserve error (%)")
    ax.set_xticks(range(1, 10))
    ax.set_title("Simulation S-A (known truth): per-horizon calibrated multi-step model within about 20% through $h=7$")
    ax.legend(loc="lower left", frameon=False)
    _save(fig, "fig1_sim_horizon")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evidence figures (fig1-3) for Paper B.")
    ap.add_argument("--out", default=OUT, help="output directory (default: results/figures)")
    ap.add_argument("--pdf-only", action="store_true", help="write only the PDFs (no PNG previews)")
    args = ap.parse_args()
    OUT = os.path.abspath(args.out)
    if args.pdf_only:
        EXTS = ("pdf",)
    print(f"Writing figures to {OUT}")
    fig1()
    fig2()
    fig3()
