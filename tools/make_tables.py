"""
Regenerate the LaTeX tables of Paper B (paper/ime/tables/) and the claim-type
mix table owned by Paper A (paper/eaj/tables/) from the run outputs, so that
no table number is hand-transcribed.

Sources
  paper/ime/tables/table_results_full.tex   results_v2/correction_comparison.csv,
                                            results_v2/chain_ladder_cohort.csv
  paper/ime/tables/table_alpha_z.tex        results_v2/calibration_factors.csv,
                                            results_v2/partial/predictions__<flagship>.csv
  paper/ime/tables/table_alpha_h.tex        results_v2/partial/calibration__<multi-step>.csv,
                                            results_v2/partial/predictions__<multi-step>.csv
  paper/ime/tables/table_k10.tex            results_v2/ and results_v2_k10/calibration_factors.csv
  paper/ime/tables/table_noay.tex           results_v2/ and results_noay/calibration_factors.csv,
                                            results_v2/ and results_noay/correction_comparison.csv
  paper/ime/tables/table_sim_scenarios.tex  results_sim/summary_by_arm.csv,
                                            results_sim_ms/summary_by_arm.csv
  paper/eaj/tables/table_mix_by_ay.tex      the claims data via src.data_prep

Run from the repo root with the project interpreter:
    python tools/make_tables.py            # write the tables, print a diff per file
    python tools/make_tables.py --check    # only print the diffs, write nothing
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FLAGSHIP = "random_forest__per_dy__log1p__two_stage"
MULTISTEP = "random_forest__multi_step__log1p__two_stage"
IME_TABLES = os.path.join(ROOT, "paper", "ime", "tables")
EAJ_TABLES = os.path.join(ROOT, "paper", "eaj", "tables")


# Deposited layout of the public repository, used when the run directory is absent.
DEPOSIT = {"results_v2": ("results",), "results_v2_k10": ("results", "k10"), "results_noay": ("results", "noay"),
           "results_sim": ("results", "sim"), "results_sim_ms": ("results", "sim_ms"),
           "results_sim_cov": ("results", "sim_cov"), "results_ms_cl": ("results", "ms_cl"),
           "results_v2_rec": ("results", "full25"), "results_v2_val9": ("results", "val9")}


def _p(*parts: str) -> str:
    p = os.path.join(ROOT, *parts)
    if not os.path.exists(p) and parts and parts[0] in DEPOSIT:
        q = os.path.join(ROOT, *DEPOSIT[parts[0]], *parts[1:])
        if os.path.exists(q):
            return q
    return p


def _int(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def _pct_int(x: float) -> str:
    if not np.isfinite(x) or abs(x) > 1e4:
        return r"$\!>\!10^4$"
    return f"{x:+.0f}"


def _horizon(d: pd.DataFrame) -> pd.Series:
    # calendar year of the cell minus the valuation year (I = 10)
    return d["accident_year"] + d["dev_lag"] - 1 - 10


def _realised(pred_csv: str, by: str) -> pd.Series:
    d = pd.read_csv(pred_csv, usecols=["accident_year", "dev_lag",
                                       "true_incremental_payment", "pred__none"])
    if by == "h":
        d["h"] = _horizon(d)
    g = d.groupby(by).agg(t=("true_incremental_payment", "sum"), p=("pred__none", "sum"))
    return g.t / g.p


# ------------------------------------------------------------------ results_full
OBJ_ORDER = [("mse_log1p", "RF log1p"), ("mse_raw", "RF raw"), ("poisson_rf", "RF Poisson"),
             ("poisson_hgb", "HGB Poisson"), ("gamma_hgb", "HGB Gamma"),
             ("poisson_lgbm", "LGBM Poisson"), ("tweedie_lgbm", "LGBM Tweedie")]
STRUCT = {"multi_step": "MS", "per_dy": "PD", "single_model": "SM"}
ZEROS = {"include_zeros": "IZ", "two_stage": "2S"}
ROW_ORDER = ["MS/2S", "PD/IZ", "PD/2S", "SM/IZ", "SM/2S"]
ARMS = ["none", "paper", "cv_global", "cv_dy_bs1", "cv_type_bs1", "iso_global"]


def table_results_full() -> str:
    cc = pd.read_csv(_p("results_v2", "correction_comparison.csv"))
    cc["row"] = cc.structure.map(STRUCT) + "/" + cc.zero_handling.map(ZEROS)
    cl = pd.read_csv(_p("results_v2", "chain_ladder_cohort.csv")).sum(numeric_only=True)
    cl_std = (cl.cl_standard / cl.true_total - 1) * 100
    cl_rbns = (cl.cl_rbns_cohort / cl.true_rbns - 1) * 100
    bt_path = _p("results_v2", "chain_ladder_cohort_by_type_min500.csv")
    cl_type = None
    if os.path.exists(bt_path):
        bt = pd.read_csv(bt_path); cl_type = (bt.cl_cohort_type.sum() / bt.true_rbns.sum() - 1) * 100

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\footnotesize",
        r"\caption{Total reserve error (\%) of every configuration and the main",
        r"correction arms on the MTPL lower triangle (true RBNS reserve 363.1\,M; the standard chain-ladder row is scored against the total truth of 438.6\,M). IZ:",
        r"zeros included; 2S: two-stage; SM: single model; PD: per development year;",
        r"MS: multi-step (one model per horizon, history frozen to match).",
        r"\emph{none}: uncorrected; \emph{paper}: the heuristic calibration of the",
        r"applied study \citep{JanousekPesta2026eaj}; \emph{cv}: cross-fitted ratio of",
        r"the model's own record---pooled for SM, per development year for PD (equal to the",
        r"unshrunk per-DY factor), per horizon for MS---every cross-fitted arm other than \emph{paper} needs at",
        r"least ten rows and ten positive out-of-fold cells in its group, which leaves DY~9 of the PD",
        r"models, with nine positive cells, uncorrected; \emph{cv-dy}, \emph{cv-type}:",
        r"B\"uhlmann--Straub toward 1 by DY (for MS: by horizon) / claim type---for PD",
        r"each development-year model has its own record, for SM one record is grouped;",
        r"\emph{iso}: isotonic. Raw-target configurations had no heuristic",
        r"calibration (\emph{paper}=\emph{none}). The last chain-ladder row stratifies the",
        r"report-cohort-consistent chain ladder by claim type (own factors for the three types",
        r"with at least 500 claims, all-type factors for the rest).}",
        r"\label{tab:results-full}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"objective & str./zeros & none & paper & cv & cv-dy & cv-type & iso \\",
        r"\midrule",
    ]
    first_group = True
    for obj, label in OBJ_ORDER:
        sub = cc[cc.objective == obj]
        if sub.empty:
            continue
        if not first_group:
            lines.append(r"\addlinespace")
        first_group = False
        rows = [r for r in ROW_ORDER if r in set(sub.row)]
        for i, row in enumerate(rows):
            cfg = sub[sub.row == row]
            vals = {}
            for arm in ARMS:
                v = cfg[cfg.arm == arm].total_reserve_error_pct
                if v.empty and arm == "paper":
                    v = cfg[cfg.arm == "none"].total_reserve_error_pct
                vals[arm] = _pct_int(float(v.iloc[0])) if not v.empty else "--"
            lead = label if i == 0 else ""
            lines.append(f"{lead} & {row} & " + " & ".join(vals[a] for a in ARMS) + r" \\")
    lines += [
        r"\addlinespace",
        f"CL standard (vs total truth) & -- & {_pct_int(cl_std)} & -- & -- & -- & -- & -- \\\\",
        f"CL RBNS-cohort (restricted triangle) & -- & {_pct_int(cl_rbns)} & -- & -- & -- & -- & -- \\\\",
        *([f"CL RBNS-cohort, by claim type & -- & {_pct_int(cl_type)} & -- & -- & -- & -- & -- \\\\"] if cl_type is not None else []),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ alpha_z
def _flagship_dy(path: str) -> pd.DataFrame:
    cf = pd.read_csv(path)
    f = cf[(cf.name == FLAGSHIP) & (cf.level == "dy")].copy()
    f["group"] = f.group.astype(int)
    return f.sort_values("group")


def table_alpha_z() -> str:
    f = _flagship_dy(_p("results_v2", "calibration_factors.csv"))
    realised = _realised(_p("results_v2", "partial", f"predictions__{FLAGSHIP}.csv"), "dev_lag")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Cross-fitted calibration diagnostics by development year for the",
        r"flagship configuration (per-DY, log-target, two-stage random forest), with $J=3$",
        r"folds. $\hat\alpha_k$ is the cross-fitted ratio (Thm~\ref{thm:asymp}), $\hat v_k$",
        r"its delta-method variance, $Z_k$ the Bühlmann--Straub weight toward $1$",
        r"(Thm~\ref{thm:bs}). The last column is the ratio realised on the lower triangle,",
        r"$\sum Y/\sum f$; the two are closest at DY~2, where the realised value already lies just outside the interval.",
        r"$n_{\mathrm{eff}}$ is the applied study's name for the count of cells with a positive payment, not an effective sample size.}",
        r"\label{tab:alpha-z}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"DY & $n$ & $n_{\mathrm{eff}}$ & $\hat\alpha_k$ & $\hat v_k$ & $Z_k$ & realised \\",
        r"\midrule",
    ]
    for _, r in f.iterrows():
        lines.append(f"{r.group} & {_int(r.n)} & {_int(r.n_eff)} & {r.alpha_cv:.2f} & "
                     f"{r.v_cv_delta:.3f} & {r.Z_bs1:.2f} & {realised.loc[r.group]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ alpha_h
def table_alpha_h() -> str:
    cf = pd.read_csv(_p("results_v2", "partial", f"calibration__{MULTISTEP}.csv"))
    f = cf[(cf.name == MULTISTEP) & (cf.level == "dy")].copy()
    f["group"] = f.group.astype(int)
    f = f.sort_values("group")
    realised = _realised(_p("results_v2", "partial", f"predictions__{MULTISTEP}.csv"), "h")
    cl_path = _p("results_ms_cl", "clustered_summary.csv")
    cl = pd.read_csv(cl_path) if os.path.exists(cl_path) else None
    if cl is not None:
        cl = cl[cl.h != "total"].copy(); cl["h"] = cl.h.astype(int); cl = cl.set_index("h")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{Per-horizon calibration diagnostics of the multi-step log-target",
        r"two-stage forest ($J=3$ folds): cross-fitted ratio $\hat\alpha^{(h)}$,",
        r"delta-method variance, B\"uhlmann--Straub weight toward 1 across horizons,",
        *([r"the same two quantities with the claim-clustered variance of \S\ref{sec:crossfit}",
           r"($\hat v^{(h)}_{\mathrm{cl}}$, $Z_h^{\mathrm{cl}}$; cells of one claim may be arbitrarily correlated),"] if cl is not None else []),
        r"and the ratio realised on the horizon-$h$ diagonal of the lower triangle.",
        r"The horizon-one correction agrees with the realised ratio to within $0.02$; the divergence at",
        r"$h\ge4$ is consistent with the recent-cohort mix shift documented in the applied study",
        r"\citep{JanousekPesta2026eaj}. Horizons 8 and 9 have no calibrated factor (fewer than ten positive out-of-fold cells, the eligibility rule stated in the caption of Table~\ref{tab:results-full}) and are omitted.}",
        r"\label{tab:alpha-h}",
        (r"\begin{tabular}{rrrrrrrrr}" if cl is not None else r"\begin{tabular}{rrrrrrr}"),
        r"\toprule",
        (r"$h$ & $n$ & $n_{\mathrm{eff}}$ & $\hat\alpha^{(h)}$ & $\hat v^{(h)}$ & $Z_h$ & $\hat v^{(h)}_{\mathrm{cl}}$ & $Z_h^{\mathrm{cl}}$ & realised \\"
         if cl is not None else
         r"$h$ & $n$ & $n_{\mathrm{eff}}$ & $\hat\alpha^{(h)}$ & $\hat v^{(h)}$ & $Z_h$ & realised \\"),
        r"\midrule",
    ]
    for _, r in f.iterrows():
        if r.group not in realised.index:
            continue
        extra = ""
        if cl is not None:
            extra = (f"{cl.loc[r.group, 'v_delta_cl']:.3f} & {cl.loc[r.group, 'Z_cl']:.2f} & "
                     if r.group in cl.index else "-- & -- & ")
        lines.append(f"{r.group} & {_int(r.n)} & {_int(r.n_eff)} & {r.alpha_cv:.2f} & "
                     f"{r.v_cv_delta:.3f} & {r.Z_bs1:.2f} & {extra}{realised.loc[r.group]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ k10
def table_k10() -> str:
    k3 = _flagship_dy(_p("results_v2", "calibration_factors.csv")).set_index("group")
    k10 = _flagship_dy(_p("results_v2_k10", "calibration_factors.csv")).set_index("group")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{Fold-count check for the flagship configuration: cross-fitted per-DY",
        r"ratio and its two variance estimates at $J=3$ and with up to $J=10$ folds, capped at the",
        r"number of accident years with observed cells (9, 8, 7, 6, 5, 4 and 3 at DY~2--8). The delta-method",
        r"variance is stable as the fold count changes; the point estimate moves by at most $0.44$ (DY~6);",
        r"the between-fold variance is not stable and is therefore not used. At DY~8 only",
        r"three accident years have cells, so both fold counts reduce to the same three",
        r"accident-year groups and the columns coincide.}",
        r"\label{tab:k10}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{$J=3$} & \multicolumn{3}{c}{$J\le10$} \\",
        r"DY & $\hat\alpha_k$ & $\hat v_k^{\mathrm{delta}}$ & $\hat v_k^{\mathrm{fold}}$ & $\hat\alpha_k$ & $\hat v_k^{\mathrm{delta}}$ & $\hat v_k^{\mathrm{fold}}$ \\",
        r"\midrule",
    ]
    for g in k3.index:
        if g not in k10.index:  # a development year without an estimate in one of the runs
            continue
        a, b = k3.loc[g], k10.loc[g]
        lines.append(f"{g} & {a.alpha_cv:.2f} & {a.v_cv_delta:.3f} & {a.v_cv_fold:.3f} & "
                     f"{b.alpha_cv:.2f} & {b.v_cv_delta:.3f} & {b.v_cv_fold:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ reduced features
NOAY_ARMS = [("none", "none"), ("paper", "paper"), ("cv_global", "cv"), ("cv_dy_bs1", "cv-dy"),
             ("cv_type_bs1", "cv-type"), ("iso_global", "iso")]


def table_noay() -> str:
    """Flagship configuration with and without the two accident-year features (RESERVING_EXCLUDE_AY_FEATURES=1)."""
    full = _flagship_dy(_p("results_v2", "calibration_factors.csv")).set_index("group")
    red = _flagship_dy(_p("results_noay", "calibration_factors.csv")).set_index("group")
    cc_full = pd.read_csv(_p("results_v2", "correction_comparison.csv"))
    cc_red = pd.read_csv(_p("results_noay", "correction_comparison.csv"))
    ctrl_note = []
    if os.path.exists(_p("results_v2_rec", "calibration_factors.csv")):  # same-budget full-feature control
        ctrl = _flagship_dy(_p("results_v2_rec", "calibration_factors.csv")).set_index("group")
        cc_ctrl = pd.read_csv(_p("results_v2_rec", "correction_comparison.csv"))
        d_alpha = (ctrl.alpha_cv - full.alpha_cv.reindex(ctrl.index)).abs().max()
        def _tot(cc, a):
            return float(cc[(cc.name == FLAGSHIP) & (cc.arm == a)].total_reserve_error_pct.iloc[0])
        d_tot = max(abs(_tot(cc_ctrl, a) - _tot(cc_full, a)) for a in ("none", "paper", "cv_dy_bs1"))
        ctrl_note = [r"Both refits use 25 command-line Optuna trials against the deployed run's 50, i.e.\ 12 against 25 per",
                     r"development-year model, which changes both the classifier's and the regressor's tuned parameters",
                     r"(\S\ref{sec:data}); a full-feature refit at 25",
                     f"trials moves the ratios by at most ${d_alpha:.2f}$ and the none/paper/cv-dy totals by at most ${d_tot:.1f}$ points,",
                     r"a larger effect on the ratios than removing the coordinates has."]

    def arm(cc, a):
        v = cc[(cc.name == FLAGSHIP) & (cc.arm == a)].total_reserve_error_pct
        return f"{float(v.iloc[0]):+.1f}" if not v.empty else "--"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{Reduced-feature check for the flagship configuration (\S\ref{sec:features} of",
        r"): the forest refitted without the three accident-year coordinates",
        r"(accident year, occurrence period and occurrence month, which coincide in these annual data),",
        r"so that it is a function of the reduced covariate $\check x$ by construction, against the",
        r"deployed forest. Upper block: cross-fitted",
        r"per-DY ratio, delta-method variance and B\"uhlmann--Straub weight ($J=3$ folds).",
        r"Lower block: total reserve error (\%) on the MTPL lower triangle by correction arm",
        r"(arm labels as in Table~\ref{tab:results-full}).",
        *ctrl_note,
        r"}",
        r"\label{tab:noay}",
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{full features (deployed)} & \multicolumn{3}{c}{without accident-year coordinates} \\",
        r"DY & $\hat\alpha_k$ & $\hat v_k^{\mathrm{delta}}$ & $Z_k$ & $\hat\alpha_k$ & $\hat v_k^{\mathrm{delta}}$ & $Z_k$ \\",
        r"\midrule",
    ]
    for g in full.index:
        if g not in red.index:
            continue
        a, b = full.loc[g], red.loc[g]
        lines.append(f"{g} & {a.alpha_cv:.2f} & {a.v_cv_delta:.3f} & {a.Z_bs1:.2f} & "
                     f"{b.alpha_cv:.2f} & {b.v_cv_delta:.3f} & {b.Z_bs1:.2f} \\\\")
    lines += [
        r"\midrule",
        r"arm & " + " & ".join(lab for _, lab in NOAY_ARMS) + r" \\",
        r"\midrule",
        "full features & " + " & ".join(arm(cc_full, a) for a, _ in NOAY_ARMS) + r" \\",
        "without accident-year coordinates & " + " & ".join(arm(cc_red, a) for a, _ in NOAY_ARMS) + r" \\",
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ coverage
def table_coverage() -> str:
    """Coverage of the delta-method interval for the flagship per-DY ratio against the pipeline's own targets
    (tools/coverage_sim.py on results_sim_cov/); empty string when the experiment has not been run."""
    path = _p("results_sim_cov", "coverage_summary.csv")
    if not os.path.exists(path):
        return ""
    cs = pd.read_csv(path)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{Coverage experiment for the $95\%$ delta-method interval of the cross-fitted per-DY ratio",
        r"(flagship configuration, $J=3$ folds), over $30$ simulated portfolios (three scenarios $\times$ ten seeds).",
        r"Population target: the balance factor $\E_P[Y]/\E_P[\hat f(X)]$ of the fitted per-DY predictor under the",
        r"training law, evaluated on an independent replicate portfolio with the realised payments replaced by the",
        r"generator's conditional means. Estimation-record reference: $\sum_i m_i/\sum_i\hat f^{(-j(i))}(X_i)$ with $m_i$ the",
        r"generator's conditional mean, which differs from the cross-fitted target $\alpha_n$ of Theorem~\ref{thm:asymp}(iv).",
        r"Bias is the mean of $\hat\alpha_k/\alpha_k^{\mathrm{pop}}-1$; SE/s.d.\ divides the mean plug-in standard error",
        r"$\hat v_k^{1/2}$ by the empirical standard deviation of $\hat\alpha_k-\alpha_k^{\mathrm{pop}}$ across replicates",
        r"(1 = the plug-in variance matches the sampling spread); the last column counts the misses below/above the",
        r"population target. The 240 intervals come from 30 portfolios (ten seeds per scenario; the three scenarios reuse the seeds).}",
        r"\label{tab:coverage}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"DY & replicates & cov.\ (population) & cov.\ (record) & $\bar{\hat\alpha}_k$ & bias (\%) & SE/s.d. & misses $-/+$ \\",
        r"\midrule",
    ]
    for _, r in cs.iterrows():
        if str(r.dy) == "all":
            lines.append(r"\midrule")
            lines.append(f"all & {int(r.replicates)} & {r.coverage_pop:.2f} & {r.coverage_cond:.2f} & -- & {r.mean_rel_bias_pop_pct:+.1f} & -- & {int(r.misses_below)}/{int(r.misses_above)} \\\\")
        else:
            lines.append(f"{int(float(r.dy))} & {int(r.replicates)} & {r.coverage_pop:.2f} & {r.coverage_cond:.2f} & "
                         f"{r.mean_alpha_hat:.2f} & {r.mean_rel_bias_pop_pct:+.1f} & {r.se_over_sd:.2f} & {int(r.misses_below)}/{int(r.misses_above)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ fixed-predictor coverage
def table_fixed_coverage() -> str:
    """Coverage of the delta-method interval for a FIXED predictor (Theorem 4.6(i)), tools/fixed_predictor_coverage.py."""
    path = _p("results_sim_cov", "fixed_predictor_coverage_summary.csv")
    if not os.path.exists(path):
        return ""
    cs = pd.read_csv(path)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\caption{Fixed-predictor check of Theorem~\ref{thm:asymp}(i): the predictor $f(c,k,r)=e^{\mu_c+\beta_c(k-r)}-1$ (the",
        r"generator's type-level severity curve, no fitting) is scored on the observed cells of each simulated portfolio;",
        r"$\hat\alpha_k=\sum Y/\sum f$ with the delta-method variance, nominal $95\%$ interval. The population factor",
        r"$\alpha_{0,k}=\E_P[Y]/\E_P[f]$ is estimated on the ten pooled fresh replicate portfolios of the scenario with",
        r"the generator's conditional means; the last column is its Monte Carlo standard error relative to the interval's.",
        r"SE/s.d.\ divides the mean plug-in standard error by the empirical spread of $\hat\alpha_k-\alpha_{0,k}$ across the",
        r"$30$ portfolios (1 = the variance matches).}",
        r"\label{tab:fixedcov}",
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"DY & portfolios & coverage & $\bar{\hat\alpha}_k$ & $\bar\alpha_{0,k}$ & SE/s.d. & target SE / interval SE \\",
        r"\midrule",
    ]
    for _, r in cs.iterrows():
        if str(r.dy) == "all":
            lines.append(r"\midrule")
            lines.append(f"all & {int(r.replicates)} & {r.coverage:.2f} & -- & -- & -- & -- \\\\")
        else:
            lines.append(f"{int(float(r.dy))} & {int(r.replicates)} & {r.coverage:.2f} & {r.mean_alpha_hat:.3f} & {r.mean_alpha_0:.3f} & "
                         f"{r.se_over_sd:.2f} & {r.target_se_over_interval_se:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ simulation
SIM_ROWS = [
    ("log-target forest, per DY", "results_sim", "random_forest__per_dy__log1p__two_stage", "cv_global"),
    ("log-target forest, single model", "results_sim", "random_forest__single_model__log1p__two_stage", "cv_global"),
    ("Poisson forest, per DY", "results_sim", "random_forest_poisson__per_dy__raw__two_stage", "cv_global"),
    ("Poisson forest, single model", "results_sim", "random_forest_poisson__single_model__raw__two_stage", "cv_global"),
    ("log-target, multi-step", "results_sim_ms", MULTISTEP, "cv_dy"),
    ("Poisson, multi-step", "results_sim_ms", "random_forest_poisson__multi_step__raw__two_stage", "cv_dy"),
]
SCENARIOS = ["S-A", "S-B", "S-C"]


def _msd(s: pd.DataFrame, scen: str, name: str, arm: str) -> str:
    r = s[(s.scenario == scen) & (s.name == name) & (s.arm == arm)]
    if r.empty:
        return "--"
    return f"{r.mean_error_pct.iloc[0]:+.1f} ({r.sd_error_pct.iloc[0]:.1f})"


def table_sim_scenarios() -> str:
    S = {d: pd.read_csv(_p(d, "summary_by_arm.csv")) for d in ("results_sim", "results_sim_ms")}
    flag = S["results_sim"]
    n_seeds = {sc: int(flag[(flag.scenario == sc) & (flag.name == SIM_ROWS[0][2]) & (flag.arm == "none")].n_seeds.iloc[0])
               for sc in SCENARIOS}
    words = {10: "ten", 20: "twenty", 30: "thirty"}
    if len(set(n_seeds.values())) == 1:
        seeds_txt = f"{words.get(n_seeds['S-A'], n_seeds['S-A'])} seeds per scenario"
    else:
        seeds_txt = (f"{words.get(n_seeds['S-A'], n_seeds['S-A'])} seeds in S-A and "
                     f"{words.get(n_seeds['S-B'], n_seeds['S-B'])} in S-B and S-C")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\footnotesize",
        r"\caption{Simulation study: total reserve error (\%), mean (s.d.) over " + seeds_txt + ".",
        r"``calibrated\textquotedblright{} is the model's own cross-fitted ratio (per",
        r"development year for the per-DY structure, pooled for the single model) and the per-horizon",
        r"ratio for the multi-step models.",
        r"Chain-ladder rows come from the same simulated portfolios as the multi-step models: the report-cohort-consistent chain ladder against the RBNS truth and the standard chain ladder against the total truth (\S\ref{sec:results}).",
        r"Errors are relative to the realised lower-triangle payments of each portfolio; the deposited files also carry the error against the latent-information oracle.}",
        r"\label{tab:sim-scenarios}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{S-A stable} & \multicolumn{2}{c}{S-B mix} & \multicolumn{2}{c}{S-C infl.} \\",
        r"configuration & none & calib. & none & calib. & none & calib. \\",
        r"\midrule",
    ]
    for label, d, name, calib in SIM_ROWS:
        cells = []
        for sc in SCENARIOS:
            cells += [_msd(S[d], sc, name, "none"), _msd(S[d], sc, name, calib)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    ms = S["results_sim_ms"]
    for label, name in [("chain ladder, RBNS-cohort", "chain_ladder_rbns_cohort"),
                        ("chain ladder, standard (vs total)", "chain_ladder_total")]:
        cells = []
        for sc in SCENARIOS:
            r = ms[(ms.scenario == sc) & (ms.name == name)]
            cells.append(r"\multicolumn{2}{c}{" + f"{r.mean_error_pct.iloc[0]:+.1f} ({r.sd_error_pct.iloc[0]:.1f})" + "}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ mix by AY (Paper A)
MAIN_TYPES = ["Material", "Technical", "Bodily", "Annuity"]


def table_mix_by_ay() -> str:
    from src import data_prep  # noqa: WPS433 (needs the data directory)

    claims, transactions = data_prep.load_raw_data()
    claims_rbns, _ = data_prep.filter_rbns_observed(claims, transactions)
    ct = claims_rbns["claim_type"].astype(str).str.strip().str.capitalize()
    ct = ct.where(ct.isin(MAIN_TYPES), "other")
    tab = pd.crosstab(claims_rbns["accident_year"], ct)
    for c in MAIN_TYPES + ["other"]:
        if c not in tab:
            tab[c] = 0
    n = tab.sum(axis=1)
    pct = tab.div(n, axis=0) * 100
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{RBNS claim-type mix by accident year (\% of the claims in the",
        r"modelling set). The long-tailed share (Annuity + Bodily) falls from about",
        r"15\% in the oldest cohort to 7\% in the most recent one, so recent accident",
        r"years appear to develop more lightly than the history they are compared",
        r"with.}",
        r"\label{tab:mix-ay}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"AY & claims & Material & Technical & Bodily & Annuity & other & long-tailed \\",
        r"\midrule",
    ]
    for ay in sorted(tab.index):
        row = pct.loc[ay]
        lt = row["Annuity"] + row["Bodily"]
        lines.append(f"{int(ay)} & {_int(n.loc[ay])} & " +
                     " & ".join(f"{row[c]:.1f}" for c in MAIN_TYPES + ["other"]) +
                     f" & {lt:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _needs_private_records(fn):
    """Tables 1 and 2 read the per-cell out-of-fold record and predictions, which are not deposited
    (confidential claim-level data); in the public repository they are skipped with a message."""
    def wrapped():
        try:
            return fn()
        except FileNotFoundError as e:
            print(f"  ({fn.__name__}: needs the confidential per-cell records, not deposited; skipped: {os.path.basename(str(e).split(chr(39))[-2]) if chr(39) in str(e) else e})")
            return None
    wrapped.__name__ = fn.__name__
    return wrapped


table_alpha_z = _needs_private_records(table_alpha_z)
table_alpha_h = _needs_private_records(table_alpha_h)


# ------------------------------------------------------------------ transfer term by importance weighting
def table_transfer_iw() -> str:
    """Importance-weighted estimate of the transfer term (tools/transfer_iw.py on results_v2/); empty when not run."""
    path = _p("results_v2", "transfer_iw.csv")
    if not os.path.exists(path):
        return ""
    t = pd.read_csv(path)
    h1 = t[t.target == "h1"].set_index("dev_lag"); lo = t[t.target == "lower"].set_index("dev_lag")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{Importance-weighted estimate of the target factor $\alpha_Q$ for the flagship per-DY forest (\S\ref{sec:factors};",
        r"the difference from $\hat\alpha_P$ estimates the transfer term for a common positive predictor).",
        r"$\hat\alpha_P$ is the cross-fitted factor; $\hat\alpha_Q^{\mathrm{IW}}$ re-weights the same record by the",
        r"density ratio of the reduced covariates (no accident-year coordinates) between training cells and target",
        r"cells, estimated by a cross-fitted classifier; ``realised'' is the ratio on the target cells (delta-method",
        r"standard error of the realised ratio alone in parentheses; the uncertainty of the weighted factor is not included).",
        r"Left block: target = the horizon-one diagonal, the setting of Theorem~\ref{thm:transfer} (the deployed-model comparison is",
        r"diagnostic for the reasons given in \S\ref{sec:factors}); right block: target = the whole lower triangle of the development",
        r"year, which includes $h\ge2$ cells for which Proposition~\ref{prop:horizon} allows an additional feature-construction mismatch.",
        r"AUC is the classifier's cross-fitted discrimination ($0.5$ = chance level); ESS the effective sample size of the",
        r"weights as a share of the training cells. The deposited file also carries the number of positive training cells per",
        r"development year (nine at DY~9).}",
        r"\label{tab:transfer-iw}",
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{rr rrrrr rrrrr}",
        r"\toprule",
        r"& & \multicolumn{5}{c}{target: horizon one} & \multicolumn{5}{c}{target: lower triangle} \\",
        r"DY & $\hat\alpha_P$ & $\hat\alpha_Q^{\mathrm{IW}}$ & realised & AUC & ESS & $n_Q$ & $\hat\alpha_Q^{\mathrm{IW}}$ & realised & AUC & ESS & $n_Q$ \\",
        r"\midrule",
    ]
    for k in sorted(set(h1.index) | set(lo.index)):
        a = h1.loc[k] if k in h1.index else None; b = lo.loc[k] if k in lo.index else None
        aP = (a if a is not None else b).alpha_P
        def blk(r):
            if r is None:
                return "-- & -- & -- & -- & --"
            return f"{r.alpha_Q_iw:.2f} & {r.realised_ratio:.2f} ({r.realised_se:.2f}) & {r.auc:.2f} & {r.ess_share:.2f} & {_int(r.n_Q)}"
        lines.append(f"{k} & {aP:.2f} & {blk(a)} & {blk(b)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ back-test at valuation year 9
def table_backtest() -> str:
    """Back-test of the flagship calibration at valuation year 9 (tools/backtest_val9.py); empty when not run."""
    path = _p("results_v2_val9", "backtest_summary.csv"); tpath = _p("results_v2_val9", "backtest_totals.csv")
    if not (os.path.exists(path) and os.path.exists(tpath)):
        return ""
    t = pd.read_csv(path).set_index("dev_lag"); tot = pd.read_csv(tpath)
    sh_lo = tot[(tot.arm == "none") & (tot.scope == "lower")].unfitted_share_pct.iloc[0]
    sh_h1 = tot[(tot.arm == "none") & (tot.scope == "h1")].unfitted_share_pct.iloc[0]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{Back-test of the flagship calibration at valuation year~9 (\S\ref{sec:factors}). Left: per-DY cross-fitted",
        r"factor estimated on the $9\times9$ upper triangle with its $95\%$ delta interval, the ratio realised on the",
        r"calendar-year-10 diagonal (horizon one, fully observed) and on the whole lower triangle of the year-9",
        r"valuation. Right: the corresponding factor and realised ratios at valuation year~10 (the factors and lower-triangle",
        r"ratios also appear in Table~\ref{tab:alpha-z}). Bottom: lower-triangle",
        r"and horizon-one totals of the arms at year~9, over the cells with a fitted development-year model (DY$\le9$: DY~10 has",
        rf"no training cells at year~9 and holds {sh_lo:.1f}\% of the realised lower-triangle payments and {sh_h1:.2f}\% of the diagonal).",
        r"The cohort is the claims reported by year~9 (settled ones included, later-reported ones excluded), the features use payments",
        r"through year~9 and the outcomes come from the full transaction file. A dash in a factor column marks a development year",
        r"below the ten-positive-cell threshold; realised ratios appear wherever a development-year model is fitted. The chain-ladder",
        r"triangles of this run keep the year-10 RBNS cohort.}",
        r"\label{tab:backtest}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
        r"\begin{tabular}{r rrrrr c rrr}",
        r"\toprule",
        r"& \multicolumn{5}{c}{valuation year 9} & & \multicolumn{3}{c}{valuation year 10} \\",
        r"DY & $\hat\alpha_k$ & $95\%$ interval & realised $h=1$ & realised lower & $n_{h=1}$ & & $\hat\alpha_k$ & realised $h=1$ & realised lower \\",
        r"\midrule",
    ]
    for k in t.index:
        r = t.loc[k]
        nan = float("nan")
        fac9 = f"{r.alpha9:.2f} & [{r.alpha9-1.96*r.se9:.2f}, {r.alpha9+1.96*r.se9:.2f}]" if pd.notna(r.get("alpha9", nan)) else "-- & --"
        out9 = (f"{r.realised9_h1:.2f} & {r.realised9_lower:.2f} & {_int(r.n9_h1)}" if pd.notna(r.get("realised9_h1", nan)) else "-- & -- & --")
        left = fac9 + " & " + out9
        fac10 = f"{r.alpha10:.2f}" if pd.notna(r.get("alpha10", nan)) else "--"
        out10 = f"{r.realised10_h1:.2f} & {r.realised10_lower:.2f}" if pd.notna(r.get("realised10_h1", nan)) else "-- & --"
        right = fac10 + " & " + out10
        lines.append(f"{k} & {left} & & {right} \\\\")
    lines += [r"\midrule", r"\multicolumn{10}{l}{Arm totals at valuation year 9 (error \% on the lower triangle / on the horizon-one diagonal):} \\"]
    names = {"none": "uncorrected", "paper": "heuristic", "cv_dy": "unshrunk per-DY", "cv_dy_bs1": "B\\\"uhlmann--Straub per-DY"}
    for arm, lab in names.items():
        lo = tot[(tot.arm == arm) & (tot.scope == "lower")].err_pct.iloc[0]; h1 = tot[(tot.arm == arm) & (tot.scope == "h1")].err_pct.iloc[0]
        lines.append(rf"\multicolumn{{10}}{{l}}{{\quad {lab}: ${lo:+.1f}\%$ / ${h1:+.1f}\%$}} \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ driver
TABLES = {
    os.path.join(IME_TABLES, "table_results_full.tex"): table_results_full,
    os.path.join(IME_TABLES, "table_alpha_z.tex"): table_alpha_z,
    os.path.join(IME_TABLES, "table_alpha_h.tex"): table_alpha_h,
    os.path.join(IME_TABLES, "table_k10.tex"): table_k10,
    os.path.join(IME_TABLES, "table_noay.tex"): table_noay,
    os.path.join(IME_TABLES, "table_coverage.tex"): table_coverage,
    os.path.join(IME_TABLES, "table_fixed_coverage.tex"): table_fixed_coverage,
    os.path.join(IME_TABLES, "table_sim_scenarios.tex"): table_sim_scenarios,
    os.path.join(IME_TABLES, "table_transfer_iw.tex"): table_transfer_iw,
    os.path.join(IME_TABLES, "table_backtest.tex"): table_backtest,
    os.path.join(EAJ_TABLES, "table_mix_by_ay.tex"): table_mix_by_ay,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print diffs only, write nothing")
    ap.add_argument("--only", nargs="*", help="basenames to (re)generate, e.g. table_k10.tex")
    args = ap.parse_args()
    changed = 0
    for path, fn in TABLES.items():
        base = os.path.basename(path)
        if args.only and base not in args.only:
            continue
        new = fn()
        if not new:  # skipped (inputs not deposited) or nothing to write
            continue
        old = open(path).read() if os.path.exists(path) else ""
        if new == old:
            print(f"{base}: unchanged")
            continue
        changed += 1
        print(f"{base}: DIFFERS" + ("" if args.check else " -> rewritten"))
        for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), "committed", "regenerated", lineterm="", n=0):
            if ln.startswith(("---", "+++", "@@")):
                continue
            print("   " + ln)
        if not args.check:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(new)
    print(f"{changed} table(s) differ")


if __name__ == "__main__":
    main()
