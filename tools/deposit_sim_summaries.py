#!/usr/bin/env python3
"""Deposit the per-horizon summaries behind Paper B's Figures 1 and 3 as CSVs (referee request).

Reuses the computations of tools/evidence_figures.py:
  results_v2/horizon_errors_real.csv          real data: per horizon h, share of true reserve and error % of the
                                              five Figure-1 series (uncorrected/calibrated log1p forest, Poisson
                                              forest, LightGBM Tweedie, per-h calibrated multi-step)
  results_sim_ms/horizon_errors_sim.csv       simulation: per scenario, arm and h, mean/min/max error % over seeds
                                              for the four Figure-3 series (S-A; S-B/S-C where present)
Usage: python tools/deposit_sim_summaries.py
"""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence_figures as ef

ROOT = ef.ROOT

def real():
    series = [("random_forest__per_dy__log1p__two_stage", "pred__none", "log1p forest, uncorrected"),
              ("random_forest__per_dy__log1p__two_stage", "pred__cv_global", "log1p forest, calibrated"),
              ("random_forest_poisson__per_dy__raw__two_stage", "pred__none", "Poisson forest"),
              ("lightgbm_tweedie__per_dy__raw__two_stage", "pred__none", "LightGBM Tweedie"),
              ("random_forest__multi_step__log1p__two_stage", "pred__cv_dy", "multi-step log1p, per-h calibrated")]
    rows = []
    for cfg, col, lab in series:
        d = ef._real(cfg); h, err, true = ef.horizon_err(d, col)
        for hh, e, t in zip(h, err, true):
            rows.append(dict(series=lab, config=cfg, arm=col, h=int(hh), true_M=t / 1e6, share_pct=100 * t / true.sum(), err_pct=e))
    out = os.path.join(ROOT, "results_v2", "horizon_errors_real.csv"); pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out)

def sim():
    cfgs = [("results_sim", "random_forest__per_dy__log1p__two_stage", "pred__cv_global", "log1p per-DY, calibrated"),
            ("results_sim", "random_forest__per_dy__log1p__two_stage", "pred__none", "log1p per-DY, uncorrected"),
            ("results_sim", "random_forest_poisson__per_dy__raw__two_stage", "pred__none", "Poisson per-DY, uncorrected"),
            ("results_sim_ms", "random_forest__multi_step__log1p__two_stage", "pred__cv_dy", "multi-step log1p, per-h calibrated")]
    rows = []
    for scen in ("S-A", "S-B", "S-C"):
        for root_dir, cfg, arm, lab in cfgs:
            per_seed = []
            for sd in sorted(glob.glob(os.path.join(ROOT, root_dir, scen, "seed*"))):
                f = os.path.join(sd, "partial", f"predictions__{cfg}.csv")
                if not os.path.exists(f): continue
                d = pd.read_csv(f, usecols=["accident_year", "dev_lag", "true_incremental_payment", arm])
                d["h"] = d.accident_year + d.dev_lag - 1 - 10
                g = d.groupby("h").agg(t=("true_incremental_payment", "sum"), p=(arm, "sum"))
                per_seed.append((g.p / g.t - 1) * 100)
            if not per_seed: continue
            M = pd.concat(per_seed, axis=1)
            for hh in M.index:
                rows.append(dict(scenario=scen, series=lab, config=cfg, arm=arm, h=int(hh), n_seeds=M.shape[1],
                                 mean_err_pct=M.loc[hh].mean(), min_err_pct=M.loc[hh].min(), max_err_pct=M.loc[hh].max(), sd_err_pct=M.loc[hh].std()))
    out = os.path.join(ROOT, "results_sim_ms", "horizon_errors_sim.csv"); pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out)

if __name__ == "__main__":
    real(); sim()
