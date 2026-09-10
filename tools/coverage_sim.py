"""
Coverage experiment for the delta-method interval of the cross-fitted calibration
ratio (Paper B, Theorem 4.6 / Section 5(v)), against the fitted pipeline's own targets.

For every simulated portfolio (scenario x seed) the flagship per-DY pipeline was
refitted with RESERVING_DUMP_CALIB_RECORD=1 and --fresh-eval (run_simulation.py), so
three files exist per replicate:

  <results>/<sc>/seed<ss>/partial/calibration_record__<cfg>.csv   the cross-fitted record
        (group = development year k, claim_id, y, oos_pred = out-of-fold prediction f_i)
  <data>/<sc>/seed<ss>/truth_cells_observed.csv                   oracle mean m_i of every
        observed cell of the portfolio (generator's conditional mean given its latent state)
  <results>/<sc>/seed<ss>/partial/fresh_pred__<cfg>.csv           the final per-DY models'
        uncorrected predictions on the upper-triangle cells of an INDEPENDENT replicate
        portfolio (seed + 1000), joined with <data>/<sc>/seed<ss>_fresh/truth_cells_observed.csv

Targets per development year k:
  alpha_pop_k  = sum_fresh m / sum_fresh f_hat     population balance factor of the deployed
                 predictor under the training law P (Monte Carlo on the fresh replicate,
                 realised payments replaced by the oracle means)
  alpha_cond_k = sum_rec m / sum_rec f            the same functional on the estimation sample
                 (conditional on its covariates; a finite-population target)
Estimator: alpha_hat_k = sum_rec y / sum_rec f with the pooled delta-method variance
  v_hat_k = var(y - alpha_hat f) / (n f_bar^2)  (src/corrections.delta_variance), interval
  alpha_hat +/- z v_hat^{1/2}.

Outputs (in <results>/): coverage_by_replicate.csv and coverage_summary.csv (per DY:
coverage of both targets, mean relative bias of alpha_hat against alpha_pop, ratio of the mean
estimated standard error v_hat^{1/2} to the empirical s.d. of alpha_hat - alpha_pop across
replicates (1 = the plug-in variance matches the sampling spread), misses below/above the target).

    python tools/coverage_sim.py --results results_sim_cov --data data/sim_cov
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "random_forest__per_dy__log1p__two_stage"
MIN_N, MIN_POS = 10, 10  # CALIB_MIN_GROUP_N / CALIB_MIN_GROUP_POS (src/config.py)


def delta_variance(y, f, alpha):
    n = len(y)
    m = np.mean(f)
    if n < 2 or m <= 0 or not np.isfinite(alpha):
        return np.nan
    return float(np.var(y - alpha * f, ddof=1) / (n * m * m))


def one_replicate(res_dir, data_dir, sc, seed, level):
    rec = pd.read_csv(os.path.join(res_dir, "partial", f"calibration_record__{CFG}.csv"))
    obs = pd.read_csv(os.path.join(data_dir, "truth_cells_observed.csv"))
    fresh = pd.read_csv(os.path.join(res_dir, "partial", f"fresh_pred__{CFG}.csv"))
    fobs = pd.read_csv(os.path.join(data_dir + "_fresh", "truth_cells_observed.csv"))
    rec = rec.merge(obs[["claim_no", "dev_lag", "oracle_payment"]].rename(columns={"claim_no": "claim_id"}),
                    on=["claim_id", "dev_lag"], how="left")
    fresh = fresh.merge(fobs[["claim_no", "dev_lag", "oracle_payment"]], on=["claim_no", "dev_lag"], how="left")
    assert rec.oracle_payment.notna().all(), "unmatched record cells"
    assert fresh.oracle_payment.notna().all(), "unmatched fresh cells"
    z = norm.ppf(0.5 + level / 2)
    rows = []
    for k, g in rec.groupby("group"):
        y, f, m = g.y.values, g.oos_pred.values, g.oracle_payment.values
        if len(g) < MIN_N or (y > 0).sum() < MIN_POS or f.sum() <= 0:
            continue
        a_hat = y.sum() / f.sum()
        v_hat = delta_variance(y, f, a_hat)
        a_cond = m.sum() / f.sum()
        fg = fresh[fresh.dev_lag == k]
        a_pop = fg.oracle_payment.sum() / fg.pred.sum() if len(fg) and fg.pred.sum() > 0 else np.nan
        half = z * np.sqrt(v_hat)
        rows.append(dict(scenario=sc, seed=seed, dy=int(k), n=len(g), n_fresh=len(fg), alpha_hat=a_hat,
                         v_hat=v_hat, half_width=half, alpha_cond=a_cond, alpha_pop=a_pop,
                         cover_cond=float(abs(a_hat - a_cond) <= half),
                         cover_pop=float(abs(a_hat - a_pop) <= half) if np.isfinite(a_pop) else np.nan,
                         rel_bias_pop=a_hat / a_pop - 1 if np.isfinite(a_pop) else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results_sim_cov"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "sim_cov"))
    ap.add_argument("--level", type=float, default=0.95)
    a = ap.parse_args()
    rows = []
    for res_dir in sorted(glob.glob(os.path.join(a.results, "S-*", "seed[0-9]*"))):
        sc = os.path.basename(os.path.dirname(res_dir)); seed = int(os.path.basename(res_dir)[4:])
        data_dir = os.path.join(a.data, sc, os.path.basename(res_dir))
        if not os.path.exists(os.path.join(res_dir, "partial", f"fresh_pred__{CFG}.csv")):
            print("  (incomplete, skipped)", sc, seed); continue
        rows += one_replicate(res_dir, data_dir, sc, seed, a.level)
    if not rows:
        sys.exit("no complete replicates")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.results, "coverage_by_replicate.csv"), index=False)
    g = df.groupby("dy")
    summ = pd.DataFrame({
        "replicates": g.size(),
        "coverage_pop": g.cover_pop.mean(),
        "coverage_cond": g.cover_cond.mean(),
        "mean_alpha_hat": g.alpha_hat.mean(),
        "mean_alpha_pop": g.alpha_pop.mean(),
        "mean_rel_bias_pop_pct": g.rel_bias_pop.mean() * 100,
        "se_over_sd": g.v_hat.apply(lambda x: np.sqrt(x).mean()) / (df.assign(d=df.alpha_hat - df.alpha_pop).groupby("dy").d.std()),
        "misses_below": df.assign(b=((df.cover_pop == 0) & (df.alpha_hat < df.alpha_pop)).astype(float)).groupby("dy").b.sum(),
        "misses_above": df.assign(b=((df.cover_pop == 0) & (df.alpha_hat > df.alpha_pop)).astype(float)).groupby("dy").b.sum(),
    }).reset_index()
    total = pd.DataFrame([{"dy": "all", "replicates": len(df), "coverage_pop": df.cover_pop.mean(),
                           "coverage_cond": df.cover_cond.mean(), "mean_alpha_hat": np.nan, "mean_alpha_pop": np.nan,
                           "mean_rel_bias_pop_pct": df.rel_bias_pop.mean() * 100, "se_over_sd": np.nan,
                           "misses_below": float(((df.cover_pop == 0) & (df.alpha_hat < df.alpha_pop)).sum()),
                           "misses_above": float(((df.cover_pop == 0) & (df.alpha_hat > df.alpha_pop)).sum())}])
    summ = pd.concat([summ, total], ignore_index=True)
    summ.to_csv(os.path.join(a.results, "coverage_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summ.to_string(index=False))
    # per-scenario coverage of the population target
    print("\ncoverage of alpha_pop by scenario:", df.groupby("scenario").cover_pop.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
