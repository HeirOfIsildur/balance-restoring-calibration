"""
Fixed-predictor coverage experiment for Theorem 4.6(i) of Paper B (referee request, 9 Sep 2026).

A FIXED positive predictor f (no fitting) is scored on every observed cell of each simulated portfolio:
    f(c, k, r) = exp(mu_c + beta_c (k - r)) - 1,
the generator's type-level severity curve without frailty (src/simulate.TYPE_PARAMS), a function of observed
covariates only (claim type, development year, reporting year).  For each portfolio (scenario x seed) and development
year k the cross-fitted-type ratio reduces to the plain ratio alpha_hat_k = sum Y / sum f over the observed cells of
DY k whose claim is reported (k >= report_dy), with the delta-method variance of Theorem 4.6(i).  The population
target alpha_0,k = E_P[Y] / E_P[f] under the training law is estimated on the ten independent fresh replicate portfolios
of the same scenario (data/sim_cov/<sc>/seedXX_fresh), pooled, with the realised payments replaced by the generator's
conditional means; its Monte Carlo standard error is reported.  Coverage of the nominal 95% interval over the 30
portfolios per DY (and pooled over scenarios) tests the fixed-predictor asymptotics directly, separately from any
deployment or cross-fitting effect.

    python tools/fixed_predictor_coverage.py [--data data/sim_cov] [--out results_sim_cov]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.simulate import TYPE_PARAMS  # noqa: E402


def f_fixed(df):
    mu = df.claim_type.map(lambda c: TYPE_PARAMS[c]["mu"]).astype(float)
    beta = df.claim_type.map(lambda c: TYPE_PARAMS[c]["beta"]).astype(float)
    return np.exp(mu + beta * (df.dev_lag - df.report_dy)) - 1.0


def cells(data_dir):
    obs = pd.read_csv(os.path.join(data_dir, "truth_cells_observed.csv"))
    cl = pd.read_csv(os.path.join(data_dir, "claims_individual.csv"), usecols=["claim_no", "report_dy"])
    d = obs.merge(cl, on="claim_no", how="left")
    d = d[d.dev_lag >= d.report_dy].copy()      # reporting-consistent risk set
    d["f"] = f_fixed(d)
    return d[d.f > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "sim_cov"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results_sim_cov"))
    ap.add_argument("--level", type=float, default=0.95)
    a = ap.parse_args()
    z = norm.ppf(0.5 + a.level / 2)
    rows = []
    for sc in sorted(os.listdir(a.data)):
        seeds = sorted(glob.glob(os.path.join(a.data, sc, "seed[0-9][0-9]")))
        fresh = [cells(s + "_fresh") for s in seeds if os.path.isdir(s + "_fresh")]
        if not seeds or not fresh:
            continue
        pool = pd.concat(fresh, ignore_index=True)
        # population target per DY from the pooled fresh replicates (oracle means), with its MC standard error
        tgt = {}
        for k, g in pool.groupby("dev_lag"):
            a0 = g.oracle_payment.sum() / g.f.sum()
            se0 = np.sqrt(np.var(g.oracle_payment - a0 * g.f, ddof=1) / (len(g) * g.f.mean() ** 2))
            tgt[k] = (a0, se0, len(g))
        for s in seeds:
            d = cells(s)
            for k, g in d.groupby("dev_lag"):
                if len(g) < 10 or (g.realized_payment > 0).sum() < 10 or k not in tgt:
                    continue
                y, f = g.realized_payment.values, g.f.values
                ah = y.sum() / f.sum()
                v = np.var(y - ah * f, ddof=1) / (len(g) * f.mean() ** 2)
                a0, se0, n0 = tgt[k]
                rows.append(dict(scenario=sc, seed=os.path.basename(s), dy=int(k), n=len(g), alpha_hat=ah, se=np.sqrt(v),
                                 alpha_0=a0, se_alpha_0=se0, n_pool=n0, cover=float(abs(ah - a0) <= z * np.sqrt(v))))
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "fixed_predictor_coverage_by_replicate.csv"), index=False)
    g = df.groupby("dy")
    summ = pd.DataFrame({"replicates": g.size(), "coverage": g.cover.mean(),
                         "mean_alpha_hat": g.alpha_hat.mean(), "mean_alpha_0": g.alpha_0.mean(),
                         "se_over_sd": g.se.mean() / df.assign(d=df.alpha_hat - df.alpha_0).groupby("dy").d.std(),
                         "target_se_over_interval_se": (g.se_alpha_0.mean() / g.se.mean())}).reset_index()
    total = pd.DataFrame([{"dy": "all", "replicates": len(df), "coverage": df.cover.mean(), "mean_alpha_hat": np.nan,
                           "mean_alpha_0": np.nan, "se_over_sd": np.nan, "target_se_over_interval_se": np.nan}])
    summ = pd.concat([summ, total], ignore_index=True)
    summ.to_csv(os.path.join(a.out, "fixed_predictor_coverage_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summ.to_string(index=False))
    print("coverage by scenario:", df.groupby("scenario").cover.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
