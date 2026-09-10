#!/usr/bin/env python3
"""Back-test of the flagship calibration at an earlier valuation date (Paper B, Sect. 6.3; Paper A, Sect. S5).

Compares the run at valuation year I (results_v2_val<I>/, produced with RESERVING_VALUATION_YEAR=I) with the deployed
run at year 10 (results_v2/):
  * per development year k: the cross-fitted factor alpha_k at year I, its 95% delta interval, the ratio realised on the
    horizon-one diagonal (calendar year I+1, fully observed) and on the whole lower triangle of the I-year valuation
    (cells with a + k - 1 > I, k <= 10), the factor estimated at year 10, and the realised ratio there;
  * the arm totals at year I (none / heuristic / unshrunk per-DY / Buhlmann-Straub per-DY) on the lower triangle and on
    the horizon-one diagonal alone.
Usage:  python tools/backtest_val9.py [--year 9]   ->  results_v2_val<I>/backtest_summary.csv (+ printed tables)
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAGSHIP = "random_forest__per_dy__log1p__two_stage"
ARMS = ["pred__none", "pred__paper", "pred__cv_dy", "pred__cv_dy_bs1"]


def realised(df, col):
    t, p = df["true_incremental_payment"].sum(), df[col].sum()
    return t / p if p > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, default=9); a = ap.parse_args()
    I = a.year; d9 = os.path.join(ROOT, f"results_v2_val{I}"); d10 = os.path.join(ROOT, "results_v2")
    p9 = pd.read_csv(os.path.join(d9, "partial", f"predictions__{FLAGSHIP}.csv"))
    p10 = pd.read_csv(os.path.join(d10, "partial", f"predictions__{FLAGSHIP}.csv"))
    c9 = pd.read_csv(os.path.join(d9, "calibration_factors.csv")); c9 = c9[(c9.name == FLAGSHIP) & (c9.level == "dy")].set_index("group")
    c10 = pd.read_csv(os.path.join(d10, "calibration_factors.csv")); c10 = c10[(c10.name == FLAGSHIP) & (c10.level == "dy")].set_index("group")
    p9["h"] = p9["accident_year"] + p9["dev_lag"] - 1 - I; p10["h"] = p10["accident_year"] + p10["dev_lag"] - 1 - 10  # calendar year minus valuation year
    rows = []
    c9.index = c9.index.astype(float).astype(int); c10.index = c10.index.astype(float).astype(int)  # 'group' is read as text
    # development years without training cells at year I (DY 10) have no model: their predictions are NaN in every arm
    fit9 = p9["pred__none"].notna(); fit10 = p10["pred__none"].notna()
    for k in sorted(set(c9.index) | set(c10.index) | set(p9.loc[fit9, "dev_lag"].unique())):
        r = dict(dev_lag=int(k))
        if k in c9.index:
            r.update(alpha9=c9.loc[k, "alpha_cv"], se9=np.sqrt(c9.loc[k, "v_cv"]), n9=int(c9.loc[k, "n"]), Zbs9=c9.loc[k, "Z_bs1"])
        q = p9[(p9.dev_lag == k) & fit9]
        if len(q):  # realised ratios exist wherever a model predicts, whether or not a factor was estimated
            r.update(realised9_h1=realised(q[q.h == 1], "pred__none"), realised9_lower=realised(q, "pred__none"), n9_h1=int((q.h == 1).sum()))
            if k in c9.index:
                r["inside9_h1"] = abs(r["realised9_h1"] - r["alpha9"]) <= 1.96 * r["se9"]
        if k in c10.index:
            r.update(alpha10=c10.loc[k, "alpha_cv"], se10=np.sqrt(c10.loc[k, "v_cv"]))
        q = p10[(p10.dev_lag == k) & fit10]
        if len(q):
            r.update(realised10_h1=realised(q[q.h == 1], "pred__none"), realised10_lower=realised(q, "pred__none"))
        rows.append(r)
    tab = pd.DataFrame(rows)
    print("== per-DY factors and realised ratios (valuation 9 vs 10) =="); print(tab.round(3).to_string(index=False))
    tot = []
    for arm in ARMS:
        for lab, qa in (("lower", p9), ("h1", p9[p9.h == 1])):
            q = qa[qa["pred__none"].notna()]  # totals over the cells with a fitted model; the rest is reported, not zero-filled
            miss = qa.loc[qa["pred__none"].isna(), "true_incremental_payment"].sum()
            tot.append(dict(arm=arm[6:], scope=lab, err_pct=100 * (q[arm].sum() / q["true_incremental_payment"].sum() - 1),
                            true_M=q["true_incremental_payment"].sum() / 1e6, n_cells=len(q), n_unfitted=int(qa["pred__none"].isna().sum()),
                            unfitted_true_M=miss / 1e6, unfitted_share_pct=100 * miss / qa["true_incremental_payment"].sum(),
                            unfitted_dev_lags=",".join(str(int(x)) for x in sorted(qa.loc[qa["pred__none"].isna(), "dev_lag"].unique()))))
    tot = pd.DataFrame(tot); print("== arm totals at valuation 9 =="); print(tot.round(2).to_string(index=False))
    tab.to_csv(os.path.join(d9, "backtest_summary.csv"), index=False); tot.to_csv(os.path.join(d9, "backtest_totals.csv"), index=False)
    print("wrote", os.path.join(d9, "backtest_summary.csv"))


if __name__ == "__main__":
    main()
