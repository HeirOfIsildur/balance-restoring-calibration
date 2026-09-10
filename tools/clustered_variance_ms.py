"""
Claim-clustered variances for the multi-step arm (Paper B, Section 6.3 / Table 2).

Reads the rerun of the multi-step log-target two-stage forest made with
RESERVING_DUMP_CALIB_RECORD=1 (results_ms_cl/), whose calibration table carries, per
horizon h, the pooled delta-method variance v_cv_delta, its claim-clustered version
v_cv_delta_cl (src/corrections.delta_variance_clustered), the number of claims
n_clusters, and the Buhlmann-Straub weights and factors computed from each
(Z_bs1 / Z_bs1_cl, factor_cv_bs1 / factor_cv_bs1_cl).  The clustered arm's
lower-triangle total is obtained by rescaling the per-horizon calibrated
predictions: pred__cv_dy = alpha_h * base, so clustered = base * factor_cv_bs1_cl.

Writes results_ms_cl/clustered_summary.csv (one row per horizon + a 'total' row over the whole lower
triangle, horizons without a calibration row keeping the arm's fallback prediction) and
checks that the rerun reproduces results_v2/ (same seeds, same tuning budget).

    python tools/clustered_variance_ms.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "random_forest__multi_step__log1p__two_stage"
I = 10


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_ms_cl")
    cal = pd.read_csv(os.path.join(d, "partial", f"calibration__{CFG}.csv"))
    h = cal[(cal.name == CFG) & (cal.level == "dy")].copy()
    h["group"] = h.group.astype(int)
    h = h.sort_values("group").set_index("group")
    ref_path = os.path.join(ROOT, "results_v2", "partial", f"calibration__{CFG}.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path); ref = ref[(ref.name == CFG) & (ref.level == "dy")].copy()
        ref["group"] = ref.group.astype(int); ref = ref.set_index("group")
        diff = (h.alpha_cv - ref.alpha_cv.reindex(h.index)).abs().max()
        print(f"reproduction check: max |alpha_h(rerun) - alpha_h(results_v2)| = {diff:.4f}")
    pred = pd.read_csv(os.path.join(d, "partial", f"predictions__{CFG}.csv"))
    pred["h"] = pred.accident_year + pred.dev_lag - 1 - I
    rows = []
    tot = dict(true=0.0, bs=0.0, bs_cl=0.0, none=0.0)
    for hh, r in h.iterrows():
        p = pred[pred.h == hh]
        base = p["pred__none"].values
        true = p.true_incremental_payment.sum()
        bs = (base * r.factor_cv_bs1).sum(); bs_cl = (base * r.factor_cv_bs1_cl).sum()
        tot["true"] += true; tot["bs"] += bs; tot["bs_cl"] += bs_cl; tot["none"] += base.sum()
        rows.append(dict(h=hh, n_cells=int(r.n), n_claims=int(r.n_clusters), cells_per_claim=r.n / r.n_clusters,
                         alpha=r.alpha_cv, v_delta=r.v_cv_delta, v_delta_cl=r.v_cv_delta_cl,
                         var_ratio=r.v_cv_delta_cl / r.v_cv_delta, Z=r.Z_bs1, Z_cl=r.Z_bs1_cl,
                         factor=r.factor_cv_bs1, factor_cl=r.factor_cv_bs1_cl,
                         err_bs_pct=(bs / true - 1) * 100 if true > 0 else np.nan,
                         err_bs_cl_pct=(bs_cl / true - 1) * 100 if true > 0 else np.nan))
    # horizons without an estimable calibration row keep the arm's existing (fallback) prediction in both totals
    rest = pred[~pred.h.isin(h.index)]
    tot["true"] += rest.true_incremental_payment.sum(); tot["none"] += rest["pred__none"].sum()
    tot["bs"] += rest["pred__cv_dy_bs1"].sum(); tot["bs_cl"] += rest["pred__cv_dy_bs1"].sum()
    rows.append(dict(h="total", n_cells=int(h.n.sum()), n_claims=np.nan, cells_per_claim=np.nan, alpha=np.nan,
                     v_delta=np.nan, v_delta_cl=np.nan, var_ratio=np.nan, Z=np.nan, Z_cl=np.nan, factor=np.nan,
                     factor_cl=np.nan, err_bs_pct=(tot["bs"] / tot["true"] - 1) * 100,
                     err_bs_cl_pct=(tot["bs_cl"] / tot["true"] - 1) * 100))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, "clustered_summary.csv"), index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
    a_bs, a_cl = h.a_bs1.iloc[0], h.a_bs1_cl.iloc[0]
    print(f"structural variance a_hat: pooled {a_bs:.3f}, clustered {a_cl:.3f}; uncorrected total {(tot['none']/tot['true']-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
