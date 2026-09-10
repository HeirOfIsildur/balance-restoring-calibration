"""
Diagnostics computed from a dumped cross-fitted calibration record of the flagship
per-DY configuration (RESERVING_DUMP_CALIB_RECORD=1 main.py ... --results-dir <dir>):

  1. Residual imbalance of the heuristic (paper) arm on the record: the per-type shrunk and
     clipped factors of src/corrections.paper_factors applied to the out-of-fold predictions,
     sum(factor * f) / sum(y) - 1 per development year and overall (the unshrunk global ratio
     restores the balance exactly; this quantifies "approximately after shrinkage").
  2. Pooled vs fold-weighted delta-method variance (Paper B, Theorem 4.6 plug-in):
     pooled  = var(y - alpha f) / (n fbar^2)               (src/corrections.delta_variance)
     fold    = sum_j (n_j/n) var_j(y - alpha f) / (n fbar^2)  (within-fold variances, fold means removed)

Claim types are joined from the claims file (claim_no -> claim_type).

    python tools/record_diagnostics.py [results_dir] [claims_csv]
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import corrections  # noqa: E402
import src.config as cfg  # noqa: E402

CFG = "random_forest__per_dy__log1p__two_stage"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_v2_rec")
    claims_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(cfg.DATA_DIR, "claims_individual.csv")
    rec = pd.read_csv(os.path.join(d, "partial", f"calibration_record__{CFG}.csv"))
    claims = pd.read_csv(claims_csv, usecols=["claim_no", "claim_type"])
    rec = rec.merge(claims.rename(columns={"claim_no": "claim_id"}), on="claim_id", how="left")
    assert rec.claim_type.notna().all(), "claim types missing for some record rows"
    rows = []
    tot = dict(y=0.0, paper=0.0, f=0.0)
    for k, g in rec.groupby("group"):
        r = {"y": g.y.values, "oos_pred": g.oos_pred.values, "fold_id": g.fold_id.values,
             "claim_type": g.claim_type.values}
        fac = corrections.paper_factors(r)
        per_row = np.array([fac.get(t, fac[corrections.GLOBAL_CALIB_KEY]) for t in g.claim_type.values])
        y, f = g.y.values, g.oos_pred.values
        a = y.sum() / f.sum()
        paper_sum = float((per_row * f).sum())
        n = len(y); fbar = f.mean()
        e = y - a * f
        v_pooled = np.var(e, ddof=1) / (n * fbar ** 2)
        wv = 0.0
        for j in np.unique(g.fold_id.values):
            m = g.fold_id.values == j
            if m.sum() > 1:
                wv += m.sum() / n * np.var(e[m], ddof=1)
        v_fold = wv / (n * fbar ** 2)
        rows.append(dict(dy=int(k), n=n, alpha_cv=a, global_factor=fac[corrections.GLOBAL_CALIB_KEY],
                         paper_residual_pct=(paper_sum / y.sum() - 1) * 100,
                         unshrunk_residual_pct=(a * f.sum() / y.sum() - 1) * 100,
                         v_pooled=v_pooled, v_foldweighted=v_fold, rel_diff_pct=(v_fold / v_pooled - 1) * 100))
        tot["y"] += y.sum(); tot["paper"] += paper_sum; tot["f"] += f.sum()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(d, "record_diagnostics.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(out.to_string(index=False))
    print(f"\noverall paper-arm residual on the record: {(tot['paper'] / tot['y'] - 1) * 100:+.2f}%  "
          f"(uncorrected: {(tot['f'] / tot['y'] - 1) * 100:+.1f}%)")
    print(f"max |fold-weighted vs pooled variance| = {out.rel_diff_pct.abs().max():.2f}% (relative)")


if __name__ == "__main__":
    main()
