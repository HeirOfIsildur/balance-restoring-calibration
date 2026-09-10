"""
Share of the realised lower-triangle payments that fall on cells with a zero (uncorrected) prediction, by horizon
h = accident year + development year - 1 - I (Remark 'Zero predictions' of the IME paper).

    python tools/zero_set_shares.py [results_dir] [config ...]
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I = 10


def shares(path):
    d = pd.read_csv(path)
    d["h"] = d.accident_year + d.dev_lag - 1 - I
    d = d[d.h >= 1]
    zero = d["pred__none"] <= 0
    g = d.groupby("h")
    out = pd.DataFrame({
        "share_of_payments_on_zero_pred_pct": 100 * d[zero].groupby("h")["true_incremental_payment"].sum().reindex(g.size().index).fillna(0) / g["true_incremental_payment"].sum(),
        "share_of_cells_zero_pred_pct": 100 * zero.groupby(d.h).mean(),
    })
    tot = 100 * d.loc[zero, "true_incremental_payment"].sum() / d["true_incremental_payment"].sum()
    return out.round(1), round(tot, 1)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
    cfgs = sys.argv[2:] or ["random_forest__per_dy__log1p__two_stage", "random_forest__multi_step__log1p__two_stage"]
    for c in cfgs:
        p = os.path.join(ROOT, d, "partial", f"predictions__{c}.csv")
        if not os.path.exists(p):
            print(c, "missing"); continue
        t, tot = shares(p)
        print(f"{d} / {c}: total share of payments on zero-prediction cells {tot}%")
        print(t.T.to_string(), "\n")


if __name__ == "__main__":
    main()
