#!/usr/bin/env python3
"""Estimate the transfer term of Theorem 4.11 by importance weighting (Paper B, Sect. 4.5 / 6.3).

Under Assumption A3 the target cells' reduced covariates are observed, so the density ratio rho = dQ/dP of the
reduced covariate law can be estimated from the two triangles and E_Q[Y]/E_Q[f] = E_P[rho Y]/E_P[rho f].  For every
development year k of the flagship per-DY log1p two-stage forest this script
  * takes the cross-fitted calibration record (training cells, out-of-fold predictions),
  * fits a probabilistic classifier "training cell vs target cell" on the reduced covariates (no accident-year
    coordinates), cross-fitted over 5 folds so that no cell is weighted by a model that saw it,
  * turns the class probabilities into density-ratio weights rho_i = p_i/(1-p_i) * n_P/n_Q,
  * and reports the plain cross-fitted ratio alpha_P, the importance-weighted ratio alpha_Q^IW, and the ratio realised
    on the target cells, for two target populations: the horizon-one diagonal (h = 1, the clean case of the theorem)
    and the whole lower triangle at that development year (the deployment target, which for h >= 2 mixes the covariate
    shift with the feature-construction mismatch of Proposition 2.3).
Also reported: the effective sample size of the weights, the classifier's cross-fitted AUC (overlap diagnostic) and
the implied transfer term alpha_Q^IW - alpha_P.

Usage:  python tools/transfer_iw.py [--results-dir results_v2]   ->  <results-dir>/transfer_iw.csv
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import src.config as cfg                                   # noqa: E402
from src.data_prep import prepare_data                     # noqa: E402
from src.feature_engineering import engineer_features      # noqa: E402

FLAGSHIP = "random_forest__per_dy__log1p__two_stage"
AY_COORDS = ["accident_year", "occurrence_period", "occurrence_month"]


def density_ratio_weights(XP, XQ, seed=0):
    """Cross-fitted classifier weights rho_i for the P rows: p/(1-p) * n_P/n_Q.  Returns (rho, auc)."""
    X = np.vstack([XP, XQ]); lab = np.r_[np.zeros(len(XP)), np.ones(len(XQ))]
    p = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, lab):
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
        clf.fit(X[tr], lab[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    auc = roc_auc_score(lab, p)
    rho = (p[: len(XP)] / (1 - p[: len(XP)])) * (len(XP) / len(XQ))
    return rho, auc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default="results_v2")
    ap.add_argument("--annotate", action="store_true", help="only add n_P_pos (positive training cells per DY) to an existing transfer_iw.csv")
    a = ap.parse_args()
    rdir = os.path.join(ROOT, a.results_dir); t0 = time.time()
    if a.annotate:
        rec = pd.read_csv(os.path.join(rdir, "partial", f"calibration_record__{FLAGSHIP}.csv"))
        pos = rec.assign(pos=rec["y"] > 0).groupby("group")["pos"].sum().astype(int)
        path = os.path.join(rdir, "transfer_iw.csv"); out = pd.read_csv(path)
        out["n_P_pos"] = out["dev_lag"].map(lambda k: int(pos.get(k, pos.get(float(k), 0))))
        cols = [c for c in out.columns if c != "n_P_pos"]; i = cols.index("n_P") + 1
        out = out[cols[:i] + ["n_P_pos"] + cols[i:]]; out.to_csv(path, index=False)
        print(out[["dev_lag", "target", "n_P", "n_P_pos"]].to_string(index=False)); print("annotated", path); return
    claim_dy, _, transactions_obs, _ = prepare_data()
    claim_dy, feature_cols, feature_cols_per_dy, = engineer_features(claim_dy, transactions_obs)
    obs = (claim_dy["accident_year"] + claim_dy["dev_lag"] - 1) <= cfg.MAX_ACCIDENT_YEAR
    pre = obs & (claim_dy["dev_lag"] < claim_dy["report_dy"]); claim_dy = claim_dy.loc[~pre].reset_index(drop=True)
    feats = [c for c in feature_cols_per_dy if c not in AY_COORDS]
    rec = pd.read_csv(os.path.join(rdir, "partial", f"calibration_record__{FLAGSHIP}.csv"))
    pred = pd.read_csv(os.path.join(rdir, "partial", f"predictions__{FLAGSHIP}.csv"),
                       usecols=["claim_no", "accident_year", "dev_lag", "true_incremental_payment", "pred__none"])
    print(f"panel ready ({time.time()-t0:.0f}s); reduced features: {len(feats)}", flush=True)
    rows = []
    for k in sorted(rec["group"].unique()):
        r = rec[rec["group"] == k].merge(claim_dy[["claim_no", "dev_lag", "accident_year"] + feats],
                                         left_on=["claim_id", "dev_lag"], right_on=["claim_no", "dev_lag"], how="inner")
        if len(r) < 50 or r["oos_pred"].sum() <= 0: continue
        XP = r[feats].to_numpy(float)
        alpha_P = r["y"].sum() / r["oos_pred"].sum()
        tgt = claim_dy[(claim_dy["dev_lag"] == k) & ((claim_dy["accident_year"] + k - 1) > cfg.MAX_ACCIDENT_YEAR)]
        tgt = tgt.merge(pred[["claim_no", "dev_lag", "pred__none"]], on=["claim_no", "dev_lag"], how="left")
        for label, Q in (("h1", tgt[tgt["accident_year"] + k - 1 == cfg.MAX_ACCIDENT_YEAR + 1]), ("lower", tgt)):
            if len(Q) < 50 or Q["pred__none"].sum() <= 0: continue
            rho, auc = density_ratio_weights(XP, Q[feats].to_numpy(float))
            alpha_iw = np.sum(rho * r["y"]) / np.sum(rho * r["oos_pred"])
            realised = Q["true_incremental_payment"].sum() / Q["pred__none"].sum()
            # delta-method standard error of the realised ratio (process noise of the target cells, fixed predictor)
            e = Q["true_incremental_payment"].to_numpy(float) - realised * Q["pred__none"].to_numpy(float)
            se_real = np.sqrt(np.var(e, ddof=1) / len(Q)) / Q["pred__none"].mean()
            ess = rho.sum() ** 2 / np.sum(rho ** 2)
            rows.append(dict(dev_lag=int(k), target=label, n_P=len(r), n_P_pos=int((r["y"] > 0).sum()), n_Q=len(Q), alpha_P=alpha_P, alpha_Q_iw=alpha_iw,
                             transfer_term_iw=alpha_iw - alpha_P, realised_ratio=realised, realised_se=se_real, auc=auc, ess=ess,
                             ess_share=ess / len(r), rho_max=rho.max()))
            print(f"DY{k} {label:5s} nP={len(r):6d} nQ={len(Q):6d} alpha_P={alpha_P:.3f} alpha_Q_iw={alpha_iw:.3f} realised={realised:.3f}±{se_real:.3f} auc={auc:.3f} ess={ess:.0f}", flush=True)
    out = pd.DataFrame(rows); path = os.path.join(rdir, "transfer_iw.csv"); out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False)); print("wrote", path, f"({time.time()-t0:.0f}s)"); print("TRANSFER IW DONE")


if __name__ == "__main__":
    main()
