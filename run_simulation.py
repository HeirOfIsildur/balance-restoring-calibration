#!/usr/bin/env python3
"""
Simulation study: scenarios x seeds through the full pipeline (Section 5).

For each (scenario, seed) a portfolio is generated with src/simulate.py, the
pipeline is pointed at it (data dir, results dir), the reduced arm set is run,
and the per-arm total-reserve errors against both the realized and the oracle
reserve are collected.  Everything is aggregated into results_sim/summary.csv
and results_sim/summary_by_arm.csv.

Usage:
  python run_simulation.py                                  # S-A S-B S-C x 10 seeds
  python run_simulation.py --scenarios S-B --seeds 3 --n-per-ay 1000 --n-trials 5
  python run_simulation.py --objectives mse_log1p poisson_rf --structures per_dy single_model
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.config as cfg  # noqa: E402
from src.simulate import SCENARIOS, generate_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Simulation study for the revised manuscript")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--seeds", type=int, default=10, help="number of seeds per scenario (offset..offset+seeds-1)")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--n-per-ay", type=int, default=2000)
    p.add_argument("--s", type=float, default=1.0, help="log-scale sd of payments (homoscedastic)")
    p.add_argument("--tau", type=float, default=0.5, help="claim frailty sd")
    p.add_argument("--rho", type=float, default=0.5, help="activity persistence")
    p.add_argument("--objectives", nargs="+", default=["mse_log1p", "poisson_rf"])
    p.add_argument("--structures", nargs="+", default=["per_dy", "single_model"])
    p.add_argument("--zero-handling", nargs="+", default=["two_stage"])
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--data-root", default=os.path.join(cfg.BASE_DIR, "data", "sim"))
    p.add_argument("--out", default=os.path.join(cfg.BASE_DIR, "results_sim"))
    p.add_argument("--skip-existing", action="store_true", help="skip (scenario, seed) with results already present")
    p.add_argument("--fresh-eval", action="store_true",
                   help="also generate an independent replicate (seed+1000) and store the per-DY final models' "
                        "uncorrected predictions on its upper-triangle cells (partial/fresh_pred__<config>.csv)")
    return p.parse_args()


def _cl_row(comp, oracle_total):
    """One summary row per chain-ladder variant (RBNS-cohort, total, legacy)."""
    out = []
    for _, cl in comp[comp["method"] == "chain_ladder"].iterrows():
        pred = float(cl["total_predicted_reserve"])
        true = float(cl["total_true_reserve"])
        # the oracle is the RBNS-cohort expectation; only comparable for RBNS rows
        rbns_target = cl["name"] != "chain_ladder_total"
        out.append({
            "name": cl["name"], "objective": "chain_ladder", "method": "chain_ladder",
            "structure": "aggregate", "target": cl.get("target", "cumulative"),
            "zero_handling": "N/A", "arm": "none",
            "total_predicted_reserve": pred, "total_true_reserve": true,
            "total_reserve_error": pred - true,
            "total_reserve_error_pct": float(cl["total_reserve_error_pct"]),
            "max_ay_error_pct": float(cl["max_ay_error_pct"]) if pd.notna(cl["max_ay_error_pct"]) else np.nan,
            "rmse": np.nan, "mae": np.nan,
            "oracle_reserve": oracle_total if rbns_target else np.nan,
            "error_vs_oracle_pct": (pred - oracle_total) / oracle_total * 100.0
            if (rbns_target and oracle_total > 0) else np.nan,
        })
    return out or None


def run_one(scenario, seed, args):
    from src.data_prep import prepare_data
    from src.feature_engineering import engineer_features
    from src.chain_ladder import run_chain_ladder
    from src.train import run_all_experiments
    from src.evaluate import (
        build_comparison_table,
        build_correction_table,
        collect_calibration_diagnostics,
    )

    tag = f"{scenario}/seed{seed:02d}"
    data_dir = os.path.join(args.data_root, scenario, f"seed{seed:02d}")
    res_dir = os.path.join(args.out, scenario, f"seed{seed:02d}")
    os.makedirs(res_dir, exist_ok=True)
    done_path = os.path.join(res_dir, "correction_comparison.csv")
    if args.skip_existing and os.path.exists(done_path):
        print(f"[{tag}] exists, skipping", flush=True)
        return pd.read_csv(done_path)

    t0 = time.time()
    sim = generate_dataset(scenario, seed, n_per_ay=args.n_per_ay, s=args.s, tau=args.tau, rho=args.rho, out_dir=data_dir)
    truth = sim["summary"]
    print(f"[{tag}] generated {truth['n_claims']:,} claims ({truth['n_rbns']:,} RBNS); realized reserve "
          f"{truth['realized_rbns_reserve']:,.0f}, oracle {truth['oracle_rbns_reserve']:,.0f}", flush=True)

    # point the pipeline at this dataset
    cfg.DATA_DIR = data_dir
    cfg.RESULTS_DIR = res_dir
    cfg.ZERO_HANDLING = list(args.zero_handling)

    claim_dy, claims_rbns, transactions_obs, triangles = prepare_data()
    claim_dy, feature_cols, feature_cols_per_dy = engineer_features(claim_dy, transactions_obs)
    cl = run_chain_ladder()
    import src.train as train_mod
    train_mod.FRESH_EVAL = None
    if args.fresh_eval:
        fresh_dir = data_dir + "_fresh"
        generate_dataset(scenario, seed + 1000, n_per_ay=args.n_per_ay, s=args.s, tau=args.tau, rho=args.rho, out_dir=fresh_dir)
        cfg.DATA_DIR = fresh_dir
        f_claim_dy, _, f_txn, _ = prepare_data()
        import src.feature_engineering as fe_mod
        fe_mod.AVG_ULTIMATE_OVERRIDE = fe_mod.LAST_AVG_ULTIMATE  # deployment: reuse the training portfolio's reference ultimate
        try:
            f_claim_dy, _, f_cols_dy = engineer_features(f_claim_dy, f_txn)
        finally:
            fe_mod.AVG_ULTIMATE_OVERRIDE = None
        cfg.DATA_DIR = data_dir
        missing = [c for c in feature_cols_per_dy if c not in f_claim_dy.columns]
        for c in missing:
            f_claim_dy[c] = 0.0
        train_mod.FRESH_EVAL = {"claim_dy": f_claim_dy, "feat_cols": list(feature_cols_per_dy)}
        print(f"[{tag}] fresh replicate seed {seed + 1000}: {len(f_claim_dy):,} cells" + (f", {len(missing)} missing feature(s) zero-filled" if missing else ""), flush=True)
    results = run_all_experiments(
        claim_dy, feature_cols, feature_cols_per_dy, n_trials=args.n_trials, n_jobs=args.n_jobs,
        structures=args.structures, objectives=args.objectives, n_folds=args.n_folds,
        partial_dir=os.path.join(res_dir, "partial"),
    )
    if args.fresh_eval and train_mod.FRESH_EVAL and train_mod.FRESH_EVAL.get("preds"):
        fp = pd.concat(train_mod.FRESH_EVAL["preds"], ignore_index=True)
        if "config" in fp.columns:  # one file per configuration, never mixed
            for cfg_name, g in fp.groupby("config"):
                g.drop(columns="config").to_csv(os.path.join(res_dir, "partial", f"fresh_pred__{cfg_name}.csv"), index=False)
        else:
            fp.to_csv(os.path.join(res_dir, "partial", f"fresh_pred__{cfg.DEFAULT_BEST_MODEL}.csv"), index=False)
    train_mod.FRESH_EVAL = None
    comp = build_comparison_table(results, cl)
    corr = build_correction_table(results)
    diag = collect_calibration_diagnostics(results)

    oracle_total = float(truth["oracle_rbns_reserve"])
    corr["oracle_reserve"] = oracle_total
    corr["error_vs_oracle_pct"] = (corr["total_predicted_reserve"] - oracle_total) / oracle_total * 100.0
    cl_rows = _cl_row(comp, oracle_total)
    if cl_rows is not None:
        corr = pd.concat([corr, pd.DataFrame(cl_rows)], ignore_index=True)
    corr.insert(0, "seed", seed)
    corr.insert(0, "scenario", scenario)
    if not diag.empty:
        diag.insert(0, "seed", seed)
        diag.insert(0, "scenario", scenario)

    comp.to_csv(os.path.join(res_dir, "model_comparison.csv"), index=False)
    corr.to_csv(done_path, index=False)
    diag.to_csv(os.path.join(res_dir, "calibration_factors.csv"), index=False)
    with open(os.path.join(res_dir, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2, default=str)
    print(f"[{tag}] done in {time.time() - t0:.0f}s", flush=True)
    return corr


def aggregate(frames, out):
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    g = (
        df.groupby(["scenario", "name", "arm"])
        .agg(
            n_seeds=("seed", "nunique"),
            mean_error_pct=("total_reserve_error_pct", "mean"),
            sd_error_pct=("total_reserve_error_pct", "std"),
            rmse_error_pct=("total_reserve_error_pct", rmse),
            mean_error_vs_oracle_pct=("error_vs_oracle_pct", "mean"),
            rmse_vs_oracle_pct=("error_vs_oracle_pct", rmse),
        )
        .reset_index()
    )
    g.to_csv(os.path.join(out, "summary_by_arm.csv"), index=False)
    return g


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(args.out, exist_ok=True)
    frames = []
    for scenario in args.scenarios:
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            frames.append(run_one(scenario, seed, args))
    g = aggregate(frames, args.out)
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print("\n=== mean total reserve error (%) vs realized, by scenario x config x arm ===")
        print(g.pivot_table(index=["scenario", "name"], columns="arm", values="mean_error_pct").round(1).to_string())


if __name__ == "__main__":
    main()
