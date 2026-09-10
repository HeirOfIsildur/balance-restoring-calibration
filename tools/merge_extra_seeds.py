#!/usr/bin/env python3
"""Fold the extra S-A seeds (10-29, run with run_simulation.py --seed-offset into results_sim_x/ and
results_sim_ms_x/) into the deposited simulation directories and rebuild everything downstream:

    results_sim/S-A/seedNN, results_sim_ms/S-A/seedNN     copied (existing seeds untouched)
    results_sim/summary*.csv, results_sim_ms/summary*.csv  rebuilt from all per-seed correction_comparison.csv
    results_sim_ms/horizon_errors_sim.csv                   rebuilt (tools/deposit_sim_summaries.py)
    results/figures/fig1_sim_horizon.*, paper/ime/figures/  redrawn (tools/evidence_figures.py)
    paper/ime/tables/table_sim_scenarios.tex                regenerated (tools/make_tables.py)

Usage (repository root, project interpreter):  python tools/merge_extra_seeds.py [--dry-run]
"""
import argparse, glob, os, shutil, subprocess, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from run_simulation import aggregate  # noqa: E402

PAIRS = [("results_sim_x", "results_sim"), ("results_sim_ms_x", "results_sim_ms")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    for src, dst in PAIRS:
        for sd in sorted(glob.glob(os.path.join(ROOT, src, "S-A", "seed*"))):
            if not os.path.exists(os.path.join(sd, "correction_comparison.csv")):
                print(f"skip unfinished {sd}"); continue
            tgt = os.path.join(ROOT, dst, "S-A", os.path.basename(sd))
            if os.path.exists(tgt):
                print(f"exists {tgt}"); continue
            print(f"copy {sd} -> {tgt}")
            if not a.dry_run:
                shutil.copytree(sd, tgt)
        if a.dry_run:
            continue
        out = os.path.join(ROOT, dst)
        frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out, "S-*", "seed*", "correction_comparison.csv")))]
        g = aggregate(frames, out)
        print(f"{dst}: {len(frames)} seed files -> summary.csv, summary_by_arm.csv ({len(g)} rows); "
              f"S-A n_seeds = {sorted(g[g.scenario == 'S-A'].n_seeds.unique().tolist())}")
    if a.dry_run:
        return
    py = sys.executable
    subprocess.run([py, os.path.join(ROOT, "tools", "deposit_sim_summaries.py")], check=True, cwd=ROOT)
    subprocess.run([py, os.path.join(ROOT, "tools", "evidence_figures.py"), "--out", os.path.join(ROOT, "paper", "ime", "figures"),
                    "--pdf-only"], check=True, cwd=ROOT)
    subprocess.run([py, os.path.join(ROOT, "tools", "evidence_figures.py")], check=True, cwd=ROOT)
    subprocess.run([py, os.path.join(ROOT, "tools", "make_tables.py"), "--only", "table_sim_scenarios.tex"], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
