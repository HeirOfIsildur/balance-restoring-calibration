# Balance-restoring calibration of transformed-target tree ensembles

Code and simulation for the paper *Balance-restoring calibration of transformed-target tree
ensembles for individual claims reserving* (J. Janoušek and M. Pešta, Charles University; submitted, 2026).

## Contents

- `src/` — the claim-level reserving pipeline (random forests on claim × development-year panels; `corrections.py`
  implements every post-hoc correction arm of the paper on one cross-fitted record: raw output, global and per-group
  cross-fitted ratios, Bühlmann–Straub shrinkage, cross-validation-chosen shrinkage, shifted ratios, Duan smearing,
  isotonic recalibration); `simulate.py` is the parametric individual-claims generator with an oracle conditional
  expectation (scenarios S-A stable, S-B mix shift, S-C calendar inflation).
- `main.py` — real-data run over objectives × structures × zero handling; `run_simulation.py` — scenarios × seeds.
- `tools/make_tables.py`, `tools/evidence_figures.py` (needs the run directories, not only the deposits), `tools/deposit_sim_summaries.py`, `tools/cohort_cl.py`, `tools/coverage_sim.py`, `tools/clustered_variance_ms.py`, `tools/record_diagnostics.py` — the
  scripts that produce every table and figure of the paper and the deposited summaries.
- `results/` — deposited summaries behind the paper's tables and Figures 1 and 3 (`horizon_errors_real.csv`,
  `correction_comparison.csv`, `calibration_factors.csv`, `chain_ladder_cohort.csv`; `sim/summary_by_arm.csv` and
  `sim_ms/summary_by_arm.csv` the per-DY/single-model and multi-step simulation summaries, `sim_ms/horizon_errors_sim.csv`;
  `k10/` the ten-fold check of Table 8; `noay/` the refit without the accident-year, occurrence-period and
  occurrence-month features (the last equals the accident year in annual data) of Table 9, produced with `RESERVING_EXCLUDE_AY_FEATURES=1`; `full25/` the same-budget full-feature refit with its
  record diagnostics; `ms_cl/` the multi-step rerun with claim-clustered variances (Table 7, `tools/clustered_variance_ms.py`);
  `sim_cov/` the coverage experiment of Section 5.3, Table 3 and the fixed-predictor check of Table 4
  (`run_simulation.py --fresh-eval`, `tools/coverage_sim.py`, `tools/fixed_predictor_coverage.py`)).
  **The real-data summaries were computed on the insurer's confidential full data; running the scripts on the
  illustrative subset in `data/` reproduces the procedure, not the numbers. The simulation summaries are
  reproducible from `run_simulation.py`.**

  Provenance: every deposited summary comes from a run of this code — the regressors tuned by validation RMSE,
  the reporting-consistent risk set (observed cells preceding a claim's reporting year excluded, and multi-step
  training cells dropped when the claim was not yet reported at the freeze date), and the clipped log-normal
  oracle in the simulation. `RESERVING_KEEP_PREREPORT=1` restores the earlier panel as a sensitivity.
- `data/` — an illustrative dataset provided by an insurance company (a modified subset of the original), so the
  pipeline runs end-to-end on a clean clone. The insurer's full data are confidential and are not included.

## Reproducing

```sh
uv sync --extra boosting                 # or: pip install -e .[boosting]
# real data: the full matrix (objectives x structures x zero handling) into results_v2/, per-cell partials included;
# the multi-step structure and the LightGBM objectives are not in the defaults and must be listed
RESERVING_DUMP_CALIB_RECORD=1 uv run python main.py --n-trials 50 --results-dir results_v2 \
    --structures single_model per_dy multi_step \
    --objectives mse_raw mse_log1p poisson_rf poisson_hgb gamma_hgb poisson_lgbm tweedie_lgbm
uv run python tools/cohort_cl.py --out results_v2/chain_ladder_cohort.csv               # cohort-consistent chain ladder
uv run python tools/cohort_cl_by_h.py                                                  # ... by horizon
uv run python main.py --n-trials 50 --n-folds 10 --results-dir results_v2_k10 --objectives mse_log1p --structures per_dy   # Table 8
RESERVING_EXCLUDE_AY_FEATURES=1 uv run python main.py --n-trials 25 --results-dir results_noay --objectives mse_log1p --structures per_dy  # Table 9
RESERVING_DUMP_CALIB_RECORD=1 uv run python main.py --n-trials 25 --results-dir results_v2_rec --objectives mse_log1p --structures per_dy   # same-budget control + record diagnostics
RESERVING_DUMP_CALIB_RECORD=1 uv run python main.py --n-trials 50 --results-dir results_ms_cl --objectives mse_log1p --structures multi_step --zero-handling two_stage  # claim-clustered variances (Table 7)
# simulation: 3 scenarios x 10 seeds; per-DY/single-model, multi-step, and the coverage experiment
uv run python run_simulation.py --scenarios S-A S-B S-C --seeds 10 --objectives mse_log1p poisson_rf --structures per_dy single_model --n-trials 10 --out results_sim --data-root data/sim
uv run python run_simulation.py --scenarios S-A S-B S-C --seeds 10 --objectives mse_log1p poisson_rf --structures multi_step --n-trials 10 --out results_sim_ms --data-root data/sim_ms
RESERVING_DUMP_CALIB_RECORD=1 uv run python run_simulation.py --scenarios S-A S-B S-C --seeds 10 --objectives mse_log1p --structures per_dy --n-trials 10 --fresh-eval --out results_sim_cov --data-root data/sim_cov
# summaries, tables and figures
uv run python tools/deposit_sim_summaries.py                 # results_v2/horizon_errors_real.csv, results_sim_ms/horizon_errors_sim.csv
uv run python tools/clustered_variance_ms.py                 # claim-clustered variances of the multi-step factors (Table 7)
uv run python tools/record_diagnostics.py                    # balance of the shrunk factors on the record
uv run python tools/coverage_sim.py --results results_sim_cov --data data/sim_cov          # Table 3
uv run python tools/fixed_predictor_coverage.py --data data/sim_cov --out results_sim_cov  # Table 4
uv run python tools/cohort_cl_by_type.py 500                 # stratified cohort chain ladder (Table 5 rows)
uv run python tools/make_tables.py                           # Tables 2-5 and 8-11 (reads the run directories, falls back to results/);
                                                             # Tables 6-7 need the per-cell records and are skipped otherwise
uv run python tools/ablation_tuned.py --results-dir results_v2  # Appendix C sensitivities (gate, interactions, depth; needs the persisted per-DY models)
uv run python tools/evidence_figures.py                      # Figures 1-3 -> results/figures/ (needs results_v2/partial, results_sim, results_sim_ms)
uv run python tools/hypothesis_exact_sim.py --reps 2000 --ns 600 3000 15000 --panels law A Aprime B C D --jobs 8 --out results_sim_hyp  # Table 1 (Section 5.1); analytic, ~3 min
```

### Command-to-output manifest

| Output in the paper | Command(s) | Deposited summary |
|---|---|---|
| Table 6 (per-DY factors, variances, credibility weights of the flagship) | `main.py --n-trials 50 ... --structures per_dy --objectives mse_log1p` with `RESERVING_DUMP_CALIB_RECORD=1`; `tools/make_tables.py` | `results/calibration_factors.csv` |
| Table 7 (per-horizon factors, claim-clustered variances) | `main.py ... --structures multi_step` into `results_ms_cl`; `tools/clustered_variance_ms.py`; `tools/make_tables.py` | `results/ms_cl/clustered_summary.csv` |
| Table 5 (correction arms of every configuration; chain-ladder rows) | full `main.py` matrix; `tools/cohort_cl.py`, `tools/cohort_cl_by_type.py 500`; `tools/make_tables.py` | `results/correction_comparison.csv`, `results/chain_ladder_cohort*.csv` |
| Table 2 (simulation scenarios) | `run_simulation.py` (per-DY/single-model and multi-step runs); `tools/make_tables.py` | `results/sim/summary_by_arm.csv`, `results/sim_ms/summary_by_arm.csv` |
| Table 8 (ten folds) | `main.py --n-folds 10 ...` into `results_v2_k10`; `tools/make_tables.py` | `results/k10/calibration_factors.csv` |
| Table 9 (no accident-year coordinates) | `RESERVING_EXCLUDE_AY_FEATURES=1 main.py --n-trials 25 ...`; `tools/make_tables.py` | `results/noay/*.csv` |
| Tables 3, 4 (coverage of the deployed and the fixed predictor) | `run_simulation.py ... --fresh-eval` into `results_sim_cov`; `tools/coverage_sim.py`; `tools/fixed_predictor_coverage.py` | `results/sim_cov/*.csv` |
| Table 10 (year-9 back-test) | `RESERVING_VALUATION_YEAR=9 main.py ...` into `results_v2_val9`; `tools/backtest_val9.py` | `results/val9/backtest_*.csv` |
| Table 11 (importance-weighted target factor) | `tools/transfer_iw.py` on the calibration record of `results_v2` | `results/transfer_iw.csv` |
| Table 1 (hypothesis-exact simulation) | `tools/hypothesis_exact_sim.py` (command above; `--table-only` regenerates the LaTeX from the CSVs) | `results/sim_hyp/*.csv`, `DIGEST.md` |
| Figures 1-3 | `tools/deposit_sim_summaries.py`; `tools/evidence_figures.py` | `results/horizon_errors_real.csv`, `results/sim_ms/horizon_errors_sim.csv` |
| Appendix C sensitivities (gate, interactions, depth) | `tools/ablation_tuned.py --results-dir results_v2` | `results/ablation_flagship_tuned.csv` |
| Remark 4.13 zero-prediction shares | `tools/zero_set_shares.py` (needs the per-cell partials) | - |

The flags `--n-trials`, `--n-folds`, `--results-dir`, `--methods`, `--structures`, `--objectives` and `--zero-handling` of
`main.py` select the configurations; `python main.py --help` lists them.

Data conventions (columns, accident/development-year indexing, the evaluation square) are those of the
illustrative dataset; see `data/convert_raw_data.py`.

## Deposited files

| File | Used for |
|---|---|
| `results/correction_comparison.csv`, `results/calibration_factors.csv`, `results/model_comparison.csv` | Table 5 (correction arms of every configuration), Table 6 (per-DY factors of the flagship) |
| `results/chain_ladder_cohort.csv`, `results/chain_ladder_cohort_by_type_min{500,100}.csv` | Table 5 chain-ladder rows (cohort-consistent; stratified by claim type) |
| `results/horizon_errors_real.csv`, `results/sim_ms/horizon_errors_sim.csv` | Figures 1 and 3 (per-horizon errors) |
| `results/sim/summary_by_arm.csv`, `results/sim_ms/summary_by_arm.csv` | Table 3 (simulation scenarios) |
| `results/k10/`, `results/noay/`, `results/full25/` | Tables 8, 9 and the same-budget control; `full25/record_diagnostics.csv` for Section 4.3 |
| `results/ms_cl/` | Table 2 (claim-clustered variances of the multi-step factors) |
| `results/sim_cov/` | Tables 3 and 4 (coverage experiments) |
| `results/keeppre/correction_comparison.csv` | Section 6.1: the flagship refit that retains the pre-report cells (uncorrected -44.5 %, heuristic +3.9 %, Bühlmann-Straub +63.1 %) |
| `results/sim_hyp/` | Table 1 and Section 5.1: the hypothesis-exact simulation (summaries per panel, the analytic-law check against the generator, `DIGEST.md`; `tools/hypothesis_exact_sim.py`) |
| `results/ablation_flagship_tuned.csv` | Section 6.1 (gate-threshold, interaction and depth sensitivities of the flagship on the tuned hyperparameters; `tools/ablation_tuned.py`) |

## License

See `LICENSE`.
