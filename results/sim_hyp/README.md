# results_sim_hyp -- simulation satisfying the theorems' hypotheses exactly

Produced by `tools/hypothesis_exact_sim.py` (see its module docstring for the design).

## Commands

```
<home> tools/hypothesis_exact_sim.py --reps 2000 --ns 600 3000 15000 --panels Aprime --out results_sim_hyp --jobs 8
```

Arguments: scenarios=['S-A', 'S-B'], reps=2000, ns=[600, 3000, 15000], seed=20260911, panels=['Aprime'], jobs=8

## Runtime

- panels A/A'/D (wall, parallel): 5.2 s
- total wall: 5.2 s

## Generator portfolios used for the law check


## Files

- `analytic_law_check.csv`: empirical vs analytic laws (z-scores; |z| > 4 flagged in the log) plus the moment-formula checks
- `panelA_summary.csv`: Theorem 4.6(i) with fixed predictors
- `panelAprime_summary.csv`: unequal deterministic fold sizes
- `panelB_exact.csv`, `panelB_sampled.csv`: pure development-year mix shift (Corollary 4.12(iii))
- `panelC_exact.csv`: within-DY shifts, Theorem 4.11(i),(iii)
- `panelD_summary.csv`: cross-fitted histogram learner, Theorem 4.6(iv)
- `panelA_by_replicate.csv`, `panelD_by_replicate.csv`: the per-replicate rows behind panels A and D. Written to the run directory, but not deposited with the released summaries (~52 MB); regenerate them with the command above.
- `run.log`: console log; `DIGEST.md`: plain-text digest
- LaTeX table: `paper/ime/tables/table_hypothesis_exact.tex`

## Headline numbers

- law check: 1440 z-scores, max |z| = 7.93, 2 with |z| > 4, 3 with |z| > 3; moment formulas vs quadrature max rel. err = 4.44e-16
- law check excluding sparse-event groups (< 5 expected active cells): 1406 z-scores, max |z| = 3.64, 0 with |z| > 4
- A S-A blocks f1: coverage of alpha_n by n = 600: 0.887, 3000: 0.930, 15000: 0.941; coverage of alpha_P = 0.887, 0.930, 0.941; SE/sd = 0.887, 0.964, 0.968
- A S-A blocks f2: coverage of alpha_n by n = 600: 0.914, 3000: 0.944, 15000: 0.944; coverage of alpha_P = 0.914, 0.944, 0.944; SE/sd = 0.906, 0.980, 0.981
- A S-A random f1: coverage of alpha_n by n = 600: 0.844, 3000: 0.910, 15000: 0.939; coverage of alpha_P = 0.844, 0.910, 0.939; SE/sd = 0.832, 0.928, 0.978
- A S-A random f2: coverage of alpha_n by n = 600: 0.869, 3000: 0.919, 15000: 0.942; coverage of alpha_P = 0.869, 0.919, 0.942; SE/sd = 0.755, 0.926, 0.976
- A S-B blocks f1: coverage of alpha_n by n = 600: 0.896, 3000: 0.928, 15000: 0.941; coverage of alpha_P = 0.906, 0.939, 0.943; SE/sd = 0.894, 0.949, 0.984
- A S-B blocks f2: coverage of alpha_n by n = 600: 0.912, 3000: 0.939, 15000: 0.947; coverage of alpha_P = 0.901, 0.925, 0.920; SE/sd = 0.906, 0.967, 0.989
- A S-B random f1: coverage of alpha_n by n = 600: 0.816, 3000: 0.900, 15000: 0.931; coverage of alpha_P = 0.816, 0.900, 0.931; SE/sd = 0.795, 0.923, 0.967
- A S-B random f2: coverage of alpha_n by n = 600: 0.848, 3000: 0.910, 15000: 0.939; coverage of alpha_P = 0.848, 0.910, 0.939; SE/sd = 0.723, 0.884, 0.967
- A' S-A data f1: coverage of alpha^w_n = 600: 0.891, 3000: 0.926, 15000: 0.947; with V^w SE = 0.959, 0.953, 0.954; sd sqrt(n)/sqrt(V^w) = 0.989, 1.005, 0.986
- A' S-A data f2: coverage of alpha^w_n = 600: 0.910, 3000: 0.935, 15000: 0.948; with V^w SE = 0.947, 0.949, 0.955; sd sqrt(n)/sqrt(V^w) = 1.045, 1.012, 0.994
- A' S-A proportional f1: coverage of alpha^w_n = 600: 0.887, 3000: 0.925, 15000: 0.940; with V^w SE = 0.958, 0.952, 0.950; sd sqrt(n)/sqrt(V^w) = 1.004, 0.997, 1.009
- A' S-A proportional f2: coverage of alpha^w_n = 600: 0.912, 3000: 0.936, 15000: 0.948; with V^w SE = 0.947, 0.949, 0.950; sd sqrt(n)/sqrt(V^w) = 1.065, 1.000, 1.009
- A' S-B data f1: coverage of alpha^w_n = 600: 0.884, 3000: 0.931, 15000: 0.942; with V^w SE = 0.956, 0.952, 0.951; sd sqrt(n)/sqrt(V^w) = 0.998, 1.001, 1.001
- A' S-B data f2: coverage of alpha^w_n = 600: 0.910, 3000: 0.933, 15000: 0.944; with V^w SE = 0.948, 0.949, 0.948; sd sqrt(n)/sqrt(V^w) = 1.059, 1.012, 1.002
- A' S-B proportional f1: coverage of alpha^w_n = 600: 0.887, 3000: 0.931, 15000: 0.939; with V^w SE = 0.959, 0.952, 0.949; sd sqrt(n)/sqrt(V^w) = 0.990, 1.001, 1.017
- A' S-B proportional f2: coverage of alpha^w_n = 600: 0.913, 3000: 0.938, 15000: 0.946; with V^w SE = 0.949, 0.945, 0.955; sd sqrt(n)/sqrt(V^w) = 1.027, 1.019, 1.000
- B S-A k=1..10: exact pooled rel. error +46.02% (identity residual 4.4e-16); sampled pooled/per-DY (vs population) by n = 600: +46.85%/-16.56%, 3000: +45.58%/-2.81%, 15000: +46.02%/-1.01%
- B S-A k=2..10: exact pooled rel. error +27.87% (identity residual 1.1e-16); sampled pooled/per-DY (vs population) by n = 600: +29.96%/-12.42%, 3000: +27.64%/-3.80%, 15000: +27.66%/-0.87%
- B S-B k=1..10: exact pooled rel. error +50.20% (identity residual 4.4e-16); sampled pooled/per-DY (vs population) by n = 600: +49.13%/-15.86%, 3000: +50.14%/-4.66%, 15000: +50.05%/-1.33%
- B S-B k=2..10: exact pooled rel. error +32.21% (identity residual 1.7e-16); sampled pooled/per-DY (vs population) by n = 600: +31.32%/-15.60%, 3000: +32.39%/-4.15%, 15000: +31.95%/-0.81%
- C: max identity residual 3.6e-15; chi bound / actual ranges 1.0--2414.3, TV bound / actual 1.3--8504.6
  - C S-A DY 3 f1 (own lower-triangle law): alpha_P=1.0076, alpha_Q=0.9350, |diff|=0.0726, chi bound=0.1733, TV bound=0.194, rel. err of P-factor +7.77%
  - C S-B DY 3 f1 (own lower-triangle law): alpha_P=1.0539, alpha_Q=1.1296, |diff|=0.0757, chi bound=0.2387, TV bound=0.283, rel. err of P-factor -6.70%
  - C S-A DY 5 f1 (own lower-triangle law): alpha_P=0.8570, alpha_Q=0.8005, |diff|=0.0565, chi bound=0.1383, TV bound=0.142, rel. err of P-factor +7.06%
  - C S-B DY 5 f1 (own lower-triangle law): alpha_P=0.8764, alpha_Q=0.9915, |diff|=0.1150, chi bound=0.2303, TV bound=0.224, rel. err of P-factor -11.60%
  - C S-A DY 7 f1 (own lower-triangle law): alpha_P=0.7354, alpha_Q=0.6912, |diff|=0.0442, chi bound=0.0953, TV bound=0.108, rel. err of P-factor +6.40%
  - C S-B DY 7 f1 (own lower-triangle law): alpha_P=0.6969, alpha_Q=0.8205, |diff|=0.1236, chi bound=0.2115, TV bound=0.281, rel. err of P-factor -15.07%
- D S-A blocks log: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.901/0.593/0.794, 3000: 0.932/0.662/0.879, 15000: 0.949/0.725/0.924; coverage of alpha_n with exact V = 0.938, 0.944, 0.951; mean sqrt(n) centring bias (relative to mbar_inf) = +11.241, +7.555, +4.458; SE/sd = 0.812, 0.929, 0.986; mean alpha_hat/alpha_inf = 0.7829, 0.9034, 0.9712
- D S-A blocks raw: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.884/1.000/0.885, 3000: 0.927/1.000/0.927, 15000: 0.944/1.000/0.945; coverage of alpha_n with exact V = 0.914, 0.942, 0.950; mean sqrt(n) centring bias (relative to mbar_inf) = -0.738, -0.352, -0.059; SE/sd = 0.859, 0.951, 0.991; mean alpha_hat/alpha_inf = 1.0458, 1.0058, 1.0003
- D S-A random log: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.863/0.430/0.719, 3000: 0.909/0.487/0.784, 15000: 0.941/0.590/0.865; coverage of alpha_n with exact V = 0.924, 0.945, 0.952; mean sqrt(n) centring bias (relative to mbar_inf) = +68.517, +45.902, +25.523; SE/sd = 0.355, 0.780, 0.954; mean alpha_hat/alpha_inf = 0.9590, 0.7411, 0.8784
- D S-A random raw: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.861/0.999/0.867, 3000: 0.911/1.000/0.910, 15000: 0.934/1.000/0.934; coverage of alpha_n with exact V = 0.852, 0.932, 0.944; mean sqrt(n) centring bias (relative to mbar_inf) = -2.303, -0.950, -0.221; SE/sd = 0.738, 0.920, 0.972; mean alpha_hat/alpha_inf = 1.1977, 1.0204, 1.0017
- D S-B blocks log: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.905/0.590/0.775, 3000: 0.936/0.652/0.824, 15000: 0.946/0.714/0.831; coverage of alpha_n with exact V = 0.941, 0.951, 0.952; mean sqrt(n) centring bias (relative to mbar_inf) = +9.896, +6.709, +4.054; SE/sd = 0.803, 0.952, 0.984; mean alpha_hat/alpha_inf = 0.8002, 0.9139, 0.9731
- D S-B blocks raw: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.888/0.997/0.882, 3000: 0.929/1.000/0.923, 15000: 0.941/1.000/0.933; coverage of alpha_n with exact V = 0.913, 0.945, 0.948; mean sqrt(n) centring bias (relative to mbar_inf) = -0.782, -0.327, -0.056; SE/sd = 0.860, 0.963, 0.975; mean alpha_hat/alpha_inf = 1.0485, 1.0066, 1.0003
- D S-B random log: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.857/0.457/0.728, 3000: 0.903/0.482/0.767, 15000: 0.935/0.572/0.842; coverage of alpha_n with exact V = 0.894, 0.947, 0.954; mean sqrt(n) centring bias (relative to mbar_inf) = +79.194, +72.434, +42.512; SE/sd = 0.387, 0.724, 0.923; mean alpha_hat/alpha_inf = 22.5983, 0.7297, 0.8443
- D S-B random raw: coverage alpha_n / alpha_inf / alpha_pop by n = 600: 0.850/0.998/0.856, 3000: 0.902/1.000/0.902, 15000: 0.936/1.000/0.936; coverage of alpha_n with exact V = 0.805, 0.921, 0.946; mean sqrt(n) centring bias (relative to mbar_inf) = -3.468, -1.482, -0.575; SE/sd = 0.630, 0.894, 0.971; mean alpha_hat/alpha_inf = 1.7968, 1.0366, 1.0035

## Panel A' rerun (11 Sep 2026, 17:35)

`python tools/hypothesis_exact_sim.py --reps 2000 --ns 600 3000 15000 --panels Aprime --out results_sim_hyp --jobs 8` with `W_DATA` set to the data's DY-2 accident-year-block fold shares 10,538/9,305/9,145 (0.364/0.321/0.315) instead of the earlier illustrative (0.37, 0.33, 0.30). Exact-V^w coverage 0.945-0.959, plug-in 0.884-0.913 -> 0.939-0.948; sd sqrt(n)/sqrt(V^w) = 0.99-1.07.
