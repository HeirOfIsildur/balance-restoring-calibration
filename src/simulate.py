"""
Synthetic individual-claims portfolios with a known data-generating process.

The simulator writes the same eight CSV files the pipeline reads from
``data/converted`` (claims_individual, transactions_individual and the six
run-off triangles), plus the ground truth the real data cannot provide: the
oracle conditional expectation of every unobserved cell.  It replaces the
SynthETIC-based R generator (data/generate_hostile_data.R) with a parametric
model whose assumptions coincide with Sections 1-2 of paper/theory.tex, so that
the calibration functional alpha_0 and the shifted functional alpha' are known
quantities rather than estimates.

Data-generating process (annual, I = 10 accident years, K = 10 development years)
---------------------------------------------------------------------------------
Claims.  For accident year a, N_a ~ Poisson(n_per_ay) claims.  Each claim has a
type c in {Material, Technical, Bodily, Annuity} drawn from a mix pi_a(c) that is
constant across accident years (scenarios S-A, S-C) or shifts toward the
long-tailed types (S-B: feature-observable mix shift), a notification delay
notidel ~ Exp(mean_c) years capped at 5, report year r = 1 + floor(notidel), and a
claim-level frailty U ~ N(0, tau^2) that makes payment history informative.

Payments.  For development year k >= r an activity indicator
A_k ~ Bernoulli(p_c(k)), p_c(k) = pi0_c exp(-lambda_c (k - r)) (1 + rho A_{k-1}),
capped at 0.98, and, given activity, an aggregate payment
    log(1 + Y_k) ~ N( mu_c + beta_c (k - r) + U,  s^2 ),
so the two-part log-normal benchmark of Prop. 2.4 holds exactly with
homoscedastic log-scale variance s^2.  Each active cell is split into one or
two transactions.  A calendar-year severity factor exp(gamma (a + k - 2))
multiplies payments in scenario S-C only: it is not a function of any feature,
so no feature-based learner can see it.

Truth.  For every RBNS claim (reported by year I) and every unobserved cell,
    oracle E[Y_k | c, r, U, A_{k-1}] = p_c(k) (exp(mu_c + beta_c (k-r) + U + s^2/2) - 1) x inflation.
Both the realized future payments and the oracle expectations are written, so
reserve errors can be measured against either.

Scenarios
---------
  S-A  homogeneous portfolio: constant mix, no inflation  (CL should tie or win)
  S-B  feature-observable mix shift toward Annuity/Bodily (RF should win)
  S-C  constant mix, 6 %/yr calendar-year inflation unseen by any feature
       (no feature-based method should win; Remark 3.11)

Conventions match data/converted: occurrence_period = accident_year,
payment_period = accident_year + dev_lag - 1 (calendar year 1..19),
observed_flag = payment_period <= I, ibnr_flag = reported after I, triangles
cumulative with NaN below the diagonal in the *_observed files.
"""

import json
import os

import numpy as np
from scipy.stats import norm
import pandas as pd

I_YEARS = 10
K_DY = 10

TYPES = ["Material", "Technical", "Bodily", "Annuity"]

# type-level parameters:  notidel mean (years), pi0, lambda, mu, beta
TYPE_PARAMS = {
    "Material":  dict(notidel_mean=0.10, pi0=0.90, lam=1.20, mu=7.5, beta=-0.50),
    "Technical": dict(notidel_mean=0.15, pi0=0.85, lam=1.00, mu=7.0, beta=-0.40),
    "Bodily":    dict(notidel_mean=0.60, pi0=0.70, lam=0.35, mu=9.5, beta=-0.15),
    "Annuity":   dict(notidel_mean=1.20, pi0=0.95, lam=0.12, mu=10.0, beta=0.05),
}

BASE_MIX = {"Material": 0.75, "Technical": 0.07, "Bodily": 0.15, "Annuity": 0.03}

SCENARIOS = {
    "S-A": dict(mix_shift=False, gamma=0.00, label="homogeneous, stable mix"),
    "S-B": dict(mix_shift=True,  gamma=0.00, label="feature-observable mix shift"),
    "S-C": dict(mix_shift=False, gamma=0.06, label="unobservable calendar-year inflation"),
}


def mix_for_year(a, mix_shift):
    """Claim-type mix in accident year a (1..I)."""
    if not mix_shift:
        return dict(BASE_MIX)
    t = (a - 1) / (I_YEARS - 1)  # 0 .. 1
    annuity = 0.01 + t * (0.08 - 0.01)
    bodily = 0.10 + t * (0.22 - 0.10)
    technical = 0.07
    material = 1.0 - annuity - bodily - technical
    return {"Material": material, "Technical": technical, "Bodily": bodily, "Annuity": annuity}


def activity_prob(c, k, r, prev_active, rho):
    p = TYPE_PARAMS[c]
    if k < r:
        return 0.0
    base = p["pi0"] * np.exp(-p["lam"] * (k - r))
    return float(min(0.98, base * (1.0 + rho * prev_active)))


def generate_dataset(scenario, seed, n_per_ay=2000, s=1.0, tau=0.5, rho=0.5, out_dir=None):
    """
    Simulate one portfolio and (optionally) write it in the pipeline's format.

    Returns a dict with the frames (claims, transactions, truth_claims,
    truth_cells, triangles) and a summary dict; when out_dir is given the CSVs
    and truth files are written there.
    """
    sc = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)

    claims, txns, truth_rows, oracle_cells = [], [], [], []
    oracle_obs_cells = []  # observed (upper-triangle) cells of reported claims, for the coverage experiment
    claim_no = 0
    for a in range(1, I_YEARS + 1):
        n_a = int(rng.poisson(n_per_ay))
        mix = mix_for_year(a, sc["mix_shift"])
        types = rng.choice(TYPES, size=n_a, p=[mix[t] for t in TYPES])
        for c in types:
            claim_no += 1
            p = TYPE_PARAMS[c]
            notidel = float(min(rng.exponential(p["notidel_mean"]), 5.0))
            r = 1 + int(np.floor(notidel))
            U = float(rng.normal(0.0, tau))
            reported_by_I = (a + r - 1) <= I_YEARS
            ibnr_flag = 0 if reported_by_I else 1

            prev = 0
            pmt_no = 0
            claim_total = 0.0
            cum_obs = 0.0
            realized_future = 0.0
            oracle_future = 0.0
            last_pay_k = None
            for k in range(1, K_DY + 1):
                cal = a + k - 1  # calendar year index
                infl = float(np.exp(sc["gamma"] * (cal - 1)))
                pk = activity_prob(c, k, r, prev, rho)
                m = p["mu"] + p["beta"] * (k - r) + U
                # E[max(e^Z - 1, 0)] for Z ~ N(m, s^2): the clipped log-normal mean (matches the clipping of y below)
                clipped_mean = (np.exp(m + 0.5 * s * s) * norm.cdf((m + s * s) / s) - norm.cdf(m / s)) if s > 0 else max(np.exp(m) - 1.0, 0.0)
                oracle = pk * clipped_mean * infl if pk > 0 else 0.0
                active = rng.random() < pk
                y = 0.0
                if active:
                    y = max(float((np.exp(rng.normal(m, s)) - 1.0) * infl), 0.0)
                    last_pay_k = k
                    if rng.random() < 0.3:
                        u = float(rng.uniform(0.2, 0.8))
                        shares = [u, 1.0 - u]
                    else:
                        shares = [1.0]
                    for sh in shares:
                        pmt_no += 1
                        amt = y * sh
                        txns.append((claim_no, pmt_no, a, a, k, cal, 0.0, amt, amt, int(cal <= I_YEARS)))
                claim_total += y
                if cal <= I_YEARS:
                    cum_obs += y
                    if reported_by_I:
                        oracle_obs_cells.append((claim_no, a, k, c, oracle, y))
                elif reported_by_I:
                    realized_future += y
                    oracle_future += oracle
                    oracle_cells.append((claim_no, a, k, c, oracle, y))
                prev = int(active)

            setldel = float((last_pay_k - r) if last_pay_k is not None else 0)
            claims.append(
                dict(
                    claim_no=claim_no, accident_year=a, occurrence_period=a,
                    claim_size=claim_total, notidel=notidel, setldel=setldel, report_dy=r,
                    no_payments=pmt_no, cum_paid_observed=cum_obs, cum_paid_full=claim_total,
                    reserve_true=claim_total - cum_obs, ibnr_flag=ibnr_flag, claim_type=c,
                )
            )
            if reported_by_I:
                truth_rows.append(
                    dict(claim_no=claim_no, accident_year=a, claim_type=c, frailty=U,
                         oracle_reserve=oracle_future, realized_reserve=realized_future)
                )

    claims_df = pd.DataFrame(
        claims,
        columns=["claim_no", "accident_year", "occurrence_period", "claim_size", "notidel", "setldel", "report_dy",
                 "no_payments", "cum_paid_observed", "cum_paid_full", "reserve_true", "ibnr_flag", "claim_type"],
    )  # explicit columns so that an empty portfolio (every Poisson count zero) still has the schema
    txn_df = pd.DataFrame(
        txns,
        columns=["claim_no", "pmt_no", "accident_year", "occurrence_period", "dev_lag",
                 "payment_period", "claim_size", "payment_size", "payment_inflated", "observed_flag"],
    )
    size_map = claims_df.set_index("claim_no")["claim_size"]
    txn_df["claim_size"] = txn_df["claim_no"].map(size_map).values
    truth_df = pd.DataFrame(
        truth_rows, columns=["claim_no", "accident_year", "claim_type", "frailty", "oracle_reserve", "realized_reserve"]
    )
    oracle_df = pd.DataFrame(
        oracle_cells,
        columns=["claim_no", "accident_year", "dev_lag", "claim_type", "oracle_payment", "realized_payment"],
    )
    oracle_obs_df = pd.DataFrame(
        oracle_obs_cells,
        columns=["claim_no", "accident_year", "dev_lag", "claim_type", "oracle_payment", "realized_payment"],
    )

    triangles = build_triangles(claims_df, txn_df)

    summary = dict(
        scenario=scenario, label=sc["label"], seed=seed, n_per_ay=n_per_ay, s=s, tau=tau, rho=rho,
        gamma=sc["gamma"], n_claims=int(len(claims_df)), n_rbns=int((claims_df.ibnr_flag == 0).sum()),
        n_transactions=int(len(txn_df)),
        realized_rbns_reserve=float(truth_df["realized_reserve"].sum()) if len(truth_df) else 0.0,
        oracle_rbns_reserve=float(truth_df["oracle_reserve"].sum()) if len(truth_df) else 0.0,
        shift_factor_exact=float(np.exp(0.5 * s * s)),
        type_mix_by_ay={a: mix_for_year(a, sc["mix_shift"]) for a in range(1, I_YEARS + 1)},
    )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        claims_df.to_csv(os.path.join(out_dir, "claims_individual.csv"), index=False)
        txn_df.to_csv(os.path.join(out_dir, "transactions_individual.csv"), index=False)
        for name, tri in triangles.items():
            tri.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        truth_df.to_csv(os.path.join(out_dir, "truth_claims.csv"), index=False)
        oracle_df.to_csv(os.path.join(out_dir, "truth_cells.csv"), index=False)
        oracle_obs_df.to_csv(os.path.join(out_dir, "truth_cells_observed.csv"), index=False)
        with open(os.path.join(out_dir, "truth.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    return dict(claims=claims_df, transactions=txn_df, truth_claims=truth_df, truth_cells=oracle_df,
                truth_cells_observed=oracle_obs_df,
                triangles=triangles, summary=summary)


def build_triangles(claims_df, txn_df):
    """Cumulative / incremental AY x DY triangles for all claims and for RBNS claims."""
    out = {}
    for suffix, sub in (("", claims_df), ("_rbns", claims_df[claims_df.ibnr_flag == 0])):
        keep = txn_df[txn_df.claim_no.isin(sub.claim_no) & (txn_df.dev_lag <= K_DY)]
        inc = np.zeros((I_YEARS, K_DY))
        if len(keep):
            g = keep.groupby(["accident_year", "dev_lag"])["payment_size"].sum()
            for (a, k), v in g.items():
                inc[a - 1, k - 1] = v
        cum = np.cumsum(inc, axis=1)
        obs = cum.copy()
        for a in range(1, I_YEARS + 1):
            for k in range(1, K_DY + 1):
                if a + k - 1 > I_YEARS:
                    obs[a - 1, k - 1] = np.nan
        cols = [f"DY{k}" for k in range(1, K_DY + 1)]

        def _frame(mat):
            df = pd.DataFrame(mat, columns=cols)
            df.insert(0, "accident_year", range(1, I_YEARS + 1))
            return df

        out[f"triangle_observed{suffix}"] = _frame(obs)
        out[f"triangle_full{suffix}"] = _frame(cum)
        out[f"triangle_incremental{suffix}"] = _frame(inc)
    return out
