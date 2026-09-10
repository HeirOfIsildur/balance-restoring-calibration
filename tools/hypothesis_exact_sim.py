"""
Controlled simulation in which the hypotheses of Paper B's theorems hold EXACTLY (referee request, 11 Sep 2026).

Why this experiment exists
--------------------------
The portfolio simulations of Section 5 use fitted random forests and realised portfolios, so the theorems'
hypotheses (i.i.d. cells from a known law, a predictor that is a fixed function or an L^2-consistent learner,
a regression function invariant across accident years) hold only approximately and the population targets carry
Monte Carlo error.  Here the cells are sampled i.i.d. from population laws that are derived in closed form from
the generator of src/simulate.py, and every population quantity a theorem refers to (targets, variances,
predictor-weighted laws, density ratios, bounds) is computed exactly on the finite support, so the references
carry no Monte Carlo error at all.

The cell model (derived from src/simulate.py with s = 1, tau = 0.5, rho = 0.5, gamma = 0)
------------------------------------------------------------------------------------------
A claim has type c with TYPE_PARAMS[c] = (notidel_mean, pi0, lam, mu, beta); its reporting development year is
r = 1 + floor(min(Exp(notidel_mean), 5)), so P(r = j) = e^{-(j-1)/m} - e^{-j/m} for j = 1..5 and P(r = 6) =
e^{-5/m}.  In development year k >= r the claim is active with probability
p(c, k, r, prev) = min(0.98, pi0 e^{-lam (k-r)} (1 + rho prev)), prev being the previous year's activity
indicator (0 at k = r); given activity, Y = max(e^Z - 1, 0) with Z ~ N(mu + beta (k-r) + U, s^2) and a claim
frailty U ~ N(0, tau^2) independent of everything else.  The reduced covariate is the DISCRETE vector
x = (c, k, r, prev).  Because U is Gaussian and independent of the activity chain, integrating it out gives
Z | x ~ N(m0, sigma^2), m0 = mu + beta (k-r), sigma^2 = s^2 + tau^2 = 1.25, and the closed forms

    g(x)            = E[Y | x]          = p CM(m0, sigma),   CM(m, s) = e^{m+s^2/2} Phi((m+s^2)/s) - Phi(m/s),
    E[Y^2 | x]                          = p [e^{2m0+2sigma^2} Phi((m0+2sigma^2)/sigma)
                                            - 2 e^{m0+sigma^2/2} Phi((m0+sigma^2)/sigma) + Phi(m0/sigma)],
    E[log(1+Y) | x]                     = p [m0 Phi(m0/sigma) + sigma phi(m0/sigma)],
    P(Y > 0 | x)                        = p Phi(m0/sigma).

Cells are sampled as i.i.d. pairs (x, Y): each sampled cell carries its own fresh frailty, so E[Y | x] = g(x)
holds exactly (Assumption A1), all moments are finite (A2), and g does not depend on the accident year (A3,
gamma = 0).  The law of prev at DY k given (c, r) is the forward recursion of the activity chain; the law of
(c, r) for an observed post-report cell of accident year a at DY k is proportional to
mix_a(c) P(r | c) 1{r <= min(k, I - a + 1)} (reported by the valuation date I = 10, k >= r); accident years are
weighted within a development year by their expected number of eligible cells.  The upper-triangle law of DY k
is P_k (accident years a <= I-k+1), the lower-triangle law is Q_k (a >= I-k+2), whose stricter reporting
truncation r <= I-a+1 is a genuine within-DY covariate shift; in S-B the linear type-mix drift adds a second.
S-C (calendar inflation) is omitted: there g depends on a + k - 1 and Assumption A3 fails by construction.

Panels
------
  A   Fixed positive predictors f1(x) = e^{mu_c+beta_c(k-r)} - 1 and f2(x) = exp(E[log(1+Y)|x]) - 1,
      Theorem 4.6(i): coverage of the finite-sample target alpha_n with the delta-method plug-in SE and with the
      exact V_n; random folds (P_j = P_k) and accident-year blocks {1,2,3},{4,5,6},{7,..} (P_j = P_k(.|a in A_j)).
  A'  Unequal deterministic fold sizes (the data's DY-2 shares 0.364/0.321/0.315 and shares proportional to the exact block
      probabilities pi_j): target alpha^w_n = sum_j w_j mu_{Y,j} / sum_j w_j m_j and variance
      V^w = sum_j w_j Var_{P_j}(Y - alpha^w f) / (sum_j w_j m_j)^2; with w = pi the target equals alpha_P exactly.
  B   Pure development-year mix shift, Corollary 4.12(iii) / Proposition 2.3(ii): P = sum_k Pbar(k) P_k
      (upper-triangle DY weights) versus Q_b = sum_k Qbar(k) P_k (lower-triangle DY weights, same conditional
      laws); exact identity alpha_Q - alpha_P = sum_k (Qtilde(k) - Ptilde(k)) alpha_{P,k} and a sampled check.
  C   Controlled within-group shift at fixed k, Theorem 4.11(i),(iii): the generator's own lower-triangle law
      Q_k, a parametric type-mix shift Q_eps and a reporting-delay shift; exact alpha_Q - alpha_P =
      Cov_{Ptilde}(rho, g/f), the chi^2 and total-variation bounds.
  D   Cross-fitted learner satisfying Assumption A4 exactly (cell-mean "histogram" estimator on the finite
      support, log target with f_inf = f2 and raw target with f_inf = g), Theorem 4.6(iv): coverage at the
      cross-fitted target alpha_n (exact conditional fold means m_hat_j = sum_x P_j(x) f_hat^{(-j)}(x)), at
      alpha_inf, and at the deployed-predictor target alpha^pop = E_{P_k} Y / E_{P_k} f_hat_full.

Usage
-----
    python tools/hypothesis_exact_sim.py --quick                       # smoke test
    python tools/hypothesis_exact_sim.py --reps 2000 --ns 600 3000 15000 --panels law A Aprime B C D
    python tools/hypothesis_exact_sim.py --verify-d                    # print one panel-D replicate by hand

Outputs go to results_sim_hyp/ (CSV per panel, README.md) and paper/ime/tables/table_hypothesis_exact.tex.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.simulate import (BASE_MIX, I_YEARS, K_DY, SCENARIOS, TYPE_PARAMS, TYPES,  # noqa: E402
                          activity_prob, generate_dataset, mix_for_year)

S_LOG, TAU, RHO = 1.0, 0.5, 0.5          # generator defaults
SIGMA = float(np.sqrt(S_LOG ** 2 + TAU ** 2))
R_MAX = 6                                # reporting DY 1..6 (delay capped at 5 years)
J = 3
Z975 = float(norm.ppf(0.975))
SCEN = ["S-A", "S-B"]
AY_BLOCKS = [(1, 2, 3), (4, 5, 6), tuple(range(7, I_YEARS + 1))]
W_DATA = (10538 / 28988, 9305 / 28988, 9145 / 28988)   # the data's DY-2 accident-year-block fold shares 10,538 / 9,305 / 9,145 (0.364 / 0.321 / 0.315)
PANEL_CODE = {"A": 1, "Aprime": 2, "B": 3, "C": 4, "D": 5, "law": 6}
SC_CODE = {"S-A": 1, "S-B": 2}
DESIGN_CODE = {"random": 1, "blocks": 2, "data": 3, "proportional": 4}


# ----------------------------------------------------------------------------------------------------------------
# closed-form conditional moments
# ----------------------------------------------------------------------------------------------------------------
def clipped_mean(m, sig):
    """E[max(e^Z - 1, 0)], Z ~ N(m, sig^2)."""
    return np.exp(m + 0.5 * sig ** 2) * norm.cdf((m + sig ** 2) / sig) - norm.cdf(m / sig)


def clipped_second(m, sig):
    """E[max(e^Z - 1, 0)^2], Z ~ N(m, sig^2)."""
    return (np.exp(2 * m + 2 * sig ** 2) * norm.cdf((m + 2 * sig ** 2) / sig)
            - 2 * np.exp(m + 0.5 * sig ** 2) * norm.cdf((m + sig ** 2) / sig) + norm.cdf(m / sig))


def pos_part_mean(m, sig):
    """E[max(Z, 0)] = E[log(1 + max(e^Z - 1, 0))], Z ~ N(m, sig^2)."""
    return m * norm.cdf(m / sig) + sig * norm.pdf(m / sig)


def verify_moment_formulas(seed, n_mc=10_000_000):
    """Check the three closed forms against adaptive quadrature and a 1e7-sample Monte Carlo draw."""
    rng = np.random.default_rng([seed, 99])
    rows = []
    for m in (2.0, 3.0, 7.0, 10.5):
        Z = rng.normal(m, SIGMA, n_mc)
        W = np.maximum(np.exp(Z) - 1.0, 0.0)
        for name, exact, samp, integrand in (
            ("clipped_mean", clipped_mean(m, SIGMA), W, lambda z: max(np.exp(z) - 1, 0) * norm.pdf(z, m, SIGMA)),
            ("clipped_second", clipped_second(m, SIGMA), W ** 2, lambda z: max(np.exp(z) - 1, 0) ** 2 * norm.pdf(z, m, SIGMA)),
            ("pos_part_mean", pos_part_mean(m, SIGMA), np.log1p(W), lambda z: max(z, 0) * norm.pdf(z, m, SIGMA)),
        ):
            quad, _ = integrate.quad(integrand, 0.0, m + 12 * SIGMA, limit=400)
            mc, se = samp.mean(), samp.std(ddof=1) / np.sqrt(n_mc)
            rows.append(dict(kind="moment_formula", scenario="-", dy=np.nan, group=f"{name} m0={m}", n=n_mc,
                             empirical=mc, analytic=exact, se=se, z=(mc - exact) / se,
                             quad_rel_err=quad / exact - 1.0))
    return rows


# ----------------------------------------------------------------------------------------------------------------
# reporting law, activity chain, finite support
# ----------------------------------------------------------------------------------------------------------------
def report_pmf(c, delay_scale=1.0):
    """P(r = j | c), j = 1..R_MAX, for delay ~ Exp(mean) capped at 5, r = 1 + floor(delay)."""
    md = TYPE_PARAMS[c]["notidel_mean"] * delay_scale
    j = np.arange(1, R_MAX + 1, dtype=float)
    p = np.exp(-(j - 1) / md) - np.exp(-j / md)
    p[-1] = np.exp(-(R_MAX - 1) / md)
    return p


def prev_prob(c, r, k):
    """P(active in DY k-1 | c, r) for k > r (prev = 0 at k = r): forward recursion of the activity chain."""
    q1 = 0.0
    for kk in range(r, k):
        q1 = (1.0 - q1) * activity_prob(c, kk, r, 0, RHO) + q1 * activity_prob(c, kk, r, 1, RHO)
    return q1


class Support:
    """All reduced covariate points x = (k, c, r, prev), k = 1..K, r <= min(k, 6), with their exact conditional moments."""

    def __init__(self):
        rows = [(k, ci, r, prev) for k in range(1, K_DY + 1) for ci in range(len(TYPES))
                for r in range(1, min(k, R_MAX) + 1) for prev in (0, 1)]
        arr = np.array(rows, dtype=int)
        self.k, self.ci, self.r, self.prev = (arr[:, i] for i in range(4))
        self.n = len(rows)
        self.index = {row: i for i, row in enumerate(rows)}
        mu = np.array([TYPE_PARAMS[c]["mu"] for c in TYPES])[self.ci]
        beta = np.array([TYPE_PARAMS[c]["beta"] for c in TYPES])[self.ci]
        self.m0 = mu + beta * (self.k - self.r)
        self.p_act = np.array([activity_prob(TYPES[ci], k, r, prev, RHO) for k, ci, r, prev in rows])
        self.g = self.p_act * clipped_mean(self.m0, SIGMA)              # E[Y | x]
        self.EY2 = self.p_act * clipped_second(self.m0, SIGMA)          # E[Y^2 | x]
        self.Elog = self.p_act * pos_part_mean(self.m0, SIGMA)          # E[log(1+Y) | x]
        self.ppos = self.p_act * norm.cdf(self.m0 / SIGMA)              # P(Y > 0 | x)
        self.f1 = np.exp(self.m0) - 1.0                                 # type severity curve
        self.f2 = np.exp(self.Elog) - 1.0                               # population limit of a log-target learner
        self.prev1 = np.array([prev_prob(TYPES[ci], r, k) for k, ci, r, prev in rows])  # P(prev = 1 | c, r, k)
        self.predictors = {"f1": self.f1, "f2": self.f2}

    def frame(self):
        return pd.DataFrame(dict(k=self.k, type=[TYPES[c] for c in self.ci], r=self.r, prev=self.prev, p_act=self.p_act,
                                 m0=self.m0, g=self.g, EY2=self.EY2, Elog=self.Elog, f1=self.f1, f2=self.f2))


_SUP = None


def support():
    global _SUP
    if _SUP is None:
        _SUP = Support()
    return _SUP


class Law:
    """A probability vector on the finite support with exact expectations and i.i.d. sampling of (x, Y)."""

    def __init__(self, prob, name=""):
        prob = np.asarray(prob, dtype=float)
        self.p = prob / prob.sum()
        self.nz = np.flatnonzero(self.p > 0)
        self.p_nz = self.p[self.nz] / self.p[self.nz].sum()
        self.name = name
        self.sup = support()

    def E(self, h):
        return float(self.p @ h)

    def mean_Y(self):
        return self.E(self.sup.g)

    def var_resid(self, alpha, f):
        """Var_P(Y - alpha f(X)) exactly."""
        s = self.sup
        second = self.E(s.EY2) - 2 * alpha * self.E(s.g * f) + alpha ** 2 * self.E(f * f)
        return second - (self.E(s.g) - alpha * self.E(f)) ** 2

    def sample(self, rng, n):
        """n i.i.d. cells: draw x from the law, then U, activity and Y exactly as the generator does."""
        s = self.sup
        idx = self.nz[rng.choice(len(self.nz), size=n, p=self.p_nz)]
        U = rng.normal(0.0, TAU, n)
        active = rng.random(n) < s.p_act[idx]
        Z = s.m0[idx] + U + S_LOG * rng.standard_normal(n)
        Y = np.where(active, np.maximum(np.exp(Z) - 1.0, 0.0), 0.0)
        return idx, Y


def cr_weights(mix, k, a, delay_scale=1.0):
    """Unnormalised weights of (c, r) for a post-report cell of accident year a at DY k whose claim is reported by I."""
    rmax = min(k, I_YEARS - a + 1, R_MAX)
    W = np.zeros((len(TYPES), R_MAX))
    if rmax >= 1:
        for ci, c in enumerate(TYPES):
            W[ci, :rmax] = mix[c] * report_pmf(c, delay_scale)[:rmax]
    return W


def dy_law(scenario, k, ays, mix_override=None, delay_scale=1.0):
    """Law of x at DY k for the cells of accident years `ays` (AY-weighted by expected eligible counts).

    Returns (Law, total) where total = sum of the unnormalised weights = expected eligible cells per claim of one AY."""
    s = support()
    W = np.zeros((len(TYPES), R_MAX))
    for a in ays:
        mix = mix_override if mix_override is not None else mix_for_year(a, SCENARIOS[scenario]["mix_shift"])
        W += cr_weights(mix, k, a, delay_scale)
    prob = np.zeros(s.n)
    for ci in range(len(TYPES)):
        for r in range(1, min(k, R_MAX) + 1):
            w = W[ci, r - 1]
            if w > 0:
                q1 = prev_prob(TYPES[ci], r, k)
                prob[s.index[(k, ci, r, 1)]] = w * q1
                prob[s.index[(k, ci, r, 0)]] = w * (1.0 - q1)
    return Law(prob, f"{scenario} k={k} a={list(ays)}"), float(W.sum())


def upper_ays(k):
    return list(range(1, I_YEARS - k + 2))


def lower_ays(k):
    return list(range(I_YEARS - k + 2, I_YEARS + 1))


def training_law(scenario, k):
    return dy_law(scenario, k, upper_ays(k))[0]


def target_law(scenario, k):
    return dy_law(scenario, k, lower_ays(k))[0]


def block_laws(scenario, k):
    """Fold laws P_j = P_k(. | a in A_j) restricted to the eligible AYs, with exact block probabilities pi_j; None if a block is empty."""
    laws, totals = [], []
    elig = set(upper_ays(k))
    for A in AY_BLOCKS:
        ays = [a for a in A if a in elig]
        if not ays:
            return None, None
        law, tot = dy_law(scenario, k, ays)
        laws.append(law)
        totals.append(tot)
    pi = np.array(totals) / np.sum(totals)
    return laws, pi


def fold_laws(scenario, k, design):
    if design == "random":
        P = training_law(scenario, k)
        return [P] * J, np.full(J, 1.0 / J)
    return block_laws(scenario, k)


# ----------------------------------------------------------------------------------------------------------------
# exact population quantities for a fixed predictor and fold laws
# ----------------------------------------------------------------------------------------------------------------
def stratified_target(laws, f, w=None):
    """alpha^w_n = sum_j w_j mu_{Y,j} / sum_j w_j m_j and V^w = sum_j w_j Var_{P_j}(Y - alpha f) / (sum_j w_j m_j)^2."""
    w = np.full(len(laws), 1.0 / len(laws)) if w is None else np.asarray(w, dtype=float)
    mu = np.array([L.mean_Y() for L in laws])
    m = np.array([L.E(f) for L in laws])
    alpha = float(w @ mu) / float(w @ m)
    V = float(np.sum(w * np.array([L.var_resid(alpha, f) for L in laws]))) / float(w @ m) ** 2
    return alpha, V, mu, m


def ratio_and_se(Y, F):
    """Plain ratio and the delta-method plug-in SE with the pooled residual variance (as in the paper / tools/fixed_predictor_coverage.py)."""
    ah = Y.sum() / F.sum()
    se = np.sqrt(np.var(Y - ah * F, ddof=1) / (len(Y) * F.mean() ** 2))
    return ah, se


def sample_folds(laws, sizes, rng):
    idx, Y, fold = [], [], []
    for j, (L, nj) in enumerate(zip(laws, sizes)):
        i, y = L.sample(rng, nj)
        idx.append(i)
        Y.append(y)
        fold.append(np.full(nj, j))
    return np.concatenate(idx), np.concatenate(Y), np.concatenate(fold)


def fold_sizes(n, w):
    sizes = np.round(np.asarray(w) * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()
    return sizes


# ----------------------------------------------------------------------------------------------------------------
# Panel A / A'
# ----------------------------------------------------------------------------------------------------------------
def run_panel_A(job):
    """One (scenario, k, design) configuration of panel A over all n and R replications; both predictors on the same samples."""
    scenario, k, design, ns, R, seed = job["scenario"], job["k"], job["design"], job["ns"], job["reps"], job["seed"]
    s = support()
    laws, pi = fold_laws(scenario, k, design)
    if laws is None:
        return [], []
    P = training_law(scenario, k)
    exact = {}
    for pname, f in s.predictors.items():
        alpha_n, V_n, mu, m = stratified_target(laws, f)
        exact[pname] = dict(alpha_n=alpha_n, V_n=V_n, alpha_P=P.mean_Y() / P.E(f), mu=mu, m=m)
    summ, reps = [], []
    for n in ns:
        rng = np.random.default_rng([seed, PANEL_CODE["A"], SC_CODE[scenario], k, DESIGN_CODE[design], n])
        sizes = np.full(J, n // J)
        out = {p: np.zeros((R, 2)) for p in s.predictors}
        for rep in range(R):
            idx, Y, _ = sample_folds(laws, sizes, rng)
            for pname, f in s.predictors.items():
                out[pname][rep] = ratio_and_se(Y, f[idx])
        for pname in s.predictors:
            ah, se = out[pname][:, 0], out[pname][:, 1]
            e = exact[pname]
            se_V = np.sqrt(e["V_n"] / n)
            cov_n = np.abs(ah - e["alpha_n"]) <= Z975 * se
            cov_nV = np.abs(ah - e["alpha_n"]) <= Z975 * se_V
            cov_P = np.abs(ah - e["alpha_P"]) <= Z975 * se
            summ.append(dict(scenario=scenario, dy=k, design=design, predictor=pname, n=n, reps=R, n_all_zero_Y=int((ah == 0).sum()),
                             alpha_n=e["alpha_n"], alpha_P=e["alpha_P"], V_n=e["V_n"],
                             mean_alpha_hat=ah.mean(), sd_alpha_hat=ah.std(ddof=1),
                             coverage_alpha_n=cov_n.mean(), coverage_alpha_n_Vn=cov_nV.mean(), coverage_alpha_P=cov_P.mean(),
                             mean_se_plugin=se.mean(), se_Vn=se_V, se_over_sd=se.mean() / ah.std(ddof=1),
                             sd_sqrtn_over_sqrtVn=np.sqrt(n) * ah.std(ddof=1) / np.sqrt(e["V_n"]),
                             bias_rel=(ah.mean() - e["alpha_n"]) / e["alpha_n"],
                             misses_below=int(((ah - e["alpha_n"]) < -Z975 * se).sum()),
                             misses_above=int(((ah - e["alpha_n"]) > Z975 * se).sum()),
                             mu_Y_folds=";".join(f"{x:.6g}" for x in e["mu"]), m_folds=";".join(f"{x:.6g}" for x in e["m"])))
            for rep in range(R):
                reps.append(dict(scenario=scenario, dy=k, design=design, predictor=pname, n=n, rep=rep,
                                 alpha_hat=ah[rep], se_plugin=se[rep]))
    return summ, reps


def run_panel_Aprime(job):
    """Unequal deterministic fold sizes on the accident-year blocks: shares W_DATA and shares proportional to pi_j."""
    scenario, k, weighting, ns, R, seed = job["scenario"], job["k"], job["design"], job["ns"], job["reps"], job["seed"]
    s = support()
    laws, pi = block_laws(scenario, k)
    if laws is None:
        return [], []
    P = training_law(scenario, k)
    w_nom = np.array(W_DATA) if weighting == "data" else pi
    summ = []
    for n in ns:
        sizes = fold_sizes(n, w_nom)
        w_real = sizes / n
        rng = np.random.default_rng([seed, PANEL_CODE["Aprime"], SC_CODE[scenario], k, DESIGN_CODE[weighting], n])
        exact = {}
        for pname, f in s.predictors.items():
            alpha_w, V_w, mu, m = stratified_target(laws, f, w_real)
            alpha_w_nom, _, _, _ = stratified_target(laws, f, w_nom)
            exact[pname] = dict(alpha_w=alpha_w, V_w=V_w, alpha_w_nominal=alpha_w_nom, alpha_P=P.mean_Y() / P.E(f),
                                alpha_equal=stratified_target(laws, f)[0])
        out = {p: np.zeros((R, 2)) for p in s.predictors}
        for rep in range(R):
            idx, Y, _ = sample_folds(laws, sizes, rng)
            for pname, f in s.predictors.items():
                out[pname][rep] = ratio_and_se(Y, f[idx])
        for pname in s.predictors:
            ah, se = out[pname][:, 0], out[pname][:, 1]
            e = exact[pname]
            se_V = np.sqrt(e["V_w"] / n)
            summ.append(dict(scenario=scenario, dy=k, weighting=weighting, predictor=pname, n=n, reps=R,
                             n_folds=";".join(str(x) for x in sizes), w_realised=";".join(f"{x:.6f}" for x in w_real),
                             w_nominal=";".join(f"{x:.6f}" for x in w_nom),
                             alpha_w=e["alpha_w"], alpha_w_nominal=e["alpha_w_nominal"], alpha_P=e["alpha_P"],
                             alpha_equal_weight=e["alpha_equal"], V_w=e["V_w"],
                             mean_alpha_hat=ah.mean(), sd_alpha_hat=ah.std(ddof=1),
                             coverage_alpha_w=(np.abs(ah - e["alpha_w"]) <= Z975 * se).mean(),
                             coverage_alpha_w_Vw=(np.abs(ah - e["alpha_w"]) <= Z975 * se_V).mean(),
                             coverage_alpha_P=(np.abs(ah - e["alpha_P"]) <= Z975 * se).mean(),
                             coverage_alpha_equal=(np.abs(ah - e["alpha_equal"]) <= Z975 * se).mean(),
                             mean_se_plugin=se.mean(), se_Vw=se_V, se_over_sd=se.mean() / ah.std(ddof=1),
                             sd_sqrtn_over_sqrtVw=np.sqrt(n) * ah.std(ddof=1) / np.sqrt(e["V_w"])))
    return summ, []


# ----------------------------------------------------------------------------------------------------------------
# Panel B: pure development-year mix shift
# ----------------------------------------------------------------------------------------------------------------
def pooled_laws(scenario, ks):
    """P = sum_k Pbar(k) P_k with upper-triangle DY weights and Q_b = sum_k Qbar(k) P_k with lower-triangle DY weights."""
    s = support()
    Pk, wP, wQ = {}, {}, {}
    for k in ks:
        Pk[k], wP[k] = dy_law(scenario, k, upper_ays(k))
        wQ[k] = dy_law(scenario, k, lower_ays(k))[1] if lower_ays(k) else 0.0
    Pbar = np.array([wP[k] for k in ks]) / sum(wP.values())
    Qbar = np.array([wQ[k] for k in ks]) / sum(wQ.values())
    P = Law(sum(Pbar[i] * Pk[k].p for i, k in enumerate(ks)), "pooled P")
    Q = Law(sum(Qbar[i] * Pk[k].p for i, k in enumerate(ks)), "pooled Q_b")
    return Pk, Pbar, Qbar, P, Q


def run_panel_B(scenario, ns, R, seed, pname="f1"):
    s = support()
    f = s.predictors[pname]
    exact_rows, sampled_rows = [], []
    for variant, ks in (("k=1..10", list(range(1, K_DY + 1))), ("k=2..10", list(range(2, K_DY + 1)))):
        Pk, Pbar, Qbar, P, Q = pooled_laws(scenario, ks)
        alpha_P, alpha_Q = P.mean_Y() / P.E(f), Q.mean_Y() / Q.E(f)
        EfP, EfQ = P.E(f), Q.E(f)
        Pt = np.array([Pbar[i] * Pk[k].E(f) for i, k in enumerate(ks)]) / EfP          # predictor-weighted DY mixes
        Qt = np.array([Qbar[i] * Pk[k].E(f) for i, k in enumerate(ks)]) / EfQ
        aPk = np.array([Pk[k].mean_Y() / Pk[k].E(f) for k in ks])
        between = float(np.sum((Qt - Pt) * aPk))
        rel_err_exact = -between / alpha_Q
        # Corollary 4.12(i) in population: the per-DY P-factors balance the Q-aggregate exactly under a pure mix shift
        perdy_agg = float(np.sum(aPk * Qbar * np.array([Pk[k].E(f) for k in ks])))
        perdy_rel_err_exact = perdy_agg / Q.mean_Y() - 1.0
        for i, k in enumerate(ks):
            exact_rows.append(dict(scenario=scenario, variant=variant, predictor=pname, dy=k, Pbar=Pbar[i], Qbar=Qbar[i],
                                   Ptilde=Pt[i], Qtilde=Qt[i], alpha_Pk=aPk[i], EY_k=Pk[k].mean_Y(), Ef_k=Pk[k].E(f),
                                   alpha_P=alpha_P, alpha_Q=alpha_Q, between_term=between,
                                   identity_residual=(alpha_Q - alpha_P) - between,
                                   pooled_rel_err_exact=rel_err_exact, pooled_rel_err_check=alpha_P / alpha_Q - 1.0,
                                   perdy_rel_err_exact=perdy_rel_err_exact,
                                   sum_Ptilde=Pt.sum(), sum_Qtilde=Qt.sum()))
        kk = s.k
        vcode = 0 if variant == "k=1..10" else 1
        for n in ns:
            rng = np.random.default_rng([seed, PANEL_CODE["B"], SC_CODE[scenario], vcode, 0, n])
            errs = np.zeros((R, 4))
            for rep in range(R):
                iP, YP = P.sample(rng, n)
                iQ, YQ = Q.sample(rng, n)
                fP, fQ = f[iP], f[iQ]
                a_pool = YP.sum() / fP.sum()
                agg_pool = a_pool * fQ.sum()
                agg_dy = 0.0
                for k in ks:
                    mQ = kk[iQ] == k
                    if not mQ.any():
                        continue
                    mP = kk[iP] == k
                    a_k = YP[mP].sum() / fP[mP].sum() if mP.any() else a_pool
                    agg_dy += a_k * fQ[mQ].sum()
                truth_real, truth_pop = YQ.sum(), n * Q.mean_Y()
                errs[rep] = (agg_pool / truth_real - 1, agg_dy / truth_real - 1, agg_pool / truth_pop - 1, agg_dy / truth_pop - 1)
            sampled_rows.append(dict(scenario=scenario, variant=variant, predictor=pname, n=n, reps=R, alpha_P=alpha_P, alpha_Q=alpha_Q,
                                     pooled_rel_err_exact=rel_err_exact,
                                     pooled_rel_err_vs_realised=errs[:, 0].mean(), perdy_rel_err_vs_realised=errs[:, 1].mean(),
                                     pooled_rel_err_vs_population=errs[:, 2].mean(), perdy_rel_err_vs_population=errs[:, 3].mean(),
                                     sd_pooled_rel_err=errs[:, 0].std(ddof=1), sd_perdy_rel_err=errs[:, 1].std(ddof=1),
                                     mc_se_pooled=errs[:, 2].std(ddof=1) / np.sqrt(R), mc_se_perdy=errs[:, 3].std(ddof=1) / np.sqrt(R)))
    return exact_rows, sampled_rows


# ----------------------------------------------------------------------------------------------------------------
# Panel C: controlled within-group shift at fixed k (exact)
# ----------------------------------------------------------------------------------------------------------------
def transfer_exact(P, Q, f, label):
    s = support()
    assert np.all(Q.p[P.p == 0] == 0), "Q not absolutely continuous w.r.t. P"
    Pt = P.p * f / P.E(f)
    Qt = Q.p * f / Q.E(f)
    nz = Pt > 0
    rho = np.zeros_like(Pt)
    rho[nz] = Qt[nz] / Pt[nz]
    r = np.zeros_like(Pt)
    r[nz] = s.g[nz] / f[nz]
    alpha_P, alpha_Q = P.mean_Y() / P.E(f), Q.mean_Y() / Q.E(f)
    cov = float(np.sum(Pt * (rho - 1.0) * (r - alpha_P)))
    chi = float(np.sqrt(np.sum(Pt * (rho - 1.0) ** 2)))
    sd_r = float(np.sqrt(np.sum(Pt * (r - alpha_P) ** 2)))
    tv = 0.5 * float(np.sum(np.abs(Pt - Qt)))
    rng_r = float(r[nz].max() - r[nz].min())
    actual = abs(alpha_Q - alpha_P)
    return dict(**label, alpha_P=alpha_P, alpha_Q=alpha_Q, discrepancy=alpha_Q - alpha_P, cov_rho_r=cov,
                identity_residual=(alpha_Q - alpha_P) - cov, E_Ptilde_rho=float(np.sum(Pt * rho)),
                chi=chi, sd_r=sd_r, chi_bound=sd_r * chi, tv=tv, range_r=rng_r, tv_bound=rng_r * tv,
                ratio_chi_bound=(sd_r * chi) / actual if actual > 0 else np.inf,
                ratio_tv_bound=(rng_r * tv) / actual if actual > 0 else np.inf,
                rel_err_P_factor=alpha_P / alpha_Q - 1.0, r_min=float(r[nz].min()), r_max=float(r[nz].max()),
                support_P=int(nz.sum()), support_Q=int((Qt > 0).sum()))


def shifted_mix(eps):
    tgt = {"Annuity": 0.08, "Bodily": 0.22, "Technical": 0.07}
    tgt["Material"] = 1.0 - sum(tgt.values())
    mix = {c: (1 - eps) * BASE_MIX[c] + eps * tgt[c] for c in TYPES}
    mix = {c: max(v, 0.0) for c, v in mix.items()}
    tot = sum(mix.values())
    return {c: v / tot for c, v in mix.items()}


def run_panel_C(ks=(3, 5, 7)):
    s = support()
    rows = []
    for k in ks:
        for pname, f in s.predictors.items():
            for sc in SCEN:
                P, Q = training_law(sc, k), target_law(sc, k)
                rows.append(transfer_exact(P, Q, f, dict(scenario=sc, dy=k, family="generator lower triangle", param=np.nan, predictor=pname)))
            P = training_law("S-A", k)
            for eps in (0.25, 0.5, 1.0, 2.0):
                Q = dy_law("S-A", k, upper_ays(k), mix_override=shifted_mix(eps))[0]
                rows.append(transfer_exact(P, Q, f, dict(scenario="S-A", dy=k, family="type-mix shift eps", param=eps, predictor=pname)))
            for lam in (0.5, 1.5, 2.0):
                Q = dy_law("S-A", k, upper_ays(k), delay_scale=lam)[0]
                rows.append(transfer_exact(P, Q, f, dict(scenario="S-A", dy=k, family="reporting-delay scale", param=lam, predictor=pname)))
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Panel D: cross-fitted histogram learner (Assumption A4 holds exactly)
# ----------------------------------------------------------------------------------------------------------------
def fit_hist(idx, Y, S):
    """Cell-mean learners on the finite support: (log-target, raw-target) predictors as vectors over the support, plus visit counts."""
    cnt = np.bincount(idx, minlength=S).astype(float)
    sum_log = np.bincount(idx, weights=np.log1p(Y), minlength=S)
    sum_raw = np.bincount(idx, weights=Y, minlength=S)
    n = len(Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_log = np.where(cnt > 0, np.exp(sum_log / cnt) - 1.0, np.exp(sum_log.sum() / n) - 1.0)
        f_raw = np.where(cnt > 0, sum_raw / cnt, sum_raw.sum() / n)
    return {"log": f_log, "raw": f_raw}, cnt


def run_panel_D(job):
    scenario, k, design, ns, R, seed = job["scenario"], job["k"], job["design"], job["ns"], job["reps"], job["seed"]
    s = support()
    laws, pi = fold_laws(scenario, k, design)
    if laws is None:
        return [], []
    P = training_law(scenario, k)
    finf = {"log": s.f2, "raw": s.g}
    mu = np.array([L.mean_Y() for L in laws])
    alpha_inf = {L_: mu.mean() / np.mean([Lw.E(finf[L_]) for Lw in laws]) for L_ in finf}
    mbar_inf = {L_: np.mean([Lw.E(finf[L_]) for Lw in laws]) for L_ in finf}
    # exact limit variance V of Theorem 4.6(ii)/(iv): J^{-1} sum_j Var_{P_j}(Y - alpha_inf f_inf) / mbar_inf^2
    V_inf = {L_: np.mean([Lw.var_resid(alpha_inf[L_], finf[L_]) for Lw in laws]) / mbar_inf[L_] ** 2 for L_ in finf}
    summ, reps = [], []
    for n in ns:
        rng = np.random.default_rng([seed, PANEL_CODE["D"], SC_CODE[scenario], k, DESIGN_CODE[design], n])
        sizes = np.full(J, n // J)
        rec = {L_: np.zeros((R, 6)) for L_ in finf}   # alpha_hat, se, alpha_n, alpha_pop, mbar_hat, frac_fallback
        for rep in range(R):
            idx, Y, fold = sample_folds(laws, sizes, rng)
            fhat_val = {L_: np.zeros(n) for L_ in finf}
            mhat = {L_: np.zeros(J) for L_ in finf}
            fallback = np.zeros(J)
            for j in range(J):
                tr, te = fold != j, fold == j
                fj, cnt = fit_hist(idx[tr], Y[tr], s.n)
                fallback[j] = (cnt[idx[te]] == 0).mean()
                for L_ in finf:
                    fhat_val[L_][te] = fj[L_][idx[te]]
                    mhat[L_][j] = laws[j].E(fj[L_])                    # exact E_{P_j}[f_hat^{(-j)}(X) | D^{(-j)}]
            ffull, _ = fit_hist(idx, Y, s.n)
            for L_ in finf:
                # the ratios are defined on the event that their denominators are positive (Theorem 4.6); when every Y of the
                # replicate is zero both cell-mean learners vanish identically and the replicate is recorded as undefined
                den_hat, den_full, den_n = fhat_val[L_].sum(), P.E(ffull[L_]), mhat[L_].mean()
                if den_hat > 0 and den_full > 0 and den_n > 0:
                    ah, se = ratio_and_se(Y, fhat_val[L_])
                    rec[L_][rep] = (ah, se, mu.mean() / den_n, P.mean_Y() / den_full, den_n, fallback.mean())
                else:
                    rec[L_][rep] = (np.nan, np.nan, np.nan, np.nan, den_n, fallback.mean())
        for L_ in finf:
            ok = np.isfinite(rec[L_][:, 0])
            n_undefined = int((~ok).sum())
            ah, se, an, apop, mh, fb = rec[L_][ok].T
            d = ah - an
            se_V = np.sqrt(V_inf[L_] / n)
            summ.append(dict(scenario=scenario, dy=k, design=design, learner=L_, n=n, reps=R, n_defined=int(ok.sum()), n_undefined=n_undefined,
                             alpha_inf=alpha_inf[L_], V_inf=V_inf[L_], mbar_inf=mbar_inf[L_],
                             mean_alpha_hat=ah.mean(), mean_alpha_n=an.mean(), mean_alpha_pop=apop.mean(),
                             coverage_alpha_n=(np.abs(ah - an) <= Z975 * se).mean(),
                             coverage_alpha_n_V=(np.abs(ah - an) <= Z975 * se_V).mean(),
                             coverage_alpha_inf=(np.abs(ah - alpha_inf[L_]) <= Z975 * se).mean(),
                             coverage_alpha_inf_V=(np.abs(ah - alpha_inf[L_]) <= Z975 * se_V).mean(),
                             coverage_alpha_pop=(np.abs(ah - apop) <= Z975 * se).mean(),
                             mean_se_plugin=se.mean(), se_V=se_V, sd_alpha_hat=ah.std(ddof=1), sd_alpha_hat_minus_alpha_n=d.std(ddof=1),
                             sd_alpha_hat_minus_alpha_inf=(ah - alpha_inf[L_]).std(ddof=1),
                             se_over_sd=se.mean() / d.std(ddof=1), sd_sqrtn_over_sqrtV=np.sqrt(n) * d.std(ddof=1) / np.sqrt(V_inf[L_]),
                             mean_sqrtn_centring_bias=(np.sqrt(n) * (mh - mbar_inf[L_])).mean(),
                             sd_sqrtn_centring_bias=(np.sqrt(n) * (mh - mbar_inf[L_])).std(ddof=1),
                             mean_sqrtn_centring_bias_rel=(np.sqrt(n) * (mh - mbar_inf[L_]) / mbar_inf[L_]).mean(),
                             mean_rel_centring_bias=((mh - mbar_inf[L_]) / mbar_inf[L_]).mean(),
                             mean_gap_pop_minus_n_over_se=((apop - an) / se).mean(),
                             mean_gap_inf_minus_n_over_se=((alpha_inf[L_] - an) / se).mean(),
                             mean_alpha_hat_over_alpha_inf=ah.mean() / alpha_inf[L_],
                             frac_fallback_cells=fb.mean(),
                             misses_below_n=int((d < -Z975 * se).sum()), misses_above_n=int((d > Z975 * se).sum())))
            for rep in range(R):
                reps.append(dict(scenario=scenario, dy=k, design=design, learner=L_, n=n, rep=rep, alpha_hat=rec[L_][rep, 0], se_plugin=rec[L_][rep, 1],
                                 alpha_n=rec[L_][rep, 2], alpha_pop=rec[L_][rep, 3], mbar_hat=rec[L_][rep, 4]))
    return summ, reps


def verify_panel_D_replicate(scenario="S-B", k=3, design="blocks", n=3000, seed=20260911, n_mc=2_000_000):
    """Print one replicate's fold predictors and check the exact m_hat_j against a brute-force Monte Carlo mean over P_j."""
    s = support()
    laws, pi = fold_laws(scenario, k, design)
    rng = np.random.default_rng([seed, PANEL_CODE["D"], SC_CODE[scenario], k, DESIGN_CODE[design], n])
    sizes = np.full(J, n // J)
    idx, Y, fold = sample_folds(laws, sizes, rng)
    print(f"\n=== panel D hand check: {scenario}, DY {k}, {design}, n={n}, fold sizes {sizes.tolist()}, pi_j={np.round(pi, 4).tolist()} ===")
    pts = np.flatnonzero(sum(L.p for L in laws) > 0)
    tab = pd.DataFrame(dict(type=[TYPES[c] for c in s.ci[pts]], r=s.r[pts], prev=s.prev[pts], g=s.g[pts], f2=s.f2[pts]))
    fh_te = {"log": np.zeros(n), "raw": np.zeros(n)}
    for j in range(J):
        tr, te = fold != j, fold == j
        fj, cnt = fit_hist(idx[tr], Y[tr], s.n)
        tab[f"P{j + 1}(x)"] = laws[j].p[pts]
        tab[f"cnt_train{j + 1}"] = cnt[pts].astype(int)
        tab[f"fhat_log{j + 1}"] = fj["log"][pts]
        for L_ in fh_te:
            fh_te[L_][te] = fj[L_][idx[te]]
        for L_ in ("log", "raw"):
            exact = laws[j].E(fj[L_])
            i_mc, _ = laws[j].sample(np.random.default_rng([seed, 7, j]), n_mc)
            mc = fj[L_][i_mc].mean()
            mc_se = fj[L_][i_mc].std(ddof=1) / np.sqrt(n_mc)
            print(f"fold {j + 1} {L_:3s}: exact m_hat_j = {exact:.6f}   MC mean over P_j (n={n_mc}) = {mc:.6f} +- {mc_se:.6f}   z = {(mc - exact) / mc_se:+.2f}"
                  f"   E_Pj f_inf = {laws[j].E(s.f2 if L_ == 'log' else s.g):.6f}   mu_Y,j = {laws[j].mean_Y():.6f}")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.5g}".format):
        print(tab.to_string(index=False))
    for L_ in ("log", "raw"):
        ah, se = ratio_and_se(Y, fh_te[L_])
        print(f"{L_}: alpha_hat = {ah:.5f}, plug-in SE = {se:.5f}")


# ----------------------------------------------------------------------------------------------------------------
# analytic law check against one large portfolio from src.simulate.generate_dataset
# ----------------------------------------------------------------------------------------------------------------
def law_check(scenario, n_per_ay, min_group=50):
    s = support()
    t0 = time.time()
    d = generate_dataset(scenario, seed=0, n_per_ay=n_per_ay)
    gen_time = time.time() - t0
    cl = d["claims"][["claim_no", "report_dy"]]
    rows = []
    ci_of = {c: i for i, c in enumerate(TYPES)}
    for tri, frame, law_fn, ay_fn in (("upper", d["truth_cells_observed"], training_law, upper_ays),
                                      ("lower", d["truth_cells"], target_law, lower_ays)):
        cells = frame.merge(cl, on="claim_no", how="left")
        cells = cells[cells.dev_lag >= cells.report_dy].copy()
        cells["ci"] = cells.claim_type.map(ci_of)
        for k, gk in cells.groupby("dev_lag"):
            ays = ay_fn(int(k))
            if not ays:
                continue
            L = law_fn(scenario, int(k))
            N = len(gk)
            # (c, r) joint frequency
            for ci in range(len(TYPES)):
                for r in range(1, min(int(k), R_MAX) + 1):
                    p = L.p[s.index[(int(k), ci, r, 0)]] + L.p[s.index[(int(k), ci, r, 1)]]
                    cnt = int(((gk.ci == ci) & (gk.report_dy == r)).sum())
                    if N * p < 5 and cnt == 0:
                        continue
                    se = np.sqrt(N * p * (1 - p)) if 0 < p < 1 else np.nan
                    rows.append(dict(kind=f"{tri} P(c,r|k)", scenario=scenario, dy=int(k), group=f"{TYPES[ci]} r={r}", n=N,
                                     empirical=cnt / N, analytic=p, se=se / N if se else np.nan,
                                     z=(cnt - N * p) / se if se and se > 0 else np.nan, quad_rel_err=np.nan))
            # accident-year composition (exact weights proportional to expected eligible counts)
            tots = np.array([dy_law(scenario, int(k), [a])[1] for a in ays])
            wa = tots / tots.sum()
            for a, p in zip(ays, wa):
                cnt = int((gk.accident_year == a).sum())
                se = np.sqrt(N * p * (1 - p))
                rows.append(dict(kind=f"{tri} P(a|k)", scenario=scenario, dy=int(k), group=f"AY {a}", n=N, empirical=cnt / N, analytic=p,
                                 se=se / N, z=(cnt - N * p) / se if se > 0 else np.nan, quad_rel_err=np.nan))
            # mean payment, oracle mean, positive fraction per DY
            EY, VY = L.mean_Y(), L.E(s.EY2) - L.mean_Y() ** 2
            y = gk.realized_payment.values
            rows.append(dict(kind=f"{tri} E[Y|k]", scenario=scenario, dy=int(k), group="all", n=N, empirical=y.mean(), analytic=EY,
                             se=np.sqrt(VY / N), z=(y.mean() - EY) / np.sqrt(VY / N), quad_rel_err=np.nan))
            o = gk.oracle_payment.values
            rows.append(dict(kind=f"{tri} E[oracle|k]", scenario=scenario, dy=int(k), group="all", n=N, empirical=o.mean(), analytic=EY,
                             se=o.std(ddof=1) / np.sqrt(N), z=(o.mean() - EY) / (o.std(ddof=1) / np.sqrt(N)), quad_rel_err=np.nan))
            pp = L.E(s.ppos)
            rows.append(dict(kind=f"{tri} P(Y>0|k)", scenario=scenario, dy=int(k), group="all", n=N, empirical=(y > 0).mean(), analytic=pp,
                             se=np.sqrt(pp * (1 - pp) / N), z=((y > 0).mean() - pp) / np.sqrt(pp * (1 - pp) / N), quad_rel_err=np.nan))
            # per (c, k, r): E[Y] and P(Y>0) marginalised over prev (exact conditional variance)
            for (ci, r), gg in gk.groupby(["ci", "report_dy"]):
                if len(gg) < min_group:
                    continue
                i0, i1 = s.index[(int(k), int(ci), int(r), 0)], s.index[(int(k), int(ci), int(r), 1)]
                q1 = s.prev1[i0]
                gm = (1 - q1) * s.g[i0] + q1 * s.g[i1]
                v = (1 - q1) * s.EY2[i0] + q1 * s.EY2[i1] - gm ** 2
                ppm = (1 - q1) * s.ppos[i0] + q1 * s.ppos[i1]
                yy = gg.realized_payment.values
                m = len(gg)
                # expected number of active cells in the group: the z-score of a mean is only meaningfully normal when this is not tiny
                exp_active = m * ((1 - q1) * s.p_act[i0] + q1 * s.p_act[i1])
                rows.append(dict(kind=f"{tri} E[Y|c,k,r]", scenario=scenario, dy=int(k), group=f"{TYPES[int(ci)]} r={int(r)}", n=m,
                                 empirical=yy.mean(), analytic=gm, se=np.sqrt(v / m), z=(yy.mean() - gm) / np.sqrt(v / m), quad_rel_err=np.nan,
                                 expected_active=exp_active, normal_approx_ok=bool(exp_active >= 5)))
                rows.append(dict(kind=f"{tri} P(Y>0|c,k,r)", scenario=scenario, dy=int(k), group=f"{TYPES[int(ci)]} r={int(r)}", n=m,
                                 empirical=(yy > 0).mean(), analytic=ppm, se=np.sqrt(ppm * (1 - ppm) / m),
                                 z=((yy > 0).mean() - ppm) / np.sqrt(ppm * (1 - ppm) / m), quad_rel_err=np.nan,
                                 expected_active=exp_active, normal_approx_ok=bool(exp_active >= 5)))
    return rows, dict(scenario=scenario, n_claims=int(len(d["claims"])), gen_seconds=gen_time)


# ----------------------------------------------------------------------------------------------------------------
# LaTeX table and README
# ----------------------------------------------------------------------------------------------------------------
def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pct(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.{nd}f}"


def write_table(out_dir, tex_path, ns):
    A = pd.read_csv(os.path.join(out_dir, "panelA_summary.csv"))
    Ap = pd.read_csv(os.path.join(out_dir, "panelAprime_summary.csv"))
    Bx = pd.read_csv(os.path.join(out_dir, "panelB_exact.csv"))
    Bs = pd.read_csv(os.path.join(out_dir, "panelB_sampled.csv"))
    C = pd.read_csv(os.path.join(out_dir, "panelC_exact.csv"))
    D = pd.read_csv(os.path.join(out_dir, "panelD_summary.csv"))
    ns = sorted(ns)
    ncol = len(ns)

    def by_n(df, col):
        return " & ".join(fmt(df[df.n == n][col].mean()) if (df.n == n).any() else "--" for n in ns)

    def pair(df, c1, c2, nd=3):
        return " & ".join(f"{fmt(df[df.n == n][c1].mean(), nd)}/{fmt(df[df.n == n][c2].mean(), nd)}" if (df.n == n).any() else "--" for n in ns)

    def triple(df, c1, c2, c3, nd=2):
        return " & ".join(f"{fmt(df[df.n == n][c1].mean(), nd)}/{fmt(df[df.n == n][c2].mean(), nd)}/{fmt(df[df.n == n][c3].mean(), nd)}"
                          if (df.n == n).any() else "--" for n in ns)

    def dyrange(df):
        return f"DY {int(df.dy.min())}--{int(df.dy.max())}"

    def crow(c, ncol):
        return f"\\multicolumn{{{ncol}}}{{l}}{{{fmt(abs(c.discrepancy), 4)} / {fmt(c.chi_bound, 4)} / {fmt(c.tv_bound, 3)}}} \\\\"

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering\scriptsize\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.9}")
    lines.append(r"\caption{Simulation in which the theorems' hypotheses hold exactly (\texttt{tools/hypothesis\_exact\_sim.py}; design")
    lines.append(r"in \S\ref{sec:sim}): $J=3$, nominal $95\%$, $R=2000$ replications per configuration, coverages pooled over the")
    lines.append(r"development years listed (Monte Carlo s.e.\ below $0.0065$), fixed predictors $f_1=e^{\mu_c+\beta_c(k-r)}-1$ and")
    lines.append(r"$f_2=\exp\E[\log(1+Y)\mid\check x]-1$. Panels: A, Theorem~\ref{thm:asymp}(i); A$'$, Theorem~\ref{thm:asymp}(v)(a)")
    lines.append(r"with unequal deterministic fold sizes (the data's DY-2 shares; shares proportional to the block probabilities")
    lines.append(r"$\pi_j$, at which the nominal target equals $\alpha_P$); B, Corollary~\ref{cor:pergroup}(iii), a pure")
    lines.append(r"development-year mix shift, relative error against the population aggregate; C,")
    lines.append(r"Theorem~\ref{thm:transfer}(i),(iii); D, Theorem~\ref{thm:asymp}(iv), for which Assumption~\ref{as:stable} holds,")
    lines.append(r"the centring term $\sqrt n(\bar{\hat m}-\bar m_\infty)/\bar m_\infty$ and $\bar{\hat\alpha}/\alpha_\infty$ being")
    lines.append(r"means over replications and development years.}")
    lines.append(r"\label{tab:hypothesis-exact}")
    lines.append(r"\begin{tabular}{l>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{3.05cm}" + "r" * ncol + "}")
    lines.append(r"\toprule")
    lines.append("Panel & Setting & Quantity & " + " & ".join(f"$n={n}$" for n in ns) + r" \\")
    lines.append(r"\midrule")
    for sc, pn, ptex in (("S-A", "f1", "$f_1$"), ("S-A", "f2", "$f_2$"), ("S-B", "f1", "$f_1$")):
        sub = A[(A.scenario == sc) & (A.design == "random") & (A.predictor == pn)]
        if len(sub):
            lines.append(f"A & {sc}, random folds, {ptex} & cov.\\ of $\\alpha_n$: plug-in / exact $V_n$ ({dyrange(sub)}) & {pair(sub, 'coverage_alpha_n', 'coverage_alpha_n_Vn')} \\\\")
    sub = A[(A.scenario == "S-B") & (A.design == "blocks") & (A.predictor == "f1")]
    if len(sub):
        lines.append(f"A & S-B, AY blocks, $f_1$ & cov.\\ of $\\alpha_n$ / of pooled $\\alpha_P$ ({dyrange(sub)}) & {pair(sub, 'coverage_alpha_n', 'coverage_alpha_P')} \\\\")
    lines.append(r"\midrule")
    for wt, wtex in (("data", "the data's DY-2 shares"), ("proportional", "shares $w_j=\\pi_j$")):
        sub = Ap[(Ap.scenario == "S-B") & (Ap.weighting == wt) & (Ap.predictor == "f1")]
        if len(sub):
            lines.append(f"A$'$ & S-B, AY blocks, {wtex}, $f_1$ & cov.\\ of $\\alpha^w_n$: plug-in / exact $V^w$ ({dyrange(sub)}) & {pair(sub, 'coverage_alpha_w', 'coverage_alpha_w_Vw')} \\\\")
    lines.append(r"\midrule")
    for sc in SCEN:
      for variant, vtex in (("k=1..10", "DY 1--10"), ("k=2..10", "DY 2--10")):
        bs = Bs[(Bs.scenario == sc) & (Bs.variant == variant)] if "variant" in Bs.columns else Bs[Bs.scenario == sc]
        bx = Bx[(Bx.scenario == sc) & (Bx.variant == variant)]
        if len(bs):
            ex = bx.pooled_rel_err_exact.iloc[0]
            lines.append(f"B & {sc}, DY mix shift over {vtex}, $f_1$ & rel.\\ error pooled / per-DY (\\%); exact ${pct(ex)}$ / $0$ & "
                         + " & ".join(f"{pct(bs[bs.n == n].pooled_rel_err_vs_population.mean())}/{pct(bs[bs.n == n].perdy_rel_err_vs_population.mean())}" if (bs.n == n).any() else "--" for n in ns) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{C: within-DY shift of $P_k$, $f_1$ (exact)} & \multicolumn{" + str(ncol) + r"}{l}{$|\alpha_Q-\alpha_P|$ / $\chi^2$ bound / TV bound} \\")
    for k in (3, 5, 7):
        c = C[(C.scenario == "S-A") & (C.dy == k) & (C.family == "generator lower triangle") & (C.predictor == "f1")]
        if len(c):
            lines.append(f"C & S-A, DY {k}, own lower-triangle law $Q_k$ & reporting truncation $r\\le I-a+1$ & {crow(c.iloc[0], ncol)}")
    c = C[(C.scenario == "S-B") & (C.dy == 5) & (C.family == "generator lower triangle") & (C.predictor == "f1")]
    if len(c):
        lines.append(f"C & S-B, DY 5, own lower-triangle law $Q_k$ & truncation $+$ type-mix drift & {crow(c.iloc[0], ncol)}")
    c = C[(C.scenario == "S-A") & (C.dy == 5) & (C.family == "type-mix shift eps") & (C.param == 2.0) & (C.predictor == "f1")]
    if len(c):
        lines.append(f"C & S-A, DY 5, type-mix shift $\\varepsilon=2$ & Annuity $0.13$, Bodily $0.29$, Material $0.51$ & {crow(c.iloc[0], ncol)}")
    lines.append(r"\midrule")
    sub = D[(D.scenario == "S-A") & (D.design == "random") & (D.learner == "log")]
    if len(sub):
        lines.append(f"D & S-A, random folds, log-target cell means & cov.\\ at $\\alpha_n$ / $\\alpha_\\infty$ / $\\alpha^{{\\mathrm{{pop}}}}$ ({dyrange(sub)}) & {triple(sub, 'coverage_alpha_n', 'coverage_alpha_inf', 'coverage_alpha_pop')} \\\\")
        lines.append(f"D & S-A, random folds, log-target cell means & centring $\\sqrt n(\\bar{{\\hat m}}-\\bar m_\\infty)/\\bar m_\\infty$; $\\bar{{\\hat\\alpha}}/\\alpha_\\infty$ & "
                     + " & ".join(f"{fmt(sub[sub.n == n].mean_sqrtn_centring_bias_rel.mean(), 2)}; {fmt(sub[sub.n == n].mean_alpha_hat_over_alpha_inf.mean(), 3)}" if (sub.n == n).any() else "--" for n in ns) + r" \\")
    sub = D[(D.scenario == "S-A") & (D.design == "random") & (D.learner == "raw")]
    if len(sub):
        lines.append(f"D & S-A, random folds, raw cell means ($\\alpha_\\infty=1$) & cov.\\ at $\\alpha_n$ / $\\alpha_\\infty$ / $\\alpha^{{\\mathrm{{pop}}}}$ ({dyrange(sub)}) & {triple(sub, 'coverage_alpha_n', 'coverage_alpha_inf', 'coverage_alpha_pop')} \\\\")
    sub = D[(D.scenario == "S-B") & (D.design == "blocks") & (D.learner == "log")]
    if len(sub):
        lines.append(f"D & S-B, AY blocks, log-target cell means & cov.\\ at $\\alpha_n$ / $\\alpha_\\infty$ / $\\alpha^{{\\mathrm{{pop}}}}$ ({dyrange(sub)}) & {triple(sub, 'coverage_alpha_n', 'coverage_alpha_inf', 'coverage_alpha_pop')} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_readme(out_dir, args, timings, law_info, headline):
    lines = ["# results_sim_hyp -- simulation satisfying the theorems' hypotheses exactly", "",
             "Produced by `tools/hypothesis_exact_sim.py` (see its module docstring for the design).", "",
             "## Commands", "", "```", f"{sys.executable} tools/hypothesis_exact_sim.py " + " ".join(sys.argv[1:]), "```", "",
             f"Arguments: scenarios={args.scenarios}, reps={args.reps}, ns={args.ns}, seed={args.seed}, panels={args.panels}, jobs={args.jobs}", "",
             "## Runtime", ""]
    lines += [f"- {k}: {v:.1f} s" for k, v in timings.items()]
    lines += ["", "## Generator portfolios used for the law check", ""]
    lines += [f"- {li['scenario']}: {li['n_claims']} claims, generate_dataset {li['gen_seconds']:.1f} s" for li in law_info]
    lines += ["", "## Files", "",
              "- `analytic_law_check.csv`: empirical vs analytic laws (z-scores; |z| > 4 flagged in the log) plus the moment-formula checks",
              "- `panelA_summary.csv`: Theorem 4.6(i) with fixed predictors",
              "- `panelAprime_summary.csv`: unequal deterministic fold sizes",
              "- `panelB_exact.csv`, `panelB_sampled.csv`: pure development-year mix shift (Corollary 4.12(iii))",
              "- `panelC_exact.csv`: within-DY shifts, Theorem 4.11(i),(iii)",
              "- `panelD_summary.csv`: cross-fitted histogram learner, Theorem 4.6(iv)",
              "- `panelA_by_replicate.csv`, `panelD_by_replicate.csv`: the per-replicate rows behind panels A and D. Written to the run directory, but not deposited with the released summaries (~52 MB); regenerate them with the command above.",
              "- `run.log`: console log; `DIGEST.md`: plain-text digest",
              "- LaTeX table: `paper/ime/tables/table_hypothesis_exact.tex`", "", "## Headline numbers", ""]
    lines += headline
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def headline_lines(out_dir):
    out = []
    p = os.path.join
    if os.path.exists(p(out_dir, "analytic_law_check.csv")):
        L = pd.read_csv(p(out_dir, "analytic_law_check.csv"))
        zz = L.z.dropna()
        out.append(f"- law check: {len(zz)} z-scores, max |z| = {zz.abs().max():.2f}, {(zz.abs() > 4).sum()} with |z| > 4, "
                   f"{(zz.abs() > 3).sum()} with |z| > 3; moment formulas vs quadrature max rel. err = {L.quad_rel_err.abs().max():.2e}")
        if "normal_approx_ok" in L.columns:
            ok = L[(L.normal_approx_ok != False) | L.expected_active.isna()].z.dropna()  # noqa: E712
            out.append(f"- law check excluding sparse-event groups (< 5 expected active cells): {len(ok)} z-scores, max |z| = {ok.abs().max():.2f}, "
                       f"{(ok.abs() > 4).sum()} with |z| > 4")
    if os.path.exists(p(out_dir, "panelA_summary.csv")):
        A = pd.read_csv(p(out_dir, "panelA_summary.csv"))
        for (sc, dsg, pn), g in A.groupby(["scenario", "design", "predictor"]):
            out.append(f"- A {sc} {dsg} {pn}: coverage of alpha_n by n = " + ", ".join(f"{n}: {g[g.n == n].coverage_alpha_n.mean():.3f}" for n in sorted(g.n.unique()))
                       + f"; coverage of alpha_P = " + ", ".join(f"{g[g.n == n].coverage_alpha_P.mean():.3f}" for n in sorted(g.n.unique()))
                       + f"; SE/sd = " + ", ".join(f"{g[g.n == n].se_over_sd.mean():.3f}" for n in sorted(g.n.unique())))
    if os.path.exists(p(out_dir, "panelAprime_summary.csv")):
        Ap = pd.read_csv(p(out_dir, "panelAprime_summary.csv"))
        for (sc, wt, pn), g in Ap.groupby(["scenario", "weighting", "predictor"]):
            out.append(f"- A' {sc} {wt} {pn}: coverage of alpha^w_n = " + ", ".join(f"{n}: {g[g.n == n].coverage_alpha_w.mean():.3f}" for n in sorted(g.n.unique()))
                       + "; with V^w SE = " + ", ".join(f"{g[g.n == n].coverage_alpha_w_Vw.mean():.3f}" for n in sorted(g.n.unique()))
                       + f"; sd sqrt(n)/sqrt(V^w) = " + ", ".join(f"{g[g.n == n].sd_sqrtn_over_sqrtVw.mean():.3f}" for n in sorted(g.n.unique())))
    if os.path.exists(p(out_dir, "panelB_sampled.csv")):
        Bs = pd.read_csv(p(out_dir, "panelB_sampled.csv"))
        Bx = pd.read_csv(p(out_dir, "panelB_exact.csv"))
        if "variant" not in Bs.columns:
            Bs = Bs.assign(variant="k=1..10")
        for (sc, variant), g in Bs.groupby(["scenario", "variant"]):
            ex = Bx[(Bx.scenario == sc) & (Bx.variant == variant)]
            out.append(f"- B {sc} {variant}: exact pooled rel. error {100 * g.pooled_rel_err_exact.iloc[0]:+.2f}% (identity residual {ex.identity_residual.abs().max():.1e}); "
                       + "sampled pooled/per-DY (vs population) by n = " + ", ".join(f"{n}: {100 * g[g.n == n].pooled_rel_err_vs_population.iloc[0]:+.2f}%/{100 * g[g.n == n].perdy_rel_err_vs_population.iloc[0]:+.2f}%" for n in sorted(g.n.unique())))
    if os.path.exists(p(out_dir, "panelC_exact.csv")):
        C = pd.read_csv(p(out_dir, "panelC_exact.csv"))
        out.append(f"- C: max identity residual {C.identity_residual.abs().max():.1e}; chi bound / actual ranges {C.ratio_chi_bound.min():.1f}--{C.ratio_chi_bound.max():.1f}, "
                   f"TV bound / actual {C.ratio_tv_bound.min():.1f}--{C.ratio_tv_bound.max():.1f}")
        for _, r in C[(C.family == "generator lower triangle") & (C.predictor == "f1")].iterrows():
            out.append(f"  - C {r.scenario} DY {int(r.dy)} f1 (own lower-triangle law): alpha_P={r.alpha_P:.4f}, alpha_Q={r.alpha_Q:.4f}, "
                       f"|diff|={abs(r.discrepancy):.4f}, chi bound={r.chi_bound:.4f}, TV bound={r.tv_bound:.3f}, rel. err of P-factor {100 * r.rel_err_P_factor:+.2f}%")
    if os.path.exists(p(out_dir, "panelD_summary.csv")):
        D = pd.read_csv(p(out_dir, "panelD_summary.csv"))
        for (sc, dsg, L_), g in D.groupby(["scenario", "design", "learner"]):
            out.append(f"- D {sc} {dsg} {L_}: coverage alpha_n / alpha_inf / alpha_pop by n = "
                       + ", ".join(f"{n}: {g[g.n == n].coverage_alpha_n.mean():.3f}/{g[g.n == n].coverage_alpha_inf.mean():.3f}/{g[g.n == n].coverage_alpha_pop.mean():.3f}" for n in sorted(g.n.unique()))
                       + "; coverage of alpha_n with exact V = " + ", ".join(f"{g[g.n == n].coverage_alpha_n_V.mean():.3f}" for n in sorted(g.n.unique()))
                       + "; mean sqrt(n) centring bias (relative to mbar_inf) = " + ", ".join(f"{g[g.n == n].mean_sqrtn_centring_bias_rel.mean():+.3f}" for n in sorted(g.n.unique()))
                       + "; SE/sd = " + ", ".join(f"{g[g.n == n].se_over_sd.mean():.3f}" for n in sorted(g.n.unique()))
                       + "; mean alpha_hat/alpha_inf = " + ", ".join(f"{g[g.n == n].mean_alpha_hat_over_alpha_inf.mean():.4f}" for n in sorted(g.n.unique())))
    return out


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------
def _run_job(job):
    fn = {"A": run_panel_A, "Aprime": run_panel_Aprime, "D": run_panel_D}[job["panel"]]
    t0 = time.time()
    summ, reps = fn(job)
    return job, summ, reps, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenarios", nargs="+", default=SCEN)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--ns", nargs="+", type=int, default=[600, 3000, 15000])
    ap.add_argument("--out", default=os.path.join(ROOT, "results_sim_hyp"))
    ap.add_argument("--tex", default=os.path.join(ROOT, "paper", "ime", "tables", "table_hypothesis_exact.tex"))
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--quick", action="store_true", help="smoke test: reps=50, ns=600 3000, law check with n_per_ay=2000")
    ap.add_argument("--panels", nargs="+", default=["law", "A", "Aprime", "B", "C", "D"])
    ap.add_argument("--law-n-per-ay", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--verify-d", action="store_true", help="print one panel-D replicate by hand and exit")
    ap.add_argument("--table-only", action="store_true", help="regenerate the LaTeX table and print the headline lines from existing CSVs")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.ns, args.law_n_per_ay = 50, [600, 3000], 2000
    os.makedirs(args.out, exist_ok=True)
    if args.verify_d:
        verify_panel_D_replicate(seed=args.seed)
        return
    if args.table_only:
        write_table(args.out, args.tex, args.ns)
        print(f"LaTeX table written to {args.tex}")
        print("\n".join(headline_lines(args.out)))
        return
    for sc in args.scenarios:
        if sc not in SC_CODE:
            raise SystemExit(f"scenario {sc} not supported (S-C violates Assumption A3 by construction; omitted)")
    t_all = time.time()
    timings, law_info = {}, []
    print(f"support: {support().n} points; sigma^2 = {SIGMA ** 2:.4f}; z = {Z975:.4f}; jobs = {args.jobs}", flush=True)

    if "law" in args.panels:
        t0 = time.time()
        rows = verify_moment_formulas(args.seed)
        for r in rows:
            print(f"moment check {r['group']:28s}: MC {r['empirical']:.6g} vs exact {r['analytic']:.6g} (z={r['z']:+.2f}); quad rel err {r['quad_rel_err']:.1e}")
        for sc in args.scenarios:
            rr, info = law_check(sc, args.law_n_per_ay)
            rows += rr
            law_info.append(info)
            df = pd.DataFrame(rr)
            zz = df.z.dropna()
            print(f"law check {sc}: {info['n_claims']} claims ({info['gen_seconds']:.1f} s); {len(zz)} z-scores, max |z| = {zz.abs().max():.2f}, "
                  f"|z|>4: {(zz.abs() > 4).sum()}, |z|>3: {(zz.abs() > 3).sum()}", flush=True)
            for kind, g in df.groupby("kind"):
                z = g.z.dropna()
                print(f"    {kind:22s} n_checks={len(z):4d}  mean z={z.mean():+.3f}  sd z={z.std(ddof=1) if len(z) > 1 else float('nan'):.3f}  max|z|={z.abs().max():.2f}")
            flagged = df[df.z.abs() > 4]
            for _, r in flagged.iterrows():
                note = ""
                if "expected_active" in df.columns and not pd.isna(r.get("expected_active", np.nan)):
                    note = f", expected active cells {r.expected_active:.2f}" + ("" if r.normal_approx_ok else " -- sparse-event group, z not normal")
                print(f"    FLAG |z|>4: {r.kind} DY {r.dy} {r.group}: empirical {r.empirical:.5g} vs analytic {r.analytic:.5g} (n={r.n}, z={r.z:+.2f}{note})")
            if "normal_approx_ok" in df.columns:
                ok = df[(df.normal_approx_ok != False) | df.expected_active.isna()].z.dropna()  # noqa: E712
                print(f"    excluding sparse-event groups (< 5 expected active cells): {len(ok)} z-scores, max |z| = {ok.abs().max():.2f}, |z|>4: {(ok.abs() > 4).sum()}")
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "analytic_law_check.csv"), index=False)
        timings["law check"] = time.time() - t0

    jobs = []
    for sc in args.scenarios:
        if "A" in args.panels:
            for k in range(1, K_DY + 1):
                jobs.append(dict(panel="A", scenario=sc, k=k, design="random", ns=args.ns, reps=args.reps, seed=args.seed))
            for k in range(1, 5):
                jobs.append(dict(panel="A", scenario=sc, k=k, design="blocks", ns=args.ns, reps=args.reps, seed=args.seed))
        if "Aprime" in args.panels:
            for k in range(1, 5):
                for wt in ("data", "proportional"):
                    jobs.append(dict(panel="Aprime", scenario=sc, k=k, design=wt, ns=args.ns, reps=args.reps, seed=args.seed))
        if "D" in args.panels:
            for k in range(2, 10):
                jobs.append(dict(panel="D", scenario=sc, k=k, design="random", ns=args.ns, reps=args.reps, seed=args.seed))
            for k in range(2, 5):
                jobs.append(dict(panel="D", scenario=sc, k=k, design="blocks", ns=args.ns, reps=args.reps, seed=args.seed))
    results = {"A": ([], []), "Aprime": ([], []), "D": ([], [])}
    if jobs:
        t0 = time.time()
        # heavy jobs first
        jobs.sort(key=lambda j: (j["panel"] != "D", j["panel"] != "A"))
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for job, summ, reps, dt in ex.map(_run_job, jobs):
                results[job["panel"]][0].extend(summ)
                results[job["panel"]][1].extend(reps)
                print(f"done {job['panel']:6s} {job['scenario']} DY {job['k']:2d} {job['design']:12s} {dt:6.1f} s", flush=True)
        timings["panels A/A'/D (wall, parallel)"] = time.time() - t0
    if "A" in args.panels:
        A = pd.DataFrame(results["A"][0])
        A.to_csv(os.path.join(args.out, "panelA_summary.csv"), index=False)
        pd.DataFrame(results["A"][1]).to_csv(os.path.join(args.out, "panelA_by_replicate.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print("\nPanel A summary (coverage of alpha_n pooled over DY):")
            print(A.groupby(["scenario", "design", "predictor", "n"])[["coverage_alpha_n", "coverage_alpha_n_Vn", "coverage_alpha_P", "se_over_sd", "sd_sqrtn_over_sqrtVn"]].mean().to_string())
    if "Aprime" in args.panels:
        Ap = pd.DataFrame(results["Aprime"][0])
        Ap.to_csv(os.path.join(args.out, "panelAprime_summary.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print("\nPanel A' summary:")
            print(Ap.groupby(["scenario", "weighting", "predictor", "n"])[["coverage_alpha_w", "coverage_alpha_w_Vw", "coverage_alpha_P", "coverage_alpha_equal", "se_over_sd", "sd_sqrtn_over_sqrtVw"]].mean().to_string())
        prop = Ap[(Ap.weighting == "proportional")]
        for _, r in prop.drop_duplicates(["scenario", "dy", "predictor"]).iterrows():
            print(f"A' proportional {r.scenario} DY {r.dy} {r.predictor}: alpha^w_n(pi) = {r.alpha_w_nominal:.12f}   alpha_P = {r.alpha_P:.12f}   "
                  f"diff = {r.alpha_w_nominal - r.alpha_P:.3e}   (realised shares {r.w_realised}: alpha^w_n = {r.alpha_w:.12f}; equal-weight alpha_n = {r.alpha_equal_weight:.12f})")
    if "B" in args.panels:
        t0 = time.time()
        ex_rows, sa_rows = [], []
        for sc in args.scenarios:
            e, sr = run_panel_B(sc, args.ns, args.reps, args.seed)
            ex_rows += e
            sa_rows += sr
        Bx, Bs = pd.DataFrame(ex_rows), pd.DataFrame(sa_rows)
        Bx.to_csv(os.path.join(args.out, "panelB_exact.csv"), index=False)
        Bs.to_csv(os.path.join(args.out, "panelB_sampled.csv"), index=False)
        timings["panel B"] = time.time() - t0
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.6f}".format):
            print("\nPanel B exact (k=1..10):")
            print(Bx[Bx.variant == "k=1..10"][["scenario", "dy", "Pbar", "Qbar", "Ptilde", "Qtilde", "alpha_Pk", "alpha_P", "alpha_Q", "identity_residual", "pooled_rel_err_exact"]].to_string(index=False))
            print("\nPanel B sampled:")
            print(Bs.to_string(index=False))
    if "C" in args.panels:
        t0 = time.time()
        C = pd.DataFrame(run_panel_C())
        C.to_csv(os.path.join(args.out, "panelC_exact.csv"), index=False)
        timings["panel C"] = time.time() - t0
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.5g}".format):
            print("\nPanel C exact:")
            print(C[["scenario", "dy", "family", "param", "predictor", "alpha_P", "alpha_Q", "discrepancy", "identity_residual", "chi_bound", "tv_bound", "ratio_chi_bound", "ratio_tv_bound", "rel_err_P_factor"]].to_string(index=False))
    if "D" in args.panels:
        D = pd.DataFrame(results["D"][0])
        D.to_csv(os.path.join(args.out, "panelD_summary.csv"), index=False)
        pd.DataFrame(results["D"][1]).to_csv(os.path.join(args.out, "panelD_by_replicate.csv"), index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
            print("\nPanel D summary (pooled over DY):")
            print(D.groupby(["scenario", "design", "learner", "n"])[["coverage_alpha_n", "coverage_alpha_n_V", "coverage_alpha_inf", "coverage_alpha_pop", "se_over_sd", "sd_sqrtn_over_sqrtV", "mean_sqrtn_centring_bias_rel", "mean_gap_pop_minus_n_over_se", "mean_alpha_hat_over_alpha_inf", "frac_fallback_cells"]].mean().to_string())
    timings["total wall"] = time.time() - t_all
    if all(os.path.exists(os.path.join(args.out, f)) for f in ("panelA_summary.csv", "panelAprime_summary.csv", "panelB_exact.csv", "panelB_sampled.csv", "panelC_exact.csv", "panelD_summary.csv")):
        write_table(args.out, args.tex, args.ns)
        print(f"\nLaTeX table written to {args.tex}")
    head = headline_lines(args.out)
    write_readme(args.out, args, timings, law_info, head)
    print("\n".join(head))
    print(f"\ntotal wall time {timings['total wall']:.1f} s")


if __name__ == "__main__":
    main()
