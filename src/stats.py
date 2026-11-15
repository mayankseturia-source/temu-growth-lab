"""A/B testing statistics (Blog 9)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def two_proportion_ztest(x1, n1, x2, n2):
    """H0: p1 == p2.  z = (p2 - p1) / sqrt(p(1-p)(1/n1 + 1/n2))"""
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci = (p2 - p1 - 1.96 * se_diff, p2 - p1 + 1.96 * se_diff)
    return {"p_control": p1, "p_test": p2, "diff": p2 - p1, "rel_lift": (p2 - p1) / p1 if p1 else np.nan,
            "z": z, "p_value": pval, "ci_low": ci[0], "ci_high": ci[1]}


def welch_ttest(a, b):
    """H0: mean(a) == mean(b), unequal variances.  t = (mean_b - mean_a) / sqrt(s_a^2/n_a + s_b^2/n_b)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = b.mean() - a.mean()
    return {"mean_control": a.mean(), "mean_test": b.mean(), "diff": d, "t": float(t), "p_value": float(p),
            "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se}


def sample_size_proportions(p1, mde, alpha=0.05, power=0.80):
    """n per group = (z_{1-a/2} sqrt(2 p(1-p)) + z_{1-b} sqrt(p1(1-p1) + p2(1-p2)))^2 / mde^2"""
    p2 = p1 + mde
    pbar = (p1 + p2) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde ** 2
    return int(np.ceil(n))


def stratified_split(strata, rng, share_test=0.5):
    """Random 50/50 (or other) split inside every stratum so both groups have the same mix."""
    strata = np.asarray(strata)
    is_test = np.zeros(len(strata), dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        k = int(round(len(idx) * share_test))
        is_test[rng.choice(idx, size=k, replace=False)] = True
    return is_test
