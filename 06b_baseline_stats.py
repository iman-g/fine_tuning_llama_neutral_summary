"""
06b_baseline_stats.py

Stats for the baseline neutrality-validation table (Table 4.5).
Reads baseline_neutrality_judged.csv produced by 06_judge_baseline.py.

Reports, for roundup vs each article type:
  - Cohen's d (pooled SD): separation of the two neutrality distributions
  - 95% CI via paired bootstrap (resamples EXAMPLES, preserving pairing)
  - Wilcoxon signed-rank p (paired, non-parametric; correct for 0-5 ordinal)
Plus a Spearman trend test: does neutrality rise with distance-from-partisan?
  encoding: left/right article = 0, center article = 1, roundup = 2
"""

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)
N_BOOT = 5000

df = pd.read_csv("baseline_neutrality_judged.csv")
df = df[df["status"] == "ok"].copy()
df["neutrality"] = df["neutrality"].astype(float)

# wide format: one row per example, one column per item_type
wide = df.pivot_table(index="example_id", columns="item_type",
                      values="neutrality", aggfunc="first")


def cohens_d(a, b):
    """Pooled-SD Cohen's d for a vs b (positive = a higher)."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def boot_ci_d(pair_df, col_a, col_b):
    """Paired bootstrap: resample examples, recompute d each time."""
    vals = pair_df[[col_a, col_b]].dropna().values
    n = len(vals)
    ds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)          # resample examples (rows)
        s = vals[idx]
        ds[i] = cohens_d(s[:, 0], s[:, 1])
    return np.percentile(ds, [2.5, 97.5])


print("=== Roundup vs each article type ===")
print(f"{'contrast':<22}{'d':>7}{'95% CI':>20}{'wilcoxon p':>14}")
for art in ["left_article", "center_article", "right_article"]:
    pair = wide[["roundup", art]].dropna()
    d = cohens_d(pair["roundup"].values, pair[art].values)
    lo, hi = boot_ci_d(pair, "roundup", art)
    # paired Wilcoxon signed-rank
    w_p = stats.wilcoxon(pair["roundup"].values, pair[art].values).pvalue
    print(f"roundup vs {art:<11}{d:>7.2f}   [{lo:>5.2f}, {hi:>5.2f}]   {w_p:>12.2e}")

# left vs right symmetry check (should be ~0, n.s.)
pair = wide[["left_article", "right_article"]].dropna()
d_lr = cohens_d(pair["left_article"].values, pair["right_article"].values)
p_lr = stats.wilcoxon(pair["left_article"].values, pair["right_article"].values).pvalue
print(f"\nleft vs right symmetry: d = {d_lr:+.2f}, wilcoxon p = {p_lr:.3f} "
      f"(expect ~0 and n.s.)")

# Spearman trend: neutrality vs distance-from-partisan
rank_map = {"left_article": 0, "right_article": 0,
            "center_article": 1, "roundup": 2}
trend = df[df["item_type"].isin(rank_map)].copy()
trend["bias_rank"] = trend["item_type"].map(rank_map)
rho, p = stats.spearmanr(trend["bias_rank"], trend["neutrality"])
print(f"\nSpearman trend (neutrality vs neutrality-rank): "
      f"rho = {rho:.3f}, p = {p:.2e}")
print("Positive rho + tiny p => judge recovers the bias gradient monotonically.")
