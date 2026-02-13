"""
Chapter 8: Confidence Intervals — Code Examples
================================================

Demonstrates:
1. CI construction for μ (z and t intervals)
2. CI for proportion p (Wald and Wilson)
3. CI for variance σ² (chi-square)
4. Two-sample CI for μ₁ - μ₂
5. Coverage simulation
6. Sample size determination
"""

import numpy as np
from scipy import stats

# =============================================================================
# 1. CI for μ (z-interval and t-interval)
# =============================================================================

print("=" * 60)
print("1. CONFIDENCE INTERVAL FOR μ")
print("=" * 60)

# Example: Blood pressure measurements
data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126,
                 119, 132, 124, 117, 129, 123, 131, 116, 127, 120])
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
alpha = 0.05

# z-interval (if σ known, say σ = 6)
sigma_known = 6
z_crit = stats.norm.ppf(1 - alpha/2)
me_z = z_crit * sigma_known / np.sqrt(n)
print(f"z-interval (σ={sigma_known} known): ({xbar - me_z:.2f}, {xbar + me_z:.2f})")

# t-interval (σ unknown, use s)
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
me_t = t_crit * s / np.sqrt(n)
print(f"t-interval (σ unknown):   ({xbar - me_t:.2f}, {xbar + me_t:.2f})")

# Using scipy directly
ci = stats.t.interval(1 - alpha, df=n-1, loc=xbar, scale=s/np.sqrt(n))
print(f"scipy t.interval:         ({ci[0]:.2f}, {ci[1]:.2f})")

# =============================================================================
# 2. CI for Proportion p
# =============================================================================

print("\n" + "=" * 60)
print("2. CONFIDENCE INTERVAL FOR p")
print("=" * 60)

# Example: 84 successes in 200 trials
x, n = 84, 200
p_hat = x / n
alpha = 0.05
z = stats.norm.ppf(1 - alpha/2)

# Wald interval
me_wald = z * np.sqrt(p_hat * (1 - p_hat) / n)
print(f"Wald interval:   ({p_hat - me_wald:.4f}, {p_hat + me_wald:.4f})")

# Wilson interval (preferred)
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2*n)) / denom
me_wilson = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
print(f"Wilson interval: ({center - me_wilson:.4f}, {center + me_wilson:.4f})")

# Agresti-Coull interval
n_tilde = n + z**2
p_tilde = (x + z**2/2) / n_tilde
me_ac = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
print(f"Agresti-Coull:   ({p_tilde - me_ac:.4f}, {p_tilde + me_ac:.4f})")

# =============================================================================
# 3. CI for Variance σ²
# =============================================================================

print("\n" + "=" * 60)
print("3. CONFIDENCE INTERVAL FOR σ²")
print("=" * 60)

data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126])
n = len(data)
s2 = data.var(ddof=1)
alpha = 0.05

chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)

ci_var = ((n-1)*s2 / chi2_upper, (n-1)*s2 / chi2_lower)
ci_sd = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))

print(f"s² = {s2:.2f}")
print(f"95% CI for σ²: ({ci_var[0]:.2f}, {ci_var[1]:.2f})")
print(f"95% CI for σ:  ({ci_sd[0]:.2f}, {ci_sd[1]:.2f})")

# =============================================================================
# 4. Two-Sample CI for μ₁ - μ₂
# =============================================================================

print("\n" + "=" * 60)
print("4. TWO-SAMPLE CI FOR μ₁ - μ₂")
print("=" * 60)

# Example: Drug A vs Drug B recovery times (days)
group_a = np.array([12, 15, 11, 14, 13, 16, 10, 15, 12, 14])
group_b = np.array([18, 20, 17, 19, 16, 21, 15, 20, 18, 17])

n1, n2 = len(group_a), len(group_b)
x1, x2 = group_a.mean(), group_b.mean()
s1, s2_val = group_a.std(ddof=1), group_b.std(ddof=1)
alpha = 0.05

# Welch's t-interval (unequal variances)
se = np.sqrt(s1**2/n1 + s2_val**2/n2)
df_welch = (s1**2/n1 + s2_val**2/n2)**2 / (
    (s1**2/n1)**2/(n1-1) + (s2_val**2/n2)**2/(n2-1)
)
t_crit = stats.t.ppf(1 - alpha/2, df=df_welch)
diff = x1 - x2
ci_welch = (diff - t_crit*se, diff + t_crit*se)
print(f"Welch's CI for μ₁-μ₂: ({ci_welch[0]:.2f}, {ci_welch[1]:.2f})")
print(f"  df = {df_welch:.1f}, diff = {diff:.2f}")

# Pooled t-interval (equal variances assumed)
sp2 = ((n1-1)*s1**2 + (n2-1)*s2_val**2) / (n1 + n2 - 2)
se_pooled = np.sqrt(sp2 * (1/n1 + 1/n2))
t_crit_p = stats.t.ppf(1 - alpha/2, df=n1+n2-2)
ci_pooled = (diff - t_crit_p*se_pooled, diff + t_crit_p*se_pooled)
print(f"Pooled CI for μ₁-μ₂:  ({ci_pooled[0]:.2f}, {ci_pooled[1]:.2f})")

# Using scipy
result = stats.ttest_ind(group_a, group_b, equal_var=False)
ci_scipy = result.confidence_interval(confidence_level=0.95)
print(f"scipy Welch CI:        ({ci_scipy.low:.2f}, {ci_scipy.high:.2f})")

# =============================================================================
# 5. Coverage Simulation
# =============================================================================

print("\n" + "=" * 60)
print("5. COVERAGE SIMULATION")
print("=" * 60)

np.random.seed(42)
mu_true, sigma_true = 100, 15
n_sim = 10_000

for n in [5, 10, 30, 100]:
    z_covers = 0
    t_covers = 0
    for _ in range(n_sim):
        sample = np.random.normal(mu_true, sigma_true, n)
        xbar = sample.mean()
        s = sample.std(ddof=1)

        # z-interval using s (common but wrong)
        me_z = 1.96 * s / np.sqrt(n)
        if xbar - me_z <= mu_true <= xbar + me_z:
            z_covers += 1

        # t-interval (correct)
        t_c = stats.t.ppf(0.975, df=n-1)
        me_t = t_c * s / np.sqrt(n)
        if xbar - me_t <= mu_true <= xbar + me_t:
            t_covers += 1

    print(f"n={n:>3}: z-coverage={z_covers/n_sim:.3f}  t-coverage={t_covers/n_sim:.3f}")

# =============================================================================
# 6. Sample Size Determination
# =============================================================================

print("\n" + "=" * 60)
print("6. SAMPLE SIZE DETERMINATION")
print("=" * 60)

# For estimating μ within ±E
sigma_est = 15  # estimated σ
for E in [1, 2, 3, 5]:
    for conf in [0.90, 0.95, 0.99]:
        z = stats.norm.ppf(1 - (1-conf)/2)
        n_needed = int(np.ceil((z * sigma_est / E)**2))
        print(f"  E=±{E}, {conf*100:.0f}% conf → n = {n_needed}")
    print()

# For estimating p within ±E (conservative: p=0.5)
print("Sample size for proportion (conservative p=0.5):")
for E in [0.01, 0.03, 0.05]:
    for conf in [0.95, 0.99]:
        z = stats.norm.ppf(1 - (1-conf)/2)
        n_needed = int(np.ceil((z / (2*E))**2))
        print(f"  E=±{E}, {conf*100:.0f}% conf → n = {n_needed}")
    print()
