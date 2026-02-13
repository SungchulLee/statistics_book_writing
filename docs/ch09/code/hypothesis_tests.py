"""
Chapter 9: Hypothesis Testing — Code Examples
==============================================

Demonstrates:
1. One-sample z-test and t-test for μ
2. One-sample test for proportion p
3. Two-sample t-test (pooled and Welch)
4. Paired t-test
5. Power analysis and sample size
6. CI–Test duality
"""

import numpy as np
from scipy import stats

# =============================================================================
# 1. One-Sample Tests for μ
# =============================================================================

print("=" * 60)
print("1. ONE-SAMPLE TESTS FOR μ")
print("=" * 60)

# Example: Factory claims mean weight is 500g
data = np.array([498, 495, 502, 497, 501, 499, 496, 503, 494, 500,
                 497, 502, 496, 501, 498, 499, 495, 503, 497, 500])
mu_0 = 500
alpha = 0.05

# t-test (σ unknown)
t_stat, p_value = stats.ttest_1samp(data, mu_0)
print(f"H₀: μ = {mu_0}  vs  H₁: μ ≠ {mu_0}")
print(f"x̄ = {data.mean():.2f}, s = {data.std(ddof=1):.2f}, n = {len(data)}")
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < alpha else 'Fail to reject H₀'}")

# One-sided test: H₁: μ < 500
p_one_sided = p_value / 2 if t_stat < 0 else 1 - p_value / 2
print(f"\nOne-sided (H₁: μ < {mu_0}): p-value = {p_one_sided:.4f}")

# Manual computation
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
t_manual = (xbar - mu_0) / (s / np.sqrt(n))
p_manual = 2 * stats.t.cdf(-abs(t_manual), df=n-1)
print(f"\nManual: t = {t_manual:.4f}, p = {p_manual:.4f}")

# =============================================================================
# 2. One-Sample Test for Proportion p
# =============================================================================

print("\n" + "=" * 60)
print("2. ONE-SAMPLE TEST FOR PROPORTION")
print("=" * 60)

# Example: Is the defect rate different from 5%?
x, n = 12, 200  # 12 defectives in 200
p_0 = 0.05
p_hat = x / n

z_stat = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
p_value = 2 * stats.norm.sf(abs(z_stat))

print(f"H₀: p = {p_0}  vs  H₁: p ≠ {p_0}")
print(f"p̂ = {p_hat:.3f}, n = {n}")
print(f"z = {z_stat:.4f}, p-value = {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < alpha else 'Fail to reject H₀'}")

# Using statsmodels
from statsmodels.stats.proportion import proportions_ztest
z_sm, p_sm = proportions_ztest(x, n, value=p_0)
print(f"statsmodels: z = {z_sm:.4f}, p = {p_sm:.4f}")

# =============================================================================
# 3. Two-Sample t-Tests
# =============================================================================

print("\n" + "=" * 60)
print("3. TWO-SAMPLE t-TESTS")
print("=" * 60)

# Example: Drug A vs Drug B recovery time
drug_a = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 5.3, 6.0, 5.1, 5.4])
drug_b = np.array([4.1, 3.8, 4.5, 4.2, 3.9, 4.6, 4.0, 4.3, 3.7, 4.4])

# Welch's t-test (default: unequal variances)
t_welch, p_welch = stats.ttest_ind(drug_a, drug_b, equal_var=False)
print(f"Welch's t-test: t = {t_welch:.4f}, p = {p_welch:.6f}")

# Pooled t-test (equal variances assumed)
t_pooled, p_pooled = stats.ttest_ind(drug_a, drug_b, equal_var=True)
print(f"Pooled t-test:  t = {t_pooled:.4f}, p = {p_pooled:.6f}")

print(f"Drug A: x̄ = {drug_a.mean():.2f}, s = {drug_a.std(ddof=1):.2f}")
print(f"Drug B: x̄ = {drug_b.mean():.2f}, s = {drug_b.std(ddof=1):.2f}")
print(f"Decision: {'Reject H₀' if p_welch < 0.05 else 'Fail to reject H₀'}")

# =============================================================================
# 4. Paired t-Test
# =============================================================================

print("\n" + "=" * 60)
print("4. PAIRED t-TEST")
print("=" * 60)

# Example: Blood pressure before and after medication
before = np.array([145, 150, 138, 155, 142, 148, 136, 152, 140, 146])
after  = np.array([138, 142, 130, 148, 135, 140, 132, 145, 134, 139])
diff = after - before

t_stat, p_value = stats.ttest_rel(after, before)
print(f"Paired t-test: t = {t_stat:.4f}, p-value = {p_value:.6f}")

# Equivalent to one-sample t-test on differences
t_stat2, p_value2 = stats.ttest_1samp(diff, 0)
print(f"Equivalent 1-sample on diffs: t = {t_stat2:.4f}, p = {p_value2:.6f}")

print(f"Mean difference: {diff.mean():.1f}, SD of diffs: {diff.std(ddof=1):.2f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# =============================================================================
# 5. Power Analysis
# =============================================================================

print("\n" + "=" * 60)
print("5. POWER ANALYSIS")
print("=" * 60)

from statsmodels.stats.power import TTestPower, TTestIndPower

# One-sample: How many subjects to detect a 5-point difference?
analysis = TTestPower()
effect_size = 5 / 15  # Cohen's d = (μ₁ - μ₀) / σ
n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                 power=0.80, alternative='two-sided')
print(f"One-sample: effect size d = {effect_size:.3f}")
print(f"  n needed for 80% power: {int(np.ceil(n_needed))}")

# Power for a given n
power = analysis.power(effect_size=effect_size, nobs=50, alpha=0.05,
                       alternative='two-sided')
print(f"  Power with n=50: {power:.3f}")

# Two-sample: How many per group?
analysis2 = TTestIndPower()
n_each = analysis2.solve_power(effect_size=0.5, alpha=0.05, power=0.80,
                                ratio=1.0, alternative='two-sided')
print(f"\nTwo-sample: medium effect d = 0.5")
print(f"  n per group for 80% power: {int(np.ceil(n_each))}")

# Power curve
print("\nPower curve (one-sample, d=0.33):")
for n in [10, 20, 30, 50, 75, 100, 150, 200]:
    pwr = analysis.power(effect_size=effect_size, nobs=n, alpha=0.05)
    bar = "█" * int(pwr * 40)
    print(f"  n={n:>4}: power={pwr:.3f} {bar}")

# =============================================================================
# 6. CI–Test Duality
# =============================================================================

print("\n" + "=" * 60)
print("6. CI-TEST DUALITY DEMONSTRATION")
print("=" * 60)

data = np.array([52, 48, 55, 50, 47, 53, 49, 51, 54, 46])
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
alpha = 0.05

# 95% CI
t_c = stats.t.ppf(1 - alpha/2, df=n-1)
me = t_c * s / np.sqrt(n)
ci = (xbar - me, xbar + me)
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

# Test various μ₀ values
for mu0 in [48, 49, 50, 51, 52, 53]:
    t_stat, p_val = stats.ttest_1samp(data, mu0)
    in_ci = ci[0] <= mu0 <= ci[1]
    reject = p_val < alpha
    print(f"  μ₀={mu0}: p={p_val:.4f}, "
          f"{'reject' if reject else 'fail to reject':>15}, "
          f"{'in CI' if in_ci else 'NOT in CI':>10}")

print("\nDuality: reject H₀ ⟺ μ₀ is NOT in the CI")
