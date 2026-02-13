"""
Chapter 20: Resampling Methods — Code Examples
================================================
Bootstrap CI methods, permutation tests, and comparisons.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================================================
# 1. Bootstrap Standard Error and Confidence Intervals
# =============================================================================

def bootstrap_ci_demo(data=None, B=10_000, alpha=0.05):
    """Demonstrate all four bootstrap CI methods for the mean and median."""
    if data is None:
        # Skewed data: exponential
        data = np.random.exponential(scale=3.0, size=50)

    n = len(data)
    theta_hat = np.mean(data)
    theta_median = np.median(data)

    # Bootstrap resampling
    boot_means = np.array([np.mean(np.random.choice(data, n, replace=True))
                           for _ in range(B)])
    boot_medians = np.array([np.median(np.random.choice(data, n, replace=True))
                             for _ in range(B)])

    z = stats.norm.ppf(1 - alpha / 2)

    for stat_name, theta, boot in [("Mean", theta_hat, boot_means),
                                    ("Median", theta_median, boot_medians)]:
        se_boot = boot.std(ddof=1)

        # Method 1: Normal
        ci_normal = (theta - z * se_boot, theta + z * se_boot)

        # Method 2: Percentile
        ci_pct = (np.percentile(boot, 100 * alpha / 2),
                  np.percentile(boot, 100 * (1 - alpha / 2)))

        # Method 3: Basic (Pivotal)
        ci_basic = (2 * theta - np.percentile(boot, 100 * (1 - alpha / 2)),
                    2 * theta - np.percentile(boot, 100 * alpha / 2))

        # Method 4: BCa
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot < theta))
        # Acceleration (jackknife)
        jack_vals = np.array([np.mean(np.delete(data, i)) if stat_name == "Mean"
                              else np.median(np.delete(data, i))
                              for i in range(n)])
        jack_mean = jack_vals.mean()
        a_hat = np.sum((jack_mean - jack_vals)**3) / (6 * np.sum((jack_mean - jack_vals)**2)**1.5)

        alpha1 = stats.norm.cdf(z0 + (z0 + stats.norm.ppf(alpha / 2)) /
                                 (1 - a_hat * (z0 + stats.norm.ppf(alpha / 2))))
        alpha2 = stats.norm.cdf(z0 + (z0 + stats.norm.ppf(1 - alpha / 2)) /
                                 (1 - a_hat * (z0 + stats.norm.ppf(1 - alpha / 2))))
        ci_bca = (np.percentile(boot, 100 * alpha1),
                  np.percentile(boot, 100 * alpha2))

        print(f"\n{stat_name}: θ̂ = {theta:.4f}, SE_boot = {se_boot:.4f}")
        print(f"  Normal:     ({ci_normal[0]:.4f}, {ci_normal[1]:.4f})")
        print(f"  Percentile: ({ci_pct[0]:.4f}, {ci_pct[1]:.4f})")
        print(f"  Basic:      ({ci_basic[0]:.4f}, {ci_basic[1]:.4f})")
        print(f"  BCa:        ({ci_bca[0]:.4f}, {ci_bca[1]:.4f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name, boot in [(axes[0], "Mean", boot_means),
                            (axes[1], "Median", boot_medians)]:
        ax.hist(boot, bins=50, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(f"Bootstrap Distribution of {name}")
        ax.set_xlabel(name)
    plt.tight_layout()
    plt.savefig("bootstrap_distributions.png", dpi=150)
    plt.show()

bootstrap_ci_demo()


# =============================================================================
# 2. Bootstrap Coverage Simulation
# =============================================================================

def bootstrap_coverage(n=30, B_boot=2000, n_sim=2000, true_mu=3.0, true_scale=3.0):
    """Check empirical coverage of bootstrap CIs for the mean of Exp(3)."""
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)

    coverage = {"Normal": 0, "Percentile": 0, "Basic": 0, "t-interval (parametric)": 0}

    for _ in range(n_sim):
        data = np.random.exponential(true_scale, n)
        theta = data.mean()

        boot = np.array([np.random.choice(data, n, replace=True).mean()
                         for _ in range(B_boot)])
        se = boot.std(ddof=1)

        # Normal
        lo, hi = theta - z * se, theta + z * se
        if lo <= true_mu <= hi:
            coverage["Normal"] += 1

        # Percentile
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if lo <= true_mu <= hi:
            coverage["Percentile"] += 1

        # Basic
        lo = 2 * theta - np.percentile(boot, 97.5)
        hi = 2 * theta - np.percentile(boot, 2.5)
        if lo <= true_mu <= hi:
            coverage["Basic"] += 1

        # Parametric t
        se_t = data.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        lo, hi = theta - t_crit * se_t, theta + t_crit * se_t
        if lo <= true_mu <= hi:
            coverage["t-interval (parametric)"] += 1

    print(f"Coverage of 95% CIs for mean of Exp({true_scale}), n={n}, {n_sim} simulations")
    for method, count in coverage.items():
        print(f"  {method:<25}: {count / n_sim:.4f}")

bootstrap_coverage()


# =============================================================================
# 3. Two-Sample Permutation Test
# =============================================================================

def permutation_test_two_sample(x, y, B=10_000, stat_func=None):
    """Two-sample permutation test for difference in means."""
    if stat_func is None:
        stat_func = lambda a, b: np.mean(a) - np.mean(b)

    t_obs = stat_func(x, y)
    m, n = len(x), len(y)
    pooled = np.concatenate([x, y])

    count = 0
    perm_stats = []
    for _ in range(B):
        perm = np.random.permutation(pooled)
        t_perm = stat_func(perm[:m], perm[m:])
        perm_stats.append(t_perm)
        if abs(t_perm) >= abs(t_obs):
            count += 1

    p_value = count / B
    return t_obs, p_value, np.array(perm_stats)


# Example: treatment vs control
treatment = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.8, 6.3, 5.1, 5.7, 6.0])
control = np.array([4.1, 3.9, 4.5, 4.2, 3.8, 4.0, 4.6, 3.7, 4.3, 4.4])

t_obs, p_perm, perm_dist = permutation_test_two_sample(treatment, control)
_, p_ttest = stats.ttest_ind(treatment, control)

print("\nTwo-Sample Permutation Test")
print(f"  Observed diff: {t_obs:.4f}")
print(f"  Permutation p-value: {p_perm:.4f}")
print(f"  t-test p-value:      {p_ttest:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(perm_dist, bins=50, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
ax.axvline(t_obs, color="red", linewidth=2, label=f"Observed = {t_obs:.3f}")
ax.axvline(-t_obs, color="red", linewidth=2, linestyle="--")
ax.set_title("Permutation Distribution Under H₀")
ax.set_xlabel("Difference in Means")
ax.legend()
plt.tight_layout()
plt.savefig("permutation_test.png", dpi=150)
plt.show()


# =============================================================================
# 4. Permutation Test for Correlation
# =============================================================================

def permutation_test_correlation(x, y, B=10_000):
    """Permutation test for Pearson correlation."""
    r_obs = np.corrcoef(x, y)[0, 1]

    count = 0
    perm_rs = []
    for _ in range(B):
        y_perm = np.random.permutation(y)
        r_perm = np.corrcoef(x, y_perm)[0, 1]
        perm_rs.append(r_perm)
        if abs(r_perm) >= abs(r_obs):
            count += 1

    return r_obs, count / B, np.array(perm_rs)


np.random.seed(123)
x_corr = np.random.normal(0, 1, 40)
y_corr = 0.4 * x_corr + np.random.normal(0, 1, 40)

r_obs, p_perm_corr, perm_rs = permutation_test_correlation(x_corr, y_corr)
_, p_pearson = stats.pearsonr(x_corr, y_corr)

print("\nPermutation Test for Correlation")
print(f"  Observed r: {r_obs:.4f}")
print(f"  Permutation p-value: {p_perm_corr:.4f}")
print(f"  Pearson p-value:     {p_pearson:.4f}")


# =============================================================================
# 5. Paired Permutation Test (Sign-Flip)
# =============================================================================

def paired_permutation_test(x, y, B=10_000):
    """Paired permutation test via sign-flipping differences."""
    d = x - y
    t_obs = np.mean(d)
    n = len(d)

    count = 0
    perm_stats = []
    for _ in range(B):
        signs = np.random.choice([-1, 1], size=n)
        t_perm = np.mean(signs * d)
        perm_stats.append(t_perm)
        if abs(t_perm) >= abs(t_obs):
            count += 1

    return t_obs, count / B, np.array(perm_stats)


before = np.array([82, 78, 91, 85, 73, 88, 79, 95, 84, 76])
after = np.array([88, 82, 95, 89, 78, 91, 84, 98, 90, 81])

t_obs_paired, p_paired, _ = paired_permutation_test(before, after)
_, p_paired_t = stats.ttest_rel(after, before)

print("\nPaired Permutation Test (Before/After)")
print(f"  Mean difference: {np.mean(after - before):.2f}")
print(f"  Permutation p-value: {p_paired:.4f}")
print(f"  Paired t-test p:     {p_paired_t:.4f}")


# =============================================================================
# 6. Bootstrap vs Permutation: Side-by-Side
# =============================================================================

def bootstrap_vs_permutation_comparison(x, y, B=10_000):
    """Compare bootstrap CI with permutation p-value."""
    m, n = len(x), len(y)
    diff_obs = np.mean(x) - np.mean(y)

    # Bootstrap CI for difference
    boot_diffs = []
    for _ in range(B):
        x_boot = np.random.choice(x, m, replace=True)
        y_boot = np.random.choice(y, n, replace=True)
        boot_diffs.append(np.mean(x_boot) - np.mean(y_boot))
    boot_diffs = np.array(boot_diffs)
    ci = np.percentile(boot_diffs, [2.5, 97.5])

    # Permutation test
    pooled = np.concatenate([x, y])
    count = 0
    for _ in range(B):
        perm = np.random.permutation(pooled)
        if abs(np.mean(perm[:m]) - np.mean(perm[m:])) >= abs(diff_obs):
            count += 1
    p_perm = count / B

    print(f"\nBootstrap vs Permutation Comparison")
    print(f"  Observed difference: {diff_obs:.4f}")
    print(f"  Bootstrap 95% CI:    ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"  Permutation p-value: {p_perm:.4f}")
    print(f"  CI excludes 0:       {'Yes' if ci[0] > 0 or ci[1] < 0 else 'No'}")
    print(f"  Perm rejects at 5%:  {'Yes' if p_perm < 0.05 else 'No'}")

bootstrap_vs_permutation_comparison(treatment, control)
