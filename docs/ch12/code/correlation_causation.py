"""
Chapter 17: Correlation and Causation — Code Examples
======================================================
Pearson/Spearman/Kendall tests, partial correlation,
Simpson's paradox demo, and spurious correlation illustration.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================================================
# 1. Pearson, Spearman, Kendall Comparison
# =============================================================================

def correlation_comparison():
    """Compare three correlation measures on different relationship types."""
    n = 100

    # (a) Linear
    x_lin = np.random.normal(0, 1, n)
    y_lin = 2 * x_lin + np.random.normal(0, 1, n)

    # (b) Monotonic nonlinear (exponential)
    x_mono = np.random.uniform(0, 3, n)
    y_mono = np.exp(x_mono) + np.random.normal(0, 2, n)

    # (c) Nonlinear (quadratic) — r ≈ 0 but strong relationship
    x_quad = np.random.normal(0, 2, n)
    y_quad = x_quad**2 + np.random.normal(0, 1, n)

    # (d) With outliers
    x_out = np.random.normal(0, 1, n)
    y_out = 0.8 * x_out + np.random.normal(0, 0.5, n)
    # Add 3 extreme outliers
    x_out[:3] = [5, -5, 6]
    y_out[:3] = [-5, 5, -6]

    datasets = [
        ("Linear", x_lin, y_lin),
        ("Monotonic Nonlinear", x_mono, y_mono),
        ("Quadratic (r ≈ 0)", x_quad, y_quad),
        ("With Outliers", x_out, y_out),
    ]

    print(f"{'Dataset':<25} {'Pearson r':>10} {'Spearman ρ':>12} {'Kendall τ':>12}")
    print("-" * 65)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, x, y) in zip(axes, datasets):
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        r_k, p_k = stats.kendalltau(x, y)
        print(f"{name:<25} {r_p:>10.4f} {r_s:>12.4f} {r_k:>12.4f}")

        ax.scatter(x, y, alpha=0.5, s=15)
        ax.set_title(f"{name}\nr={r_p:.3f}, ρₛ={r_s:.3f}, τ={r_k:.3f}", fontsize=9)

    plt.tight_layout()
    plt.savefig("correlation_comparison.png", dpi=150)
    plt.show()

correlation_comparison()


# =============================================================================
# 2. Fisher z-Transformation and CI for Correlation
# =============================================================================

def fisher_z_ci(x, y, alpha=0.05):
    """Compute CI for ρ using Fisher z-transformation."""
    n = len(x)
    r, p_val = stats.pearsonr(x, y)

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    rho_lo, rho_hi = np.tanh(z_lo), np.tanh(z_hi)

    print(f"\nFisher z-Transformation CI for ρ")
    print(f"  r = {r:.4f}, p-value = {p_val:.6f}")
    print(f"  z = arctanh(r) = {z:.4f}")
    print(f"  SE(z) = 1/√(n-3) = {se:.4f}")
    print(f"  {100*(1-alpha):.0f}% CI for ρ: ({rho_lo:.4f}, {rho_hi:.4f})")

    return rho_lo, rho_hi

np.random.seed(42)
x = np.random.normal(0, 1, 80)
y = 0.6 * x + np.random.normal(0, 0.8, 80)
fisher_z_ci(x, y)


# =============================================================================
# 3. Simpson's Paradox Demonstration
# =============================================================================

def simpsons_paradox_demo():
    """Generate and visualize Simpson's paradox."""
    np.random.seed(42)

    # Confounding variable Z (e.g., department, severity)
    # Within each group: X has NEGATIVE effect on Y
    # Overall: X appears to have POSITIVE effect on Y

    groups = {"Group A": (50, 0.2, 8, -0.5),   # n, x_mean, y_base, slope
              "Group B": (50, 0.5, 5, -0.5),
              "Group C": (50, 0.8, 2, -0.5)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_x, all_y = [], []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for (name, (n, xm, yb, slope)), c in zip(groups.items(), colors):
        x = np.random.normal(xm, 0.15, n)
        y = yb + slope * x + np.random.normal(0, 0.3, n)
        all_x.extend(x)
        all_y.extend(y)

        axes[0].scatter(x, y, alpha=0.6, label=name, color=c, s=20)
        # Within-group regression line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        axes[0].plot(x_line, m * x_line + b, color=c, linewidth=2)

    # Overall regression
    all_x, all_y = np.array(all_x), np.array(all_y)
    m_all, b_all = np.polyfit(all_x, all_y, 1)
    x_line = np.linspace(all_x.min(), all_x.max(), 50)
    axes[0].plot(x_line, m_all * x_line + b_all, 'k--', linewidth=2, label="Overall")

    r_within = np.mean([stats.pearsonr(
        np.random.normal(xm, 0.15, n),
        yb + slope * np.random.normal(xm, 0.15, n) + np.random.normal(0, 0.3, n)
    )[0] for n, xm, yb, slope in groups.values()])

    axes[0].set_title("Simpson's Paradox\n"
                      f"Within groups: negative slope\n"
                      f"Overall: slope = {m_all:.2f} (positive!)")
    axes[0].set_xlabel("X (treatment)")
    axes[0].set_ylabel("Y (outcome)")
    axes[0].legend(fontsize=8)

    # Right panel: marginal view
    axes[1].scatter(all_x, all_y, alpha=0.4, color='gray', s=15)
    axes[1].plot(x_line, m_all * x_line + b_all, 'r-', linewidth=2)
    r_overall, p_overall = stats.pearsonr(all_x, all_y)
    axes[1].set_title(f"Marginal View (Ignoring Groups)\n"
                      f"r = {r_overall:.3f}, p = {p_overall:.4f}")
    axes[1].set_xlabel("X (treatment)")
    axes[1].set_ylabel("Y (outcome)")

    plt.tight_layout()
    plt.savefig("simpsons_paradox.png", dpi=150)
    plt.show()

    print("\nSimpson's Paradox:")
    print(f"  Overall correlation: r = {r_overall:.4f} (positive)")
    print(f"  Within-group slope:  {slope:.1f} (negative)")
    print("  Confounding by group membership reverses the sign!")

simpsons_paradox_demo()


# =============================================================================
# 4. Partial Correlation
# =============================================================================

def partial_correlation_demo():
    """Demonstrate partial correlation to control for confounding."""
    n = 200
    np.random.seed(42)

    # Z (confounder) causes both X and Y
    Z = np.random.normal(0, 1, n)
    X = 0.7 * Z + np.random.normal(0, 0.5, n)
    Y = 0.6 * Z + np.random.normal(0, 0.5, n)
    # X and Y have NO direct causal connection

    r_xy, p_xy = stats.pearsonr(X, Y)
    r_xz, _ = stats.pearsonr(X, Z)
    r_yz, _ = stats.pearsonr(Y, Z)

    # Partial correlation formula
    r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    # Equivalent: residual method
    from numpy.polynomial.polynomial import polyfit
    res_x = X - np.polyval(np.polyfit(Z, X, 1), Z)
    res_y = Y - np.polyval(np.polyfit(Z, Y, 1), Z)
    r_residual, _ = stats.pearsonr(res_x, res_y)

    print("\nPartial Correlation Demo")
    print(f"  r(X, Y)     = {r_xy:.4f}  (p = {p_xy:.6f}) — appears significant!")
    print(f"  r(X, Z)     = {r_xz:.4f}")
    print(f"  r(Y, Z)     = {r_yz:.4f}")
    print(f"  r(X,Y | Z)  = {r_xy_z:.4f}  (formula)")
    print(f"  r(X,Y | Z)  = {r_residual:.4f}  (residual method)")
    print(f"  → After controlling for Z, the correlation nearly vanishes!")

partial_correlation_demo()


# =============================================================================
# 5. Spurious Correlations from Multiple Testing
# =============================================================================

def spurious_correlations_demo(n_vars=100, n_obs=30):
    """Show that searching many pairs yields 'significant' correlations by chance."""
    data = np.random.normal(0, 1, (n_obs, n_vars))

    # All pairwise correlations
    n_pairs = n_vars * (n_vars - 1) // 2
    p_values = []
    max_r, max_pair = 0, (0, 0)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            r, p = stats.pearsonr(data[:, i], data[:, j])
            p_values.append(p)
            if abs(r) > abs(max_r):
                max_r = r
                max_pair = (i, j)

    p_values = np.array(p_values)
    n_sig_005 = np.sum(p_values < 0.05)
    n_sig_001 = np.sum(p_values < 0.01)

    print(f"\nSpurious Correlations from {n_vars} Independent Variables")
    print(f"  Total pairs tested: {n_pairs}")
    print(f"  'Significant' at α = 0.05: {n_sig_005} ({100*n_sig_005/n_pairs:.1f}%)")
    print(f"  'Significant' at α = 0.01: {n_sig_001} ({100*n_sig_001/n_pairs:.1f}%)")
    print(f"  Strongest r = {max_r:.4f} between vars {max_pair}")
    print(f"  Expected false positives at 5%: {0.05 * n_pairs:.0f}")

spurious_correlations_demo()


# =============================================================================
# 6. Comparing Two Independent Correlations
# =============================================================================

def compare_two_correlations(r1, n1, r2, n2, alpha=0.05):
    """Test H0: ρ1 = ρ2 using Fisher z-transformation."""
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print(f"\nComparing Two Correlations")
    print(f"  r₁ = {r1:.4f} (n₁ = {n1}),  r₂ = {r2:.4f} (n₂ = {n2})")
    print(f"  z-statistic = {z_stat:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  {'Reject' if p_value < alpha else 'Fail to reject'} H₀: ρ₁ = ρ₂ at α = {alpha}")

# Example: is the height-weight correlation different for men vs women?
compare_two_correlations(r1=0.72, n1=100, r2=0.65, n2=120)
