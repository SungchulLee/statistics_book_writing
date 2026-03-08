"""
Hypothesis Testing with Height/Weight Data
=============================================
Adapted from Basic-Statistics-With-Python Chapter 4 notebook.

Demonstrates three core hypothesis-testing workflows:
1. One-sample t-test   — is average male height 172 cm?
2. Two-sample z-test   — do male and female heights differ?
3. Two-proportion test — do banks discriminate against women?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)

# ── Synthetic data (inspired by 500-Person dataset) ────────
N = 250
MALE_HEIGHTS   = stats.norm.rvs(loc=170, scale=8, size=N)
FEMALE_HEIGHTS = stats.norm.rvs(loc=165, scale=7, size=N)


# ── 1. One-sample t-test ───────────────────────────────────
def one_sample_ttest(data, mu0=172, alpha=0.05):
    """H0: mu = mu0  vs  H1: mu != mu0"""
    n = len(data)
    xbar = data.mean()
    s = data.std(ddof=1)
    se = s / np.sqrt(n)
    df = n - 1

    t_stat = (xbar - mu0) / se
    p_val = 2 * stats.t.cdf(-abs(t_stat), df)

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    reject = abs(t_stat) > t_crit

    return {
        "xbar": xbar, "s": s, "se": se, "df": df,
        "t": t_stat, "p": p_val, "t_crit": t_crit, "reject": reject,
    }


# ── 2. Two-sample z-test (known sigma) ─────────────────────
def two_sample_ztest(x, y, sigma_x, sigma_y, D0=0):
    """H0: mu_x - mu_y = D0  (known population sigmas)."""
    n1, n2 = len(x), len(y)
    se = np.sqrt(sigma_x**2 / n1 + sigma_y**2 / n2)
    z = (x.mean() - y.mean() - D0) / se
    p = 2 * stats.norm.cdf(-abs(z))
    return z, p, se


# ── 3. Two-proportion z-test ───────────────────────────────
def two_proportion_ztest(k1, n1, k2, n2):
    """H0: p1 = p2   vs   H1: p1 > p2."""
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p = 1 - stats.norm.cdf(z)  # one-sided
    return z, p, p1, p2


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Hypothesis Testing — Height/Weight & Proportion Examples")
    print("=" * 60)

    # --- 1. One-sample t-test ---
    print("\n--- 1. One-Sample t-Test: H0: mu_male = 172 cm ---")
    res = one_sample_ttest(MALE_HEIGHTS, mu0=172)
    print(f"  n = {len(MALE_HEIGHTS)},  x̄ = {res['xbar']:.2f},  s = {res['s']:.2f}")
    print(f"  t-stat = {res['t']:.4f},  p-value = {res['p']:.4f}")
    print(f"  t-crit = +/-{res['t_crit']:.4f}")
    print(f"  => {'Reject' if res['reject'] else 'Fail to reject'} H0")

    # --- 2. Two-sample z-test ---
    print("\n--- 2. Two-Sample z-Test: H0: mu_male = mu_female ---")
    # assume known population sigmas
    z, p, se = two_sample_ztest(MALE_HEIGHTS, FEMALE_HEIGHTS,
                                 sigma_x=8, sigma_y=7)
    print(f"  Male mean = {MALE_HEIGHTS.mean():.2f},  "
          f"Female mean = {FEMALE_HEIGHTS.mean():.2f}")
    print(f"  z-stat = {z:.4f},  p-value = {p:.6f}")
    if p < 0.05:
        print("  => Reject H0: heights differ significantly.")
    else:
        print("  => Fail to reject H0.")

    # --- 2b. Two-sample t-test (unknown sigma) ---
    print("\n--- 2b. Two-Sample t-Test (Unknown sigma, Welch) ---")
    t_w, p_w = stats.ttest_ind(MALE_HEIGHTS, FEMALE_HEIGHTS, equal_var=False)
    print(f"  Welch t-stat = {t_w:.4f},  p-value = {p_w:.6f}")

    # --- 3. Two-proportion test ---
    print("\n--- 3. Two-Proportion z-Test: Bank Discrimination ---")
    print("  Women: 59 rejected out of 649 applicants")
    print("  Men:   128 rejected out of 2490 applicants")
    z_p, p_p, p1, p2 = two_proportion_ztest(59, 649, 128, 2490)
    print(f"  p̂_women = {p1:.4f},  p̂_men = {p2:.4f}")
    print(f"  z-stat = {z_p:.4f},  p-value (one-sided) = {p_p:.4f}")
    if p_p < 0.05:
        print("  => Reject H0: women have a significantly higher rejection rate.")
    else:
        print("  => Fail to reject H0.")

    # --- Visualisation ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # panel 1: male height distribution + null
    ax = axes[0]
    ax.hist(MALE_HEIGHTS, bins=25, density=True, alpha=0.6,
            edgecolor="white", label="Male heights")
    xr = np.linspace(MALE_HEIGHTS.min()-5, MALE_HEIGHTS.max()+5, 200)
    ax.plot(xr, stats.norm.pdf(xr, 172, 8), "r--", lw=2, label="H0: N(172, 8)")
    ax.axvline(res["xbar"], color="blue", linestyle=":", lw=2,
               label=f"x̄ = {res['xbar']:.1f}")
    ax.set_title("One-Sample t-Test")
    ax.set_xlabel("Height (cm)")
    ax.legend(fontsize=8)

    # panel 2: male vs female
    ax = axes[1]
    ax.hist(MALE_HEIGHTS, bins=25, alpha=0.5, density=True,
            edgecolor="white", label="Male")
    ax.hist(FEMALE_HEIGHTS, bins=25, alpha=0.5, density=True,
            edgecolor="white", label="Female")
    ax.set_title("Two-Sample Comparison")
    ax.set_xlabel("Height (cm)")
    ax.legend()

    # panel 3: proportion comparison
    ax = axes[2]
    categories = ["Women\n(59/649)", "Men\n(128/2490)"]
    proportions = [59/649, 128/2490]
    bars = ax.bar(categories, proportions, color=["salmon", "steelblue"],
                  edgecolor="white")
    for bar, prop in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{prop:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("Rejection proportion")
    ax.set_title("Bank Loan Rejection Rates")

    plt.tight_layout()
    plt.savefig("height_weight_hypothesis_test.png", dpi=150)
    plt.show()
    print("\nFigure saved: height_weight_hypothesis_test.png")


if __name__ == "__main__":
    main()
