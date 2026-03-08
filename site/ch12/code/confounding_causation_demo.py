"""
Confounding and Causation — Simpson's Paradox in Action
========================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

A confounder C affects both a treatment T and an outcome Y,
creating a spurious correlation between T and Y even if T has
no causal effect on Y.

Demonstrates:
1. Synthetic guinea-pig supplement experiment:
   - Food intake C (confounder) → weight change Y
   - Supplement intake T is correlated with C but has NO causal effect
   - Naive (short) regression of Y on T shows spurious effect
   - Adjusted (long) regression controlling for C removes it

2. Average Treatment Effect (ATE):
   - Naive ATE vs adjusted ATE (conditioning on confounder)
   - Simpson's paradox: effect reverses when controlling for confounder
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def simulate_confounded_data(n=500, rho_tc=0.8):
    """
    Generate data where confounder C causes Y, and C is correlated
    with treatment T, but T has NO causal effect on Y.

    DAG: T <- C -> Y   (T and Y are d-separated given C)
    """
    # Joint distribution of (T, C)
    mean = [0, 0]
    cov = [[1, rho_tc], [rho_tc, 1]]
    tc = np.random.multivariate_normal(mean, cov, n)
    t = tc[:, 0]  # treatment (supplement)
    c = tc[:, 1]  # confounder (food intake)

    # Y depends ONLY on C, not on T
    y = c + np.random.normal(0, 1, n)

    return t, c, y


def compute_regressions(t, c, y):
    """Short regression (Y ~ T) vs long regression (Y ~ T + C)."""
    # Short regression: Y = a + b*T + error
    slope_short, intercept_short, r_short, p_short, se_short = (
        stats.linregress(t, y))

    # Long regression: Y = a + b1*T + b2*C + error (via OLS)
    X = np.column_stack([np.ones(len(t)), t, c])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "short_slope": slope_short,
        "short_r": r_short,
        "short_p": p_short,
        "long_beta_T": beta[1],
        "long_beta_C": beta[2],
        "long_intercept": beta[0],
    }


def simpson_paradox_demo(n=1000):
    """
    Synthetic ATE example: treatment appears harmful overall,
    but is beneficial within each subgroup.
    """
    # Confounder: severity (0 = mild, 1 = severe)
    # Severe patients more likely to get treatment
    severity = np.random.binomial(1, 0.5, n)

    # Treatment assignment depends on severity (confounded)
    p_treat = np.where(severity == 1, 0.7, 0.3)
    treatment = np.random.binomial(1, p_treat)

    # Outcome: severity hurts, treatment helps
    y = (50
         - 20 * severity      # severe cases do worse
         + 5 * treatment       # treatment helps (+5)
         + np.random.normal(0, 5, n))

    # Naive ATE
    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()

    # Adjusted ATE (within each severity group)
    ate_mild = (y[(treatment == 1) & (severity == 0)].mean()
                - y[(treatment == 0) & (severity == 0)].mean())
    ate_severe = (y[(treatment == 1) & (severity == 1)].mean()
                  - y[(treatment == 0) & (severity == 1)].mean())
    p_severe = severity.mean()
    ate_adjusted = (1 - p_severe) * ate_mild + p_severe * ate_severe

    return {
        "severity": severity, "treatment": treatment, "y": y,
        "ate_naive": ate_naive, "ate_mild": ate_mild,
        "ate_severe": ate_severe, "ate_adjusted": ate_adjusted,
        "p_severe": p_severe,
    }


def main():
    print("=" * 60)
    print("Confounding and Causation Demonstration")
    print("=" * 60)

    # ── Part 1: Confounded regression ────────────────────────
    print("\n--- Part 1: Spurious Correlation via Confounding ---")
    rho_vals = [0.8, -0.8, 0.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, rho in enumerate(rho_vals):
        t, c, y = simulate_confounded_data(n=500, rho_tc=rho)
        reg = compute_regressions(t, c, y)

        print(f"\n  rho(T, C) = {rho}:")
        print(f"    Short regression: slope(T→Y) = "
              f"{reg['short_slope']:.3f}, p = {reg['short_p']:.4f}")
        print(f"    Long regression:  beta_T = "
              f"{reg['long_beta_T']:.3f}, beta_C = "
              f"{reg['long_beta_C']:.3f}")
        print(f"    => T has {'NO' if abs(reg['long_beta_T']) < 0.3 else ''}"
              f" causal effect on Y")

        # Top row: scatter T vs Y with short regression line
        ax = axes[0, col]
        ax.scatter(t, y, c=c, cmap="coolwarm", s=15, alpha=0.6,
                   edgecolors="none")
        t_line = np.linspace(t.min(), t.max(), 100)
        ax.plot(t_line, reg["short_slope"] * t_line, "k--", lw=2,
                label=f"slope = {reg['short_slope']:.2f}")
        ax.set_xlabel("Treatment T")
        ax.set_ylabel("Outcome Y")
        ax.set_title(f"Short Regression (rho = {rho})\n"
                     f"Color = confounder C")
        ax.legend(fontsize=9)

        # Bottom row: residuals after controlling for C
        ax = axes[1, col]
        # Partial regression: residualize both T and Y on C
        t_resid = t - stats.linregress(c, t).slope * c
        y_resid = y - stats.linregress(c, y).slope * c
        ax.scatter(t_resid, y_resid, s=15, alpha=0.5, color="gray")
        slope_partial = stats.linregress(t_resid, y_resid).slope
        ax.plot(t_line, slope_partial * t_line, "r--", lw=2,
                label=f"partial slope = {slope_partial:.2f}")
        ax.set_xlabel("T (residualized on C)")
        ax.set_ylabel("Y (residualized on C)")
        ax.set_title(f"Long Regression (controlling for C)")
        ax.legend(fontsize=9)

    plt.suptitle("Confounding: Short vs Long Regression\n"
                 "T has NO causal effect on Y; "
                 "all correlation is due to confounder C",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("confounding_causation_demo_1.png", dpi=150,
                bbox_inches="tight")
    plt.show()

    # ── Part 2: Simpson's Paradox ────────────────────────────
    print("\n--- Part 2: Simpson's Paradox (ATE) ---")
    sp = simpson_paradox_demo()
    print(f"  True treatment effect: +5")
    print(f"  Naive ATE: {sp['ate_naive']:.2f} "
          f"({'WRONG sign!' if sp['ate_naive'] < 0 else 'correct sign'})")
    print(f"  ATE (mild):   {sp['ate_mild']:.2f}")
    print(f"  ATE (severe): {sp['ate_severe']:.2f}")
    print(f"  Adjusted ATE: {sp['ate_adjusted']:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: overall distribution
    ax = axes[0]
    ax.hist(sp["y"][sp["treatment"] == 0], bins=25, alpha=0.5,
            density=True, label="Control", color="steelblue",
            edgecolor="white")
    ax.hist(sp["y"][sp["treatment"] == 1], bins=25, alpha=0.5,
            density=True, label="Treatment", color="coral",
            edgecolor="white")
    ax.axvline(sp["y"][sp["treatment"] == 0].mean(), color="steelblue",
               linestyle="--", lw=2)
    ax.axvline(sp["y"][sp["treatment"] == 1].mean(), color="coral",
               linestyle="--", lw=2)
    ax.set_title(f"Overall: Naive ATE = {sp['ate_naive']:.2f}")
    ax.set_xlabel("Outcome")
    ax.legend(fontsize=9)

    # Panel 2: within subgroups
    ax = axes[1]
    labels = ["Mild\nControl", "Mild\nTreat", "Severe\nControl",
              "Severe\nTreat"]
    means = [
        sp["y"][(sp["treatment"] == 0) & (sp["severity"] == 0)].mean(),
        sp["y"][(sp["treatment"] == 1) & (sp["severity"] == 0)].mean(),
        sp["y"][(sp["treatment"] == 0) & (sp["severity"] == 1)].mean(),
        sp["y"][(sp["treatment"] == 1) & (sp["severity"] == 1)].mean(),
    ]
    colors = ["steelblue", "coral", "steelblue", "coral"]
    bars = ax.bar(labels, means, color=colors, edgecolor="white",
                  alpha=0.8)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5,
                f"{m:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("Mean Outcome")
    ax.set_title("Within Subgroups:\nTreatment helps in BOTH groups")

    # Panel 3: ATE comparison
    ax = axes[2]
    ates = [sp["ate_naive"], sp["ate_mild"], sp["ate_severe"],
            sp["ate_adjusted"]]
    ate_labels = ["Naive\nATE", "ATE\n(mild)", "ATE\n(severe)",
                  "Adjusted\nATE"]
    ate_colors = ["tomato" if a < 0 else "seagreen" for a in ates]
    ax.bar(ate_labels, ates, color=ate_colors, edgecolor="white")
    ax.axhline(5, color="black", linestyle="--", lw=2,
               label="True effect = +5")
    ax.axhline(0, color="gray", linestyle="-", lw=0.5)
    for i, a in enumerate(ates):
        ax.text(i, a + 0.3 * np.sign(a), f"{a:.2f}", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Treatment Effect")
    ax.set_title("Simpson's Paradox:\nNaive ATE is misleading")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("confounding_causation_demo_2.png", dpi=150)
    plt.show()
    print("\nFigures saved: confounding_causation_demo_1.png, "
          "confounding_causation_demo_2.png")


if __name__ == "__main__":
    main()
