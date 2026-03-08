"""
Bayesian Inference with Beta Conjugate Prior
=============================================
Adapted from ps4ds (Probability and Statistics for Data Science).

For Bernoulli/Binomial data, the Beta distribution is the
conjugate prior:
  Prior:     theta ~ Beta(a, b)
  Data:      k successes out of n trials
  Posterior: theta | data ~ Beta(a + k, b + n - k)

Demonstrates:
1. How different priors lead to different posteriors
2. Sensitivity analysis: informative vs vague priors
3. Posterior probability computation for decision-making
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def bayesian_update(a_prior, b_prior, k, n):
    """Compute posterior Beta parameters after observing k/n."""
    a_post = a_prior + k
    b_post = b_prior + n - k
    return a_post, b_post


def main():
    print("=" * 60)
    print("Bayesian Inference — Beta Conjugate Prior")
    print("=" * 60)

    # Scenario: poll with k successes out of n respondents
    # Inspired by election polling
    n = 581
    k = 281  # e.g. 281 support candidate A out of 581

    print(f"\n  Observed: {k} successes out of {n} trials")
    print(f"  MLE: p_hat = {k/n:.4f}")

    # Different priors: (a, b) pairs
    priors = [
        (1, 1, "Uniform (a=1, b=1)"),
        (5, 5, "Weakly informative (a=5, b=5)"),
        (50, 50, "Moderate prior centered at 0.5"),
        (2, 8, "Prior skewed toward low p"),
        (8, 2, "Prior skewed toward high p"),
        (100, 100, "Strong prior at 0.5"),
    ]

    theta = np.linspace(0, 1, 1000)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    print(f"\n  {'Prior':<35s} {'a_post':>6s} {'b_post':>6s} "
          f"{'Post mean':>10s} {'P(p<0.5)':>10s}")
    print("  " + "-" * 75)

    for idx, (a, b, label) in enumerate(priors):
        a_post, b_post = bayesian_update(a, b, k, n)
        post_mean = a_post / (a_post + b_post)
        p_less_half = beta.cdf(0.5, a_post, b_post)

        print(f"  {label:<35s} {a_post:>6d} {b_post:>6d} "
              f"{post_mean:>10.4f} {p_less_half:>10.4f}")

        ax = axes_flat[idx]

        # Prior
        ax.plot(theta, beta.pdf(theta, a, b), "b--", lw=2,
                label="Prior")
        # Posterior
        ax.plot(theta, beta.pdf(theta, a_post, b_post), "r-", lw=2.5,
                label="Posterior")
        # Shade P(p < 0.5)
        mask = theta <= 0.5
        ax.fill_between(theta[mask],
                        beta.pdf(theta[mask], a_post, b_post),
                        alpha=0.2, color="blue",
                        label=f"P(p<0.5) = {p_less_half:.3f}")
        # MLE
        ax.axvline(k / n, color="green", linestyle=":", lw=1.5,
                   label=f"MLE = {k/n:.3f}")

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("theta")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0.35, 0.65)

    plt.suptitle(f"Bayesian Update: {k} successes in {n} trials\n"
                 f"Prior Beta(a,b) -> Posterior Beta(a+{k}, b+{n-k})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("bayesian_beta_conjugate.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("\nFigure saved: bayesian_beta_conjugate.png")


if __name__ == "__main__":
    main()
