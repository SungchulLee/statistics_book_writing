"""
Coin Toss Simulation — Hypothesis Testing via Simulation
=========================================================
Adapted from intro2stats "Warm-up" notebook.

Instead of using a formula for the binomial distribution,
we simulate the coin-toss experiment many times and count
how often we observe 24 or more heads in 30 tosses.

If the fraction is below 5 %, we reject the null hypothesis
that the coin is fair.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ── Parameters ──────────────────────────────────────────────
TOTAL_TOSSES = 30
OBSERVED_HEADS = 24
PROB_HEAD_FAIR = 0.5
NUM_SIMULATIONS = 100_000


# ── Single experiment ───────────────────────────────────────
def single_experiment(n_tosses=TOTAL_TOSSES, p=PROB_HEAD_FAIR):
    """Simulate one round of n_tosses fair-coin flips; return head count."""
    return np.random.binomial(n_tosses, p)


# ── Repeated simulation ────────────────────────────────────
def simulate_coin_tosses(n_simulations=NUM_SIMULATIONS,
                         n_tosses=TOTAL_TOSSES,
                         p=PROB_HEAD_FAIR):
    """
    Repeat the coin-toss experiment *n_simulations* times.
    Returns an array of head counts, one per experiment.
    """
    return np.random.binomial(n_tosses, p, size=n_simulations)


# ── Main ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Coin Toss Simulation — Is the Coin Fair?")
    print("=" * 60)

    # --- single experiment demo ---
    one_trial = single_experiment()
    print(f"\nSingle trial: {one_trial} heads in {TOTAL_TOSSES} tosses")

    # --- repeated simulation ---
    head_counts = simulate_coin_tosses()

    extreme = np.sum(head_counts >= OBSERVED_HEADS)
    pct = extreme / NUM_SIMULATIONS * 100

    print(f"\nSimulation ({NUM_SIMULATIONS:,} runs):")
    print(f"  Times with >= {OBSERVED_HEADS} heads: {extreme:,}")
    print(f"  Percentage: {pct:.4f} %")

    if pct < 5:
        print("\n  => p < 5 %.  Reject H0: the coin is likely biased.")
    else:
        print("\n  => p >= 5 %.  Fail to reject H0: no evidence of bias.")

    # --- exact binomial p-value for comparison ---
    from scipy.stats import binom

# =============================================================================
# Main
# =============================================================================
    p_exact = 1 - binom.cdf(OBSERVED_HEADS - 1, TOTAL_TOSSES, PROB_HEAD_FAIR)
    print(f"\n  Exact binomial P(X >= {OBSERVED_HEADS}): {p_exact:.6f}")

    # --- visualisation ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, TOTAL_TOSSES + 2) - 0.5
    ax.hist(head_counts, bins=bins, edgecolor="white", alpha=0.7,
            label="Simulated head counts")
    ax.axvline(OBSERVED_HEADS, color="red", linestyle="--", linewidth=2,
               label=f"Observed = {OBSERVED_HEADS}")
    ax.set_xlabel("Number of heads")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Coin Toss Simulation ({NUM_SIMULATIONS:,} runs, "
                 f"n = {TOTAL_TOSSES})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("coin_toss_simulation.png", dpi=150)
    plt.show()
    print("\nFigure saved: coin_toss_simulation.png")


if __name__ == "__main__":
    main()
