"""
Gambler's Paradox — When the Law of Large Numbers Fails
========================================================
Adapted from ps4ds (Probability and Statistics for Data Science).

A gambler plays a doubling game: flip a fair coin repeatedly,
and receive 2^k dollars if the first heads appears on flip k.
The expected winnings per round are infinite (St. Petersburg paradox),
so the sample mean of i.i.d. rounds does NOT converge — the LLN
requires a finite mean.

Demonstrates:
1. Divergence of the sample mean on a log-log scale
2. Comparison with a finite-mean geometric game (converges)
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


# ── 1. St. Petersburg game (infinite mean) ───────────────────
def st_petersburg_sample_means(n_max=10_000, tries=100, n_grid=200):
    """
    Each round: flip fair coins; first heads on flip k → win 2^k.
    E[winnings] = sum_{k=1}^inf 2^k * (1/2)^k = infinity.
    """
    n_vals = np.unique(np.logspace(1, np.log10(n_max), n_grid).astype(int))
    results = []
    for n in n_vals:
        # Geometric(p=0.5) gives the trial number of first success
        flips = np.random.geometric(0.5, size=(n, tries))
        winnings = 2.0 ** flips
        means = winnings.mean(axis=0)
        results.append((n, means))
    return results


# ── 2. Bounded game (finite mean, LLN holds) ────────────────
def bounded_game_sample_means(n_max=10_000, tries=100, n_grid=200):
    """
    Same doubling game but capped at 2^10 = 1024 dollars.
    Now E[X] is finite, so the LLN guarantees convergence.
    """
    n_vals = np.unique(np.logspace(1, np.log10(n_max), n_grid).astype(int))
    results = []
    for n in n_vals:
        flips = np.random.geometric(0.5, size=(n, tries))
        winnings = np.minimum(2.0 ** flips, 1024.0)
        means = winnings.mean(axis=0)
        results.append((n, means))
    return results


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Gambler's Paradox — LLN Failure with Infinite Mean")
    print("=" * 60)

    print("\nSt. Petersburg game: win = 2^k, E[win] = infinity")
    print("Bounded game: win = min(2^k, 1024), E[win] < infinity")

    infinite_results = st_petersburg_sample_means()
    bounded_results = bounded_game_sample_means()

    # ── Visualisation ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: infinite mean — sample mean diverges
    ax = axes[0]
    for n, means in infinite_results:
        ax.loglog(n * np.ones(len(means)), means, ".", color="black",
                  ms=2, alpha=0.5)
    ax.set_xlabel("n (number of rounds)", fontsize=11)
    ax.set_ylabel("Sample mean of winnings", fontsize=11)
    ax.set_title("St. Petersburg Game (E[X] = inf)\nSample mean does NOT converge",
                 fontsize=11)

    # Panel 2: finite mean — sample mean converges
    ax = axes[1]
    for n, means in bounded_results:
        ax.semilogx(n * np.ones(len(means)), means, ".", color="steelblue",
                    ms=2, alpha=0.5)
    # theoretical mean
    k = np.arange(1, 11)
    true_mean = np.sum(np.minimum(2.0 ** k, 1024) * 0.5 ** k)
    # add remaining mass for k >= 11
    true_mean += 1024 * (0.5 ** 10)
    ax.axhline(true_mean, color="red", linestyle="--", lw=2,
               label=f"E[X] = {true_mean:.1f}")
    ax.set_xlabel("n (number of rounds)", fontsize=11)
    ax.set_ylabel("Sample mean of winnings", fontsize=11)
    ax.set_title("Bounded Game (E[X] < inf)\nSample mean converges (LLN)",
                 fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("gambler_paradox_lln.png", dpi=150)
    plt.show()
    print("\nFigure saved: gambler_paradox_lln.png")


if __name__ == "__main__":
    main()
