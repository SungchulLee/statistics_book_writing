#!/usr/bin/env python3
"""
Q-Q Plot Analysis of Financial Returns
======================================

Demonstrates Q-Q plots for detecting non-normality in financial data.
Uses Student's t distribution (heavy-tailed) to simulate realistic stock returns.

Key findings:
- Real asset returns exhibit heavier tails than normal distribution
- Q-Q plots show S-shaped pattern (bending up/down at extremes)
- Consequence: Normal-based models underestimate tail risk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simulate_returns(dist_type='normal', n=2000, seed=42):
    """
    Simulate financial returns.

    Parameters:
    -----------
    dist_type : str
        'normal' or 'heavy_tailed'
    n : int
        Number of returns
    seed : int
        Random seed

    Returns:
    --------
    np.ndarray : Daily log-returns
    """
    np.random.seed(seed)

    if dist_type == 'normal':
        # Normal returns: typical mean ~0.05%, daily volatility ~1.5%
        returns = np.random.normal(loc=0.0005, scale=0.015, size=n)
    elif dist_type == 'heavy_tailed':
        # Student's t returns: heavier tails than normal
        # df=6 gives realistic excess kurtosis ~3-4
        returns = stats.t.rvs(df=6, loc=0.0005, scale=0.015, size=n)
    else:
        raise ValueError("dist_type must be 'normal' or 'heavy_tailed'")

    return returns


def compute_statistics(returns, dist_type):
    """Compute and print distribution statistics."""
    mean = returns.mean()
    std = returns.std()
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis

    print(f"\n{dist_type.upper()} RETURNS:")
    print(f"  Mean:            {mean:.6f}")
    print(f"  Std Dev:         {std:.6f}")
    print(f"  Skewness:        {skew:.4f}")
    print(f"  Excess Kurtosis: {kurt:.4f}")

    # Normality tests
    _, p_ks = stats.kstest(returns, 'norm', args=(mean, std))
    _, p_jb = stats.jarque_bera(returns)

    print(f"  Jarque-Bera p-value: {p_jb:.6f}")
    print(f"  Interpretation: {'Reject normality' if p_jb < 0.05 else 'Cannot reject normality'}")


def plot_qq_comparison():
    """
    Create comprehensive comparison of normal vs heavy-tailed returns.
    Shows: histograms, density overlays, and Q-Q plots.
    """
    normal_returns = simulate_returns('normal', n=2000)
    heavy_tailed_returns = simulate_returns('heavy_tailed', n=2000)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # ===== Row 1: Histograms =====
    # Normal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(normal_returns, bins=50, alpha=0.6, color='blue', density=True, edgecolor='black')
    x = np.linspace(normal_returns.min(), normal_returns.max(), 200)
    ax1.plot(x, stats.norm.pdf(x, normal_returns.mean(), normal_returns.std()),
             'b-', linewidth=2, label='Normal PDF')
    ax1.set_title('Normal Returns: Histogram', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.spines[['top', 'right']].set_visible(False)

    # Heavy-tailed
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(heavy_tailed_returns, bins=50, alpha=0.6, color='red', density=True, edgecolor='black')
    x = np.linspace(heavy_tailed_returns.min(), heavy_tailed_returns.max(), 200)
    ax2.plot(x, stats.t.pdf(x, df=6, loc=heavy_tailed_returns.mean(),
                            scale=heavy_tailed_returns.std()),
             'r-', linewidth=2, label="Student's t PDF")
    ax2.set_title("Heavy-Tailed Returns: Histogram", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.spines[['top', 'right']].set_visible(False)

    # ===== Row 2: Histogram overlays for comparison =====
    ax3 = fig.add_subplot(gs[1, :])
    ax3.hist(normal_returns, bins=50, alpha=0.5, color='blue', density=True,
             edgecolor='black', label='Normal Returns')
    ax3.hist(heavy_tailed_returns, bins=50, alpha=0.5, color='red', density=True,
             edgecolor='black', label='Heavy-Tailed Returns')
    x = np.linspace(min(normal_returns.min(), heavy_tailed_returns.min()),
                    max(normal_returns.max(), heavy_tailed_returns.max()), 200)
    ax3.plot(x, stats.norm.pdf(x, normal_returns.mean(), normal_returns.std()),
             'b-', linewidth=2.5, label='Normal PDF')
    ax3.plot(x, stats.t.pdf(x, df=6, loc=heavy_tailed_returns.mean(),
                            scale=heavy_tailed_returns.std()),
             'r-', linewidth=2.5, label="Student's t PDF")
    ax3.set_title('Distribution Comparison: Notice Heavier Tails in Red', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Density')
    ax3.legend(loc='upper right')
    ax3.spines[['top', 'right']].set_visible(False)

    # ===== Row 3: Q-Q Plots =====
    # Normal data Q-Q plot
    ax4 = fig.add_subplot(gs[2, 0])
    stats.probplot(normal_returns, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot: Normal Returns\n(Points on diagonal = perfect fit)',
                  fontsize=11, fontweight='bold')
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.grid(True, alpha=0.3)

    # Heavy-tailed data Q-Q plot
    ax5 = fig.add_subplot(gs[2, 1])
    stats.probplot(heavy_tailed_returns, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot: Heavy-Tailed Returns\n(S-shaped pattern = non-normal)',
                  fontsize=11, fontweight='bold')
    ax5.spines[['top', 'right']].set_visible(False)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Q-Q Plots: Detecting Non-Normality in Asset Returns',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.show()


def analyze_tail_behavior(heavy_tailed_returns):
    """
    Analyze tail behavior and risk implications.

    Parameters:
    -----------
    heavy_tailed_returns : np.ndarray
        Return series
    """
    print("\n" + "=" * 80)
    print("TAIL BEHAVIOR ANALYSIS")
    print("=" * 80)

    # Compute tail quantiles
    percentiles = [0.1, 0.5, 1, 5]
    print("\nLeft Tail Quantiles (worst-case scenarios):")
    print(f"  {'Percentile':<12} {'Return':<12} {'Interpretation'}")
    print("-" * 50)

    for p in percentiles:
        q = np.percentile(heavy_tailed_returns, p)
        print(f"  {p:>6.1f}%     {q:>10.4f}     {abs(q)*100:>6.2f}% loss")

    # Compare with normal expectation
    normal_returns = simulate_returns('normal', n=5000)
    print("\n\nNormal Distribution (for comparison):")
    print(f"  {'Percentile':<12} {'Return':<12} {'Interpretation'}")
    print("-" * 50)

    for p in percentiles:
        q = np.percentile(normal_returns, p)
        print(f"  {p:>6.1f}%     {q:>10.4f}     {abs(q)*100:>6.2f}% loss")

    print("\nKey Insight: Heavy-tailed returns show MUCH worse tails than normal.")
    print("This is why risk models based on normality underestimate crash risk!")


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("Q-Q PLOT ANALYSIS: FINANCIAL RETURNS")
    print("=" * 80)

    # Generate returns
    print("\nGenerating return data...")
    normal_returns = simulate_returns('normal', n=2000)
    heavy_tailed_returns = simulate_returns('heavy_tailed', n=2000)

    # Compute statistics
    compute_statistics(normal_returns, 'normal')
    compute_statistics(heavy_tailed_returns, 'heavy-tailed')

    # Plot Q-Q comparison
    print("\n\nGenerating Q-Q plot comparison...")
    plot_qq_comparison()

    # Tail analysis
    analyze_tail_behavior(heavy_tailed_returns)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Q-Q plot is a powerful diagnostic tool for financial returns:

1. NORMAL RETURNS: Points closely follow the diagonal line
   → Distribution matches normal assumption
   → Standard risk models are reasonable

2. HEAVY-TAILED RETURNS: S-shaped pattern
   → Points bend down in left tail (more extreme losses)
   → Points bend up in right tail (more extreme gains)
   → Standard risk models UNDERESTIMATE tail risk

PRACTICAL IMPLICATIONS:
- Value at Risk (VaR) estimates are too low
- Hedge ratios are too small
- Capital requirements may be insufficient
- Consider using robust risk measures (Expected Shortfall)
- Use bootstrap or non-parametric methods for inference
""")


if __name__ == "__main__":
    main()
