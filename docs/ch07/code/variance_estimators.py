"""
Variance Estimators
====================
Compare and verify properties of different variance estimators:
  - Naive (1/n): biased MLE
  - Bessel-corrected (1/(n−1)): unbiased
  - MSE-optimal (1/(n+1)): minimum MSE for Normal
  - Known-mean estimator: unbiased with lower variance
  - Degrees-of-freedom intuition
  - Financial: volatility estimation with different divisors
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def bias_verification(sigma=3.0, n_sim=200_000, seed=42):
    """
    Verify the bias formula for the naive variance estimator.
    E[S̃²] = (n−1)/n · σ²  ⟹  Bias = −σ²/n.
    
    Parameters
    ----------
    sigma : float
        True population standard deviation
    n_sim : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]
    
    print("=" * 65)
    print(f"Naive Variance Estimator Bias (True σ² = {sigma2})")
    print("=" * 65)
    print(f"\n{'n':>6} {'E[S̃²]':>12} {'(n−1)/n·σ²':>14} {'Bias':>10} {'−σ²/n':>10}")
    print("-" * 55)
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        s_tilde2 = np.var(samples, axis=1, ddof=0)
        print(f"{n:>6} {s_tilde2.mean():>12.4f} {(n-1)/n*sigma2:>14.4f} "
              f"{s_tilde2.mean()-sigma2:>10.4f} {-sigma2/n:>10.4f}")


def three_estimators_mse(sigma=3.0, n_sim=100_000, seed=42):
    """
    Compare MSE of three variance estimators: 1/n, 1/(n−1), 1/(n+1).
    Show that the biased estimators can have lower MSE.
    
    Parameters
    ----------
    sigma : float
        True standard deviation
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sigma2 = sigma**2
    sigma4 = sigma**4
    sample_sizes = [3, 5, 10, 20, 50, 100]
    
    print("\n" + "=" * 80)
    print(f"MSE Comparison: 1/n vs 1/(n−1) vs 1/(n+1)  (True σ² = {sigma2})")
    print("=" * 80)
    
    all_mse = {d: [] for d in ['1/n', '1/(n-1)', '1/(n+1)']}
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        ss = np.sum((samples - samples.mean(axis=1, keepdims=True))**2, axis=1)
        
        estimators = {'1/n': ss/n, '1/(n-1)': ss/(n-1), '1/(n+1)': ss/(n+1)}
        
        print(f"\nn = {n}:")
        print(f"  {'Divisor':<10} {'Bias':>10} {'Var':>12} {'MSE':>12} {'MSE(theory)':>14}")
        print("  " + "-" * 62)
        
        theory_mse = {
            '1/n': (2*n-1)/n**2 * sigma4,
            '1/(n-1)': 2/(n-1) * sigma4,
            '1/(n+1)': (2*(n-1)+4)/(n+1)**2 * sigma4,
        }
        
        for name, est in estimators.items():
            bias = est.mean() - sigma2
            var = est.var()
            mse = np.mean((est - sigma2)**2)
            all_mse[name].append(mse)
            print(f"  {name:<10} {bias:>10.4f} {var:>12.4f} {mse:>12.4f} "
                  f"{theory_mse[name]:>14.4f}")
    
    # Plot theoretical MSE curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = np.arange(3, 101)
    ax.plot(ns, (2*ns-1)/ns**2 * sigma4, 'b-', lw=2, label='1/n (naive / MLE)')
    ax.plot(ns, 2/(ns-1) * sigma4, 'r-', lw=2, label="1/(n−1) (Bessel's)")
    ax.plot(ns, (2*(ns-1)+4)/(ns+1)**2 * sigma4, 'g-', lw=2, label='1/(n+1) (MSE-optimal)')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE of Variance Estimators (Normal Population)', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variance_estimators_mse.png', dpi=150, bbox_inches='tight')
    plt.show()


def degrees_of_freedom_intuition(seed=42):
    """
    Demonstrate why dividing by n−1: the constraint Σ(Xᵢ − X̄) = 0
    means only n−1 deviations are free.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu, sigma, n = 5.0, 2.0, 5
    sample = rng.normal(mu, sigma, n)
    x_bar = sample.mean()
    
    dev_xbar = sample - x_bar
    dev_mu = sample - mu
    
    print("\n" + "=" * 60)
    print("Degrees of Freedom Intuition")
    print(f"Sample: {np.round(sample, 3)}")
    print(f"X̄ = {x_bar:.3f}, True μ = {mu}")
    print("=" * 60)
    print(f"\n{'i':>3} {'Xᵢ':>8} {'Xᵢ−X̄':>10} {'Xᵢ−μ':>10}")
    print("-" * 35)
    for i in range(n):
        print(f"{i+1:>3} {sample[i]:>8.3f} {dev_xbar[i]:>10.3f} {dev_mu[i]:>10.3f}")
    print(f"{'Sum':>3} {'':>8} {sum(dev_xbar):>10.6f} {sum(dev_mu):>10.3f}")
    
    print(f"\n• Σ(Xᵢ − X̄) = 0 always → only n−1 = {n-1} free deviations")
    print(f"• SS(X̄) = {np.sum(dev_xbar**2):.3f}")
    print(f"• SS(μ)  = {np.sum(dev_mu**2):.3f}")
    print(f"• Difference = n·(X̄−μ)² = {n*(x_bar-mu)**2:.3f}")


def known_vs_unknown_mean(sigma=3.0, n_sim=100_000, seed=42):
    """
    Compare variance estimation when the true mean μ is known vs unknown.
    
    Parameters
    ----------
    sigma : float
        True standard deviation
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    mu, sigma2 = 5.0, sigma**2
    sample_sizes = [5, 10, 25, 50, 100]
    
    print("\n" + "=" * 55)
    print("Known vs Unknown Mean: Impact on MSE")
    print("=" * 55)
    print(f"\n{'n':>6} {'MSE(known μ)':>14} {'MSE(unknown)':>14} {'Ratio':>8}")
    print("-" * 46)
    
    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_sim, n))
        est_known = np.mean((samples - mu)**2, axis=1)
        est_unknown = np.var(samples, axis=1, ddof=0)
        mse_k = np.mean((est_known - sigma2)**2)
        mse_u = np.mean((est_unknown - sigma2)**2)
        print(f"{n:>6} {mse_k:>14.4f} {mse_u:>14.4f} {mse_u/mse_k:>8.3f}")


def volatility_estimation_finance(seed=42):
    """
    Compare divisor choice for daily return volatility estimation.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    annual_vol = 0.20
    daily_vol = annual_vol / np.sqrt(252)
    daily_mu = 0.08 / 252
    n_sim = 30_000
    
    windows = [5, 10, 21, 63, 126, 252]
    
    print("\n" + "=" * 60)
    print("Volatility Estimation: Divisor Impact")
    print(f"True annual vol: {annual_vol*100:.1f}%")
    print("=" * 60)
    print(f"\n{'Window':>8} {'Vol(1/n)':>10} {'Vol(1/n−1)':>12} {'% Diff':>8}")
    print("-" * 42)
    
    for w in windows:
        vol_n, vol_n1 = [], []
        for _ in range(n_sim):
            r = rng.normal(daily_mu, daily_vol, w)
            vol_n.append(np.sqrt(np.var(r, ddof=0) * 252))
            vol_n1.append(np.sqrt(np.var(r, ddof=1) * 252))
        vol_n, vol_n1 = np.mean(vol_n), np.mean(vol_n1)
        print(f"{w:>8} {vol_n*100:>9.2f}% {vol_n1*100:>11.2f}% "
              f"{(vol_n1-vol_n)/vol_n*100:>7.2f}%")


if __name__ == "__main__":
    bias_verification()
    three_estimators_mse()
    degrees_of_freedom_intuition()
    known_vs_unknown_mean()
    volatility_estimation_finance()
