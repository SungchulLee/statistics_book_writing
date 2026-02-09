"""
Bessel's Correction
====================
Verify and demonstrate Bessel's correction properties:
  - Unbiasedness of S² across distributions
  - Chi-squared distribution of (n−1)S²/σ²
  - Independence of X̄ and S² (Normal only)
  - Standard deviation bias (Jensen's inequality)
  - Software defaults: ddof pitfalls
  - Financial: tracking error estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma as gamma_func


def unbiasedness_across_distributions(n_sim=200_000, seed=42):
    """
    Verify E[S²] = σ² for Normal and non-Normal populations.
    
    Parameters
    ----------
    n_sim : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sigma = 4.0
    sigma2 = sigma**2
    n = 20
    
    print("=" * 65)
    print(f"Bessel's Correction: E[S²] = σ² (n={n})")
    print("=" * 65)
    
    distributions = {
        f'Normal(0, {sigma2})': (lambda: rng.normal(0, sigma, n), sigma2),
        f'Exp(scale={sigma})': (lambda: rng.exponential(sigma, n), sigma2),
        f'Uniform': (lambda: rng.uniform(0, 2*sigma*np.sqrt(3), n), sigma2),
        f'Chi²(df={int(sigma2)})': (lambda: rng.chisquare(int(sigma2), n), 2*sigma2),
    }
    
    print(f"\n{'Distribution':<25} {'True σ²':>8} {'E[S²]':>10} {'Bias':>10}")
    print("-" * 57)
    
    for name, (sampler, true_var) in distributions.items():
        s2_vals = np.array([np.var(sampler(), ddof=1) for _ in range(n_sim)])
        print(f"{name:<25} {true_var:>8.2f} {s2_vals.mean():>10.4f} "
              f"{s2_vals.mean()-true_var:>10.4f}")


def chi_squared_verification(sigma=3.0, n_sim=100_000, seed=42):
    """
    Verify (n−1)S²/σ² ~ χ²(n−1) for Normal data.
    
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
    sample_sizes = [5, 10, 25, 50]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, n in zip(axes.flat, sample_sizes):
        samples = rng.normal(0, sigma, (n_sim, n))
        s2 = np.var(samples, axis=1, ddof=1)
        chi2_vals = (n - 1) * s2 / sigma**2
        
        ax.hist(chi2_vals, bins=80, density=True, alpha=0.6,
                color='steelblue', edgecolor='white', label='Empirical')
        x = np.linspace(0, stats.chi2.ppf(0.999, n-1), 200)
        ax.plot(x, stats.chi2.pdf(x, n-1), 'r-', linewidth=2,
                label=f'χ²(df={n-1})')
        ax.set_title(f'n = {n}', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    
    plt.suptitle('(n−1)S²/σ² ~ χ²(n−1)  for Normal Data',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bessels_chi_squared.png', dpi=150, bbox_inches='tight')
    plt.show()


def independence_xbar_s2(sigma=3.0, n_sim=100_000, seed=42):
    """
    Show X̄ and S² are independent for Normal data (but not others).
    
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
    n = 20
    
    # Normal
    samp_n = rng.normal(5, sigma, (n_sim, n))
    xbar_n = samp_n.mean(axis=1)
    s2_n = np.var(samp_n, axis=1, ddof=1)
    corr_n = np.corrcoef(xbar_n, s2_n)[0, 1]
    
    # Exponential
    samp_e = rng.exponential(sigma, (n_sim, n))
    xbar_e = samp_e.mean(axis=1)
    s2_e = np.var(samp_e, axis=1, ddof=1)
    corr_e = np.corrcoef(xbar_e, s2_e)[0, 1]
    
    print("\n" + "=" * 60)
    print("Independence of X̄ and S² (Cochran's Theorem)")
    print("=" * 60)
    print(f"  Normal:      Corr(X̄, S²) = {corr_n:.6f}  (≈ 0)")
    print(f"  Exponential: Corr(X̄, S²) = {corr_e:.6f}  (≠ 0)")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(xbar_n[:2000], s2_n[:2000], s=5, alpha=0.2)
    axes[0].set_title(f'Normal: ρ = {corr_n:.4f}', fontsize=12)
    axes[0].set_xlabel('X̄'); axes[0].set_ylabel('S²'); axes[0].grid(True, alpha=0.3)
    axes[1].scatter(xbar_e[:2000], s2_e[:2000], s=5, alpha=0.2)
    axes[1].set_title(f'Exponential: ρ = {corr_e:.4f}', fontsize=12)
    axes[1].set_xlabel('X̄'); axes[1].set_ylabel('S²'); axes[1].grid(True, alpha=0.3)
    plt.suptitle('Independence of X̄ and S²: Normal Only',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bessels_independence.png', dpi=150, bbox_inches='tight')
    plt.show()


def std_deviation_bias(sigma=3.0, n_sim=200_000, seed=42):
    """
    Show E[S] < σ by Jensen's inequality and compute c₄ correction.
    
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
    sample_sizes = [3, 5, 10, 20, 50, 100, 500]
    
    print("\n" + "=" * 65)
    print(f"E[S] < σ (Jensen's Inequality)  —  True σ = {sigma}")
    print("=" * 65)
    print(f"\n{'n':>6} {'E[S]':>10} {'σ':>8} {'Bias':>10} {'c₄':>8} {'E[S/c₄]':>10}")
    print("-" * 55)
    
    for n in sample_sizes:
        samples = rng.normal(0, sigma, (n_sim, n))
        s = np.std(samples, axis=1, ddof=1)
        c4 = np.sqrt(2 / (n - 1)) * gamma_func(n / 2) / gamma_func((n - 1) / 2)
        print(f"{n:>6} {s.mean():>10.4f} {sigma:>8.4f} {s.mean()-sigma:>10.4f} "
              f"{c4:>8.4f} {(s/c4).mean():>10.4f}")


def software_defaults_pitfall():
    """Demonstrate different defaults for var() across software."""
    data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    n = len(data)
    
    print("\n" + "=" * 60)
    print("Software Defaults Pitfall")
    print(f"Data: {data}")
    print("=" * 60)
    print(f"\nnp.var(data)          = {np.var(data):.4f}  ← divides by n={n}  (BIASED)")
    print(f"np.var(data, ddof=1)  = {np.var(data, ddof=1):.4f}  ← divides by n−1={n-1}  (UNBIASED)")
    print(f"\n⚠ Always specify ddof=1 for sample variance in NumPy!")


def tracking_error_estimation(seed=42):
    """
    Estimate portfolio tracking error with Bessel's correction.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    n_months = 36
    te_true_monthly = 0.01
    te_true_annual = te_true_monthly * np.sqrt(12)
    n_sim = 50_000
    
    print("\n" + "=" * 60)
    print("Tracking Error Estimation")
    print(f"True monthly TE: {te_true_monthly*100:.2f}%")
    print(f"True annualized TE: {te_true_annual*100:.2f}%")
    print("=" * 60)
    
    te_n, te_n1 = [], []
    for _ in range(n_sim):
        excess = rng.normal(0.002, te_true_monthly, n_months)
        te_n.append(np.std(excess, ddof=0) * np.sqrt(12))
        te_n1.append(np.std(excess, ddof=1) * np.sqrt(12))
    
    te_n, te_n1 = np.array(te_n), np.array(te_n1)
    print(f"\n{'Estimator':<18} {'Mean':>10} {'Bias':>10} {'RMSE':>10}")
    print("-" * 50)
    for name, est in [('ddof=0', te_n), ('ddof=1', te_n1)]:
        print(f"{name:<18} {est.mean()*100:>9.3f}% "
              f"{(est.mean()-te_true_annual)*100:>9.3f}% "
              f"{np.sqrt(np.mean((est-te_true_annual)**2))*100:>9.3f}%")
    print(f"{'True':<18} {te_true_annual*100:>9.3f}%")


if __name__ == "__main__":
    unbiasedness_across_distributions()
    chi_squared_verification()
    independence_xbar_s2()
    std_deviation_bias()
    software_defaults_pitfall()
    tracking_error_estimation()
