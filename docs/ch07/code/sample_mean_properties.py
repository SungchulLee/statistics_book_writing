"""
Sample Mean Properties
======================
Verify and demonstrate core properties of the sample mean estimator:
  - Unbiasedness: E[X̄] = μ for any distribution
  - Variance: Var(X̄) = σ²/n
  - MSE = Var (since bias = 0)
  - Standard error convergence: SE = σ/√n
  - Efficiency comparison: mean vs median vs trimmed mean
  - Weighted mean with inverse-variance weighting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def verify_unbiasedness(mu=10.0, sigma=3.0, n=20, n_sim=100_000, seed=42):
    """
    Verify E[X̄] = μ across multiple distributions.
    
    Parameters
    ----------
    mu : float
        Target mean (used for Normal; other distributions use their own)
    sigma : float
        Standard deviation for Normal
    n : int
        Sample size
    n_sim : int
        Number of Monte Carlo simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    
    distributions = {
        f'Normal({mu}, {sigma}²)': (lambda: rng.normal(mu, sigma, n), mu),
        'Exp(λ=0.5)': (lambda: rng.exponential(2, n), 2.0),
        'Poisson(3.7)': (lambda: rng.poisson(3.7, n), 3.7),
        'Uniform(2, 8)': (lambda: rng.uniform(2, 8, n), 5.0),
        'Bernoulli(0.4)': (lambda: rng.binomial(1, 0.4, n), 0.4),
        'Chi²(df=5)': (lambda: rng.chisquare(5, n), 5.0),
    }
    
    print("=" * 60)
    print(f"Unbiasedness of X̄ (n={n}, simulations={n_sim:,})")
    print("=" * 60)
    print(f"\n{'Distribution':<22} {'True μ':>8} {'E[X̄]':>10} {'Bias':>12}")
    print("-" * 55)
    
    for name, (sampler, true_mu) in distributions.items():
        estimates = np.array([sampler().mean() for _ in range(n_sim)])
        bias = estimates.mean() - true_mu
        print(f"{name:<22} {true_mu:>8.4f} {estimates.mean():>10.4f} {bias:>12.6f}")
    
    print("\n✓ All biases ≈ 0 (within simulation noise).")


def verify_variance_and_mse(mu=10.0, sigma=3.0, n_sim=100_000, seed=42):
    """
    Verify Var(X̄) = σ²/n and MSE = Var for the sample mean.
    
    Parameters
    ----------
    mu, sigma : float
        Population mean and standard deviation
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 25, 50, 100, 500]
    
    print("\n" + "=" * 70)
    print("Variance and MSE of X̄")
    print(f"Population: μ = {mu}, σ = {sigma}")
    print("=" * 70)
    print(f"\n{'n':>6} {'Var(X̄)':>12} {'σ²/n':>12} {'MSE':>12} {'SE':>10} {'σ/√n':>10}")
    print("-" * 60)
    
    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_sim, n))
        x_bars = samples.mean(axis=1)
        
        var_xbar = x_bars.var(ddof=0)
        theory_var = sigma**2 / n
        mse = np.mean((x_bars - mu)**2)
        se = x_bars.std(ddof=0)
        theory_se = sigma / np.sqrt(n)
        
        print(f"{n:>6} {var_xbar:>12.6f} {theory_var:>12.6f} {mse:>12.6f} "
              f"{se:>10.6f} {theory_se:>10.6f}")


def efficiency_comparison(n=30, n_sim=50_000, seed=42):
    """
    Compare MSE of mean vs median vs trimmed mean across distributions.
    
    Parameters
    ----------
    n : int
        Sample size
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    
    Returns
    -------
    dict of results by distribution
    """
    rng = np.random.default_rng(seed)
    
    distributions = {
        'Normal(0,1)': lambda: rng.standard_normal(n),
        't(df=3)': lambda: rng.standard_t(3, n),
        't(df=5)': lambda: rng.standard_t(5, n),
        'Contaminated Normal': lambda: np.where(
            rng.uniform(0, 1, n) < 0.1,
            rng.normal(0, 10, n),
            rng.standard_normal(n)),
    }
    
    print("\n" + "=" * 72)
    print(f"Efficiency Comparison: Mean vs Alternatives (n = {n})")
    print("=" * 72)
    
    results = {}
    for dist_name, sampler in distributions.items():
        est = {'Mean': [], 'Median': [], 'Trim10%': [], 'Trim20%': []}
        for _ in range(n_sim):
            s = sampler()
            est['Mean'].append(np.mean(s))
            est['Median'].append(np.median(s))
            est['Trim10%'].append(stats.trim_mean(s, 0.1))
            est['Trim20%'].append(stats.trim_mean(s, 0.2))
        
        mse_mean = np.mean(np.array(est['Mean'])**2)
        print(f"\n{dist_name}:")
        print(f"  {'Estimator':<12} {'MSE':>10} {'Rel. Eff.':>12}")
        print("  " + "-" * 36)
        for name, vals in est.items():
            mse = np.mean(np.array(vals)**2)
            re = mse_mean / mse
            print(f"  {name:<12} {mse:>10.6f} {re:>12.4f}")
        results[dist_name] = {k: np.mean(np.array(v)**2) for k, v in est.items()}
    
    return results


def weighted_mean_demo(mu=5.0, n_sim=50_000, seed=42):
    """
    Demonstrate inverse-variance weighted mean vs unweighted mean.
    
    Parameters
    ----------
    mu : float
        True population mean
    n_sim : int
        Number of simulations
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sigmas = np.array([1.0, 2.0, 5.0, 10.0, 0.5])
    
    optimal_weights = 1 / sigmas**2
    optimal_weights /= optimal_weights.sum()
    
    unweighted, weighted = [], []
    for _ in range(n_sim):
        obs = rng.normal(mu, sigmas)
        unweighted.append(obs.mean())
        weighted.append(np.sum(optimal_weights * obs))
    
    unweighted = np.array(unweighted)
    weighted = np.array(weighted)
    
    print("\n" + "=" * 60)
    print("Weighted vs Unweighted Mean")
    print(f"True μ = {mu}, Observation SDs: {sigmas}")
    print("=" * 60)
    print(f"\n{'Estimator':<20} {'Mean':>8} {'Var':>10} {'MSE':>10}")
    print("-" * 50)
    print(f"{'Unweighted':<20} {unweighted.mean():>8.4f} "
          f"{unweighted.var():>10.6f} {np.mean((unweighted-mu)**2):>10.6f}")
    print(f"{'IV-Weighted':<20} {weighted.mean():>8.4f} "
          f"{weighted.var():>10.6f} {np.mean((weighted-mu)**2):>10.6f}")
    print(f"\nOptimal weights: {np.round(optimal_weights, 4)}")
    print(f"Variance reduction: {(1 - weighted.var()/unweighted.var())*100:.1f}%")


def convergence_rate_plot(mu=5.0, sigma=3.0, n_sim=50_000, seed=42):
    """
    Plot SE vs n on log-log scale to verify 1/√n rate.
    
    Parameters
    ----------
    mu, sigma : float
        Population parameters
    n_sim : int
        Number of simulations per sample size
    seed : int
        Random seed
    """
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    empirical_se = []
    for n in sample_sizes:
        estimates = np.array([rng.normal(mu, sigma, n).mean() for _ in range(n_sim)])
        empirical_se.append(estimates.std())
    
    theoretical_se = [sigma / np.sqrt(n) for n in sample_sizes]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(sample_sizes, empirical_se, 'bo-', label='Empirical SE', markersize=6)
    ax.loglog(sample_sizes, theoretical_se, 'r--', label='σ/√n', linewidth=2)
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.set_title('Convergence Rate of Sample Mean: SE = σ/√n', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('sample_mean_convergence_rate.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    verify_unbiasedness()
    verify_variance_and_mse()
    efficiency_comparison()
    weighted_mean_demo()
    convergence_rate_plot()
